#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "openfhe.h"
#include "openfhe/pke/scheme/ckksrns/gen-cryptocontext-ckksrns.h"

using namespace lbcrypto;

// helpers for polynomial evaluation, GELU function, error metrics, and dumping binary data
static inline double polyval(const std::vector<double>& c, double x) {
    double a = 0.0;
    for (auto it = c.rbegin(); it != c.rend(); ++it) a = a * x + *it;
    return a;
}
static inline double gelu(double z) {
    return 0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0)));
}
static double meanAbs(const std::vector<double>& v) {
    double s = 0; for (double e : v) s += std::abs(e); return s / v.size();
}
static void append_bin(const std::string& path, const std::vector<double>& v){
    if (path.empty()) return;
    std::ofstream ofs(path, std::ios::binary | std::ios::app);
    ofs.write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

// coeffs for piecewise polynomial approximation of GELU
const std::vector<double> F0 = { -0.5054031199708174, -0.42226581151983866,
                                 -0.11807612951181953, -0.011034134030615728 };
const std::vector<double> F1 = {  0.008526321541038084, 0.5, 0.3603292692789629, 0.0,
                                 -0.037688200365904236, 0.0, 0.0018067462606141187 };

// make a CKKS context with fixed parameters
CryptoContext<DCRTPoly> makeContext() {
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(4);
    p.SetScalingModSize(50);
    p.SetRingDim(1 << 14);
    p.SetSecurityLevel(HEStd_128_classic);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

// evaluate + dump residuals for one segment of inputs
struct Stats { double mae_ckks, mae_poly; size_t n; };

Stats eval_one_segment(const std::vector<double>& xs,
                       const std::vector<double>& coeff,
                       CryptoContext<DCRTPoly> cc,
                       const KeyPair<DCRTPoly>& kp,
                       const std::string& dump_ckks,
                       const std::string& dump_poly) {
    size_t n = xs.size();
    if (n == 0) return {0,0,0};

    // true + poly
    std::vector<double> poly(n), truth(n);
    for (size_t i = 0; i < n; ++i) {
        poly[i]  = polyval(coeff, xs[i]);
        truth[i] = gelu(xs[i]);
    }

    // CKKS evaluation
    Plaintext ptIn = cc->MakeCKKSPackedPlaintext(xs);
    auto ctIn      = cc->Encrypt(kp.publicKey, ptIn);
    auto ctOut     = cc->EvalPoly(ctIn, coeff);

    Plaintext ptDec;
    cc->Decrypt(kp.secretKey, ctOut, &ptDec);
    ptDec->SetLength(n);
    const auto& dec = ptDec->GetRealPackedValue();

    std::vector<double> ckksErr(n), polyErr(n);
    for (size_t i = 0; i < n; ++i) {
        ckksErr[i] = dec[i]  - poly[i];
        polyErr[i] = poly[i] - truth[i];
    }

    append_bin(dump_ckks, ckksErr);
    append_bin(dump_poly, polyErr);

    return { meanAbs(ckksErr), meanAbs(polyErr), n };
}


int main(int argc, char** argv) try {
    std::string infile = "gelu_inputs.bin";
    std::string dump_ckks = "gelu_ckks.bin";
    std::string dump_poly = "gelu_poly.bin";
    double a = -3.3820;
    double b =  1.0518;

    double cut = -1.95;   // boundary between F0 & F1

    // parse CLI
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--input") && i+1 < argc)      infile = argv[++i];
        else if (!std::strcmp(argv[i], "--dump-ckks") && i+1 < argc) dump_ckks = argv[++i];
        else if (!std::strcmp(argv[i], "--dump-poly") && i+1 < argc) dump_poly = argv[++i];
        else if (!std::strcmp(argv[i], "--a") && i+1 < argc)      a = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--b") && i+1 < argc)      b = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--cut") && i+1 < argc)    cut = std::atof(argv[++i]);
        else                                                      infile = argv[i]; // positional
    }

    // load activations
    std::ifstream f(infile, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << infile << "\n"; return 1; }
    f.seekg(0, std::ios::end);
    size_t n = f.tellg() / sizeof(double);
    f.seekg(0);
    std::vector<double> acts(n);
    f.read(reinterpret_cast<char*>(acts.data()), n*sizeof(double));
    std::cout << "Loaded " << n << " activations\n";

    // filter to [a,b]
    std::vector<double> seg0, seg1;
    seg0.reserve(n/2); seg1.reserve(n/2);
    for (double v : acts) {
        if (v < a || v > b) continue;
        if (v < cut) seg0.push_back(v); else seg1.push_back(v);
    }
    std::cout << "Kept " << (seg0.size()+seg1.size()) << " in ["<<a<<","<<b<<"]\n";
    std::cout << " seg0 (<= "<<cut<<") : " << seg0.size() << "\n";
    std::cout << " seg1 (>  "<<cut<<") : " << seg1.size() << "\n";

    // subsample to fit ringDim/2
    auto cc_tmp   = makeContext();
    size_t slot_n = cc_tmp->GetRingDimension() / 2;
    std::mt19937_64 rng{std::random_device{}()};
    auto subsample = [&](std::vector<double>& v){
        if (v.size() > slot_n) {
            std::shuffle(v.begin(), v.end(), rng);
            v.resize(slot_n);
        }
    };
    subsample(seg0); subsample(seg1);

    // crypto keys
    auto cc = makeContext();
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    // wipe output files (start fresh)
    if (!dump_ckks.empty()) std::ofstream(dump_ckks, std::ios::binary).close();
    if (!dump_poly.empty()) std::ofstream(dump_poly, std::ios::binary).close();

    // evaluate
    auto s0 = eval_one_segment(seg0, F0, cc, kp, dump_ckks, dump_poly);
    auto s1 = eval_one_segment(seg1, F1, cc, kp, dump_ckks, dump_poly);

    std::cout << "\nSegment F0 ["<<a<<","<<cut<<")  n="<<s0.n
              << "  ckks-MAE="<<s0.mae_ckks
              << "  poly-MAE="<<s0.mae_poly << "\n";
    std::cout <<   "Segment F1 ["<<cut<<","<<b<<"]  n="<<s1.n
              << "  ckks-MAE="<<s1.mae_ckks
              << "  poly-MAE="<<s1.mae_poly << "\n";

    return 0;
}
catch(const std::exception& e){
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
