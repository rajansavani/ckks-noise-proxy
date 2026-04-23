#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "openfhe.h"
#include "openfhe/pke/scheme/ckksrns/gen-cryptocontext-ckksrns.h"

using namespace lbcrypto;

// helpers for true function, polynomial evaluation, error metrics, and dumping binary data
static inline double inv_sqrt_true(double z) { return 1.0 / std::sqrt(z); }

static inline double taylor_seed_degree3(double z, double z0){
    const double c0 = 1.0/std::sqrt(z0);
    const double c1 = -1.0/(2.0*std::pow(z0,1.5));
    const double c2 =  3.0/(8.0*std::pow(z0,2.5));
    const double c3 = -5.0/(16.0*std::pow(z0,3.5));
    double dz = z - z0;
    return ((c3*dz + c2)*dz + c1)*dz + c0;
}

static inline double newton_step(double y, double z){
    return y * (3.0 - z * y * y) * 0.5;
}

static double meanAbs(const std::vector<double>& v){
    double s=0; for(double e:v) s += std::abs(e); return s / v.size();
}

static void append_bin(const std::string& path, const std::vector<double>& v){
    if(path.empty()) return;
    std::ofstream(path, std::ios::app | std::ios::binary)
        .write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

template<typename T>
static T quantile(std::vector<T> v, double q){
    if(v.empty()) return T(0);
    size_t idx = std::clamp<size_t>(std::llround(q*(v.size()-1)), 0, v.size()-1);
    std::nth_element(v.begin(), v.begin()+idx, v.end());
    return v[idx];
}

// helper to get seed polynomial coeffs in terms of z (for CKKS evaluation), given the expansion point z0
static std::vector<double> seed_coeff_in_z(double z0){
    // Expand seed(z) = A + B*(z - z0) + C*(z - z0)^2 + D*(z - z0)^3
    double A = 1.0/std::sqrt(z0);
    double B = -1.0/(2.0*std::pow(z0,1.5));
    double C =  3.0/(8.0*std::pow(z0,2.5));
    double D = -5.0/(16.0*std::pow(z0,3.5));

    double c0 = A + B*(-z0) + C*(z0*z0) + D*(-z0*z0*z0);
    double c1 = B + C*(-2*z0) + D*(3*z0*z0);
    double c2 = C + D*(-3*z0);
    double c3 = D;
    return {c0,c1,c2,c3};
}

// make a CKKS context with fixed parameters
CryptoContext<DCRTPoly> makeContext(){
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(4);
    p.SetScalingModSize(50);
    p.SetRingDim(1 << 15);               // 32768
    p.SetSecurityLevel(HEStd_128_classic);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

// evaluate + dump residuals for one segment of inputs
struct SegStats { double mae_ckks, mae_poly; };

SegStats eval_segment(const std::vector<double>& zs,
                      CryptoContext<DCRTPoly> cc,
                      const KeyPair<DCRTPoly>& kp,
                      const std::string& dump_ckks,
                      const std::string& dump_poly,
                      double z0,
                      int newton_steps)
{
    size_t n = zs.size();
    std::vector<double> seed(n), after_newton(n), true_val(n), poly_err(n);

    for(size_t i=0;i<n;++i){
        true_val[i] = inv_sqrt_true(zs[i]);

        // taylor seed
        seed[i] = taylor_seed_degree3(zs[i], z0);

        // plaintext Newton refinements
        double y = seed[i];
        for(int k=0;k<newton_steps;++k) y = newton_step(y, zs[i]);
        after_newton[i] = y;

        // final polynomial error AFTER Newton
        poly_err[i] = after_newton[i] - true_val[i];
    }

    // measure CKKS noise on the seed polynomial itself
    auto coeff = seed_coeff_in_z(z0);
    Plaintext ptZ = cc->MakeCKKSPackedPlaintext(zs);
    auto ctZ      = cc->Encrypt(kp.publicKey, ptZ);
    auto ctPoly   = cc->EvalPoly(ctZ, coeff);

    Plaintext ptDec;
    cc->Decrypt(kp.secretKey, ctPoly, &ptDec);
    ptDec->SetLength(n);
    const auto& dec_poly = ptDec->GetRealPackedValue();

    std::vector<double> ckksErr(n);
    for(size_t i=0;i<n;++i){
        ckksErr[i] = dec_poly[i] - seed[i];
    }

    append_bin(dump_ckks, ckksErr);
    append_bin(dump_poly, poly_err);

    return { meanAbs(ckksErr), meanAbs(poly_err) };
}

int main(int argc, char** argv){
    std::string infile = "layernorm_inputs_var.bin";
    std::string dump_ckks, dump_poly;
    double A = 0.024526, B = 68.0503; // hardcoded interval based on data
    bool use_auto_segments = true;
    double q0 = 0.05, q1 = 0.70, q2 = 0.995; // quantiles for auto segmenting

    for(int i=1;i<argc;++i){
        if(!std::strcmp(argv[i],"--in") && i+1<argc) infile = argv[++i];
        else if(!std::strcmp(argv[i],"--dump-ckks") && i+1<argc) dump_ckks = argv[++i];
        else if(!std::strcmp(argv[i],"--dump-poly") && i+1<argc) dump_poly = argv[++i];
        else if(!std::strcmp(argv[i],"--no-auto-seg")) use_auto_segments = false;
    }

    // Load values
    std::ifstream f(infile, std::ios::binary);
    if(!f){ std::cerr<<"cannot open "<<infile<<"\n"; return 1; }
    f.seekg(0,std::ios::end);
    size_t n = f.tellg()/sizeof(double);
    f.seekg(0);
    std::vector<double> vars(n);
    f.read(reinterpret_cast<char*>(vars.data()), n*sizeof(double));
    std::cout<<"Loaded "<<n<<" variance values\n";

    std::cout<<"Using hardcoded interval ["<<A<<","<<B<<"]\n";
    std::vector<double> kept; kept.reserve(vars.size());
    for(double v: vars) if(v>=A && v<=B) kept.push_back(v);
    std::cout<<"Kept "<<kept.size()<<" in range\n";

    double cut0 = 0.3, cut1 = 5.0; // defaults
    if(use_auto_segments){
        // compute quantile-based cuts inside [A,B]
        std::vector<double> tmp = kept;
        cut0 = quantile(tmp, q0);
        cut1 = quantile(tmp, q1);
        double b_hi = quantile(tmp, q2); // shrink B if desired
        // ensure monotonic
        cut0 = std::max(cut0, A);
        cut1 = std::min(std::max(cut1, cut0 + 1e-6), B);
        B    = std::max(b_hi, cut1 + 1e-6);
        std::cout<<"Auto segments by quantiles:\n"
                 <<"  q0="<<q0<<" → "<<cut0<<"\n"
                 <<"  q1="<<q1<<" → "<<cut1<<"\n"
                 <<"  q2="<<q2<<" → "<<B   <<"\n";
    }

    std::vector<double> seg0, seg1, seg2;
    for(double v: kept){
        if(v <= cut0) seg0.push_back(v);
        else if(v <= cut1) seg1.push_back(v);
        else seg2.push_back(v);
    }

    std::cout<<" seg0 ["<<A<<","<<cut0<<"] : "<<seg0.size()<<"\n";
    std::cout<<" seg1 ("<<cut0<<","<<cut1<<"] : "<<seg1.size()<<"\n";
    std::cout<<" seg2 ("<<cut1<<","<<B<<"] : "<<seg2.size()<<"\n";

    // subsample to single ciphertext
    auto ccTmp = makeContext();
    size_t cap = ccTmp->GetRingDimension()/2;
    std::random_device rd; std::mt19937_64 rng(rd());
    auto clip = [&](std::vector<double>& v){
        if(v.size() > cap){
            std::shuffle(v.begin(), v.end(), rng);
            v.resize(cap);
        }
    };
    clip(seg0); clip(seg1); clip(seg2);

    auto cc = makeContext();
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    // Segment-specific Newton steps (more near zero)
    int ns0 = 4, ns1 = 2, ns2 = 2;

    auto mid = [](double a,double b){ return 0.5*(a+b); };
    auto geom = [](double a,double b){ return std::sqrt(a*b); }; // better near 0

    double z0_0 = geom(A, cut0) + 1.0;             // geometric + 1
    double z0_1 = mid(cut0, cut1) + 1.0;
    double z0_2 = mid(cut1, B) + 1.0;

    if(!seg0.empty()){
        auto s0 = eval_segment(seg0, cc, kp, dump_ckks, dump_poly, z0_0, ns0);
        std::cout<<"Segment 0 ["<<A<<","<<cut0<<"]  n="<<seg0.size()
                 <<"  ckks-MAE="<<s0.mae_ckks<<"  postNewton-MAE="<<s0.mae_poly<<"\n";
    }
    if(!seg1.empty()){
        auto s1 = eval_segment(seg1, cc, kp, dump_ckks, dump_poly, z0_1, ns1);
        std::cout<<"Segment 1 ("<<cut0<<","<<cut1<<"]  n="<<seg1.size()
                 <<"  ckks-MAE="<<s1.mae_ckks<<"  postNewton-MAE="<<s1.mae_poly<<"\n";
    }
    if(!seg2.empty()){
        auto s2 = eval_segment(seg2, cc, kp, dump_ckks, dump_poly, z0_2, ns2);
        std::cout<<"Segment 2 ("<<cut1<<","<<B<<"]  n="<<seg2.size()
                 <<"  ckks-MAE="<<s2.mae_ckks<<"  postNewton-MAE="<<s2.mae_poly<<"\n";
    }
    return 0;
}
