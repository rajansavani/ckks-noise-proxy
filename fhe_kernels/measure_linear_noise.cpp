#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <cstring>
#include "openfhe.h"
#include "openfhe/pke/scheme/ckksrns/gen-cryptocontext-ckksrns.h"

using namespace lbcrypto;

// helper to compute mean absolute value of a vector
static double meanAbs(const std::vector<double>& v) {
    double s = 0.0; for (double e : v) s += std::abs(e); return s / v.size();
}

// parse args for this program
struct Args {
    size_t dim = 768;
    size_t n_rows = 768;
    size_t batches = 1;
    std::string dump_ckks = "matmul_ckks.bin";
    std::string act_file;
    std::string weight_file;
    int depth = 4;
    size_t ringdim = 1 << 16;
};

Args parse(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--dim") && i+1<argc)        a.dim = std::stoul(argv[++i]);
        else if(!strcmp(argv[i],"--rows") && i+1<argc)  a.n_rows = std::stoul(argv[++i]);
        else if(!strcmp(argv[i],"--batches") && i+1<argc) a.batches = std::stoul(argv[++i]);
        else if(!strcmp(argv[i],"--dump-ckks") && i+1<argc) a.dump_ckks = argv[++i];
        else if(!strcmp(argv[i],"--act-bin") && i+1<argc)   a.act_file = argv[++i];
        else if(!strcmp(argv[i],"--w-bin") && i+1<argc)     a.weight_file = argv[++i];
        else if(!strcmp(argv[i],"--depth") && i+1<argc)     a.depth = std::stoi(argv[++i]);
        else if(!strcmp(argv[i],"--ringdim") && i+1<argc)   a.ringdim = std::stoul(argv[++i]);
        else std::cerr << "Unknown arg: " << argv[i] << "\n";
    }
    return a;
}

// helper to make a CKKS context with given depth and ring dimension
CryptoContext<DCRTPoly> makeContext(int depth, size_t ringdim){
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(depth);
    p.SetScalingModSize(50);
    p.SetRingDim(ringdim);
    p.SetSecurityLevel(HEStd_128_classic);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

// reduce-all-slots sum using power-of-two rotations
Ciphertext<DCRTPoly> sumSlots(const CryptoContext<DCRTPoly>& cc,
                              Ciphertext<DCRTPoly> ct,
                              size_t len,
                              const std::vector<int>& rotKeys) {
    for(size_t sh=1; sh < len; sh <<= 1){
        auto r = cc->EvalAtIndex(ct, sh);
        ct = cc->EvalAdd(ct, r);
    }
    return ct;
}

int main(int argc, char** argv) try {
    Args args = parse(argc, argv);
    std::cout << "dim="<<args.dim<<" rows="<<args.n_rows
              <<" batches="<<args.batches<<"\n";

    // RNG
    std::random_device rd; std::mt19937_64 rng(rd());
    std::normal_distribution<> nd(0.0, 0.5);

    // load/generate activations
    std::vector<double> acts(args.dim * args.batches);
    if(!args.act_file.empty()){
        std::ifstream f(args.act_file, std::ios::binary);
        if(!f) throw std::runtime_error("cannot open act file");
        f.read(reinterpret_cast<char*>(acts.data()),
               acts.size()*sizeof(double));
    }else{
        for(double& v: acts) v = nd(rng);
    }

    // load/generate weights (row-major)
    std::vector<double> weights(args.dim * args.n_rows);
    if(!args.weight_file.empty()){
        std::ifstream f(args.weight_file, std::ios::binary);
        if(!f) throw std::runtime_error("cannot open weight file");
        f.read(reinterpret_cast<char*>(weights.data()),
               weights.size()*sizeof(double));
    }else{
        for(double& v: weights) v = nd(rng);
    }

    auto cc = makeContext(args.depth, args.ringdim);
    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);

    // rotation indices we need: 1,2,4,... up to next power of two >= dim
    std::vector<int> shifts;
    for(size_t sh=1; sh<args.dim; sh<<=1) shifts.push_back((int)sh);
    cc->EvalAtIndexKeyGen(kp.secretKey, shifts);

    std::ofstream dump(args.dump_ckks, std::ios::binary);
    if(!dump) throw std::runtime_error("cannot open dump file");

    size_t packedSlots = cc->GetRingDimension()/2;
    if(args.dim > packedSlots)
        throw std::runtime_error("dim > ringDim/2, increase ringdim");

    std::vector<double> residuals;
    residuals.reserve(args.n_rows * args.batches);

    for(size_t b=0; b<args.batches; ++b){
        const double* x = acts.data() + b*args.dim;

        Plaintext ptX = cc->MakeCKKSPackedPlaintext(std::vector<double>(x, x+args.dim));
        auto ctX = cc->Encrypt(kp.publicKey, ptX);

        for(size_t r=0; r<args.n_rows; ++r){
            const double* w = weights.data() + r*args.dim;

            Plaintext ptW = cc->MakeCKKSPackedPlaintext(std::vector<double>(w, w+args.dim));
            auto ctProd = cc->EvalMult(ctX, ptW);

            auto ctSum = sumSlots(cc, ctProd, args.dim, shifts);

            Plaintext ptDec;
            cc->Decrypt(kp.secretKey, ctSum, &ptDec);
            ptDec->SetLength(args.dim);
            double he_out = ptDec->GetRealPackedValue()[0];

            // plaintext dot
            double ref = std::inner_product(x, x+args.dim, w, 0.0);

            residuals.push_back(he_out - ref);

            // flush occasionally to keep memory reasonable
            if(residuals.size() >= 4096){
                dump.write(reinterpret_cast<const char*>(residuals.data()),
                           residuals.size()*sizeof(double));
                residuals.clear();
            }
        }
    }

    if(!residuals.empty())
        dump.write(reinterpret_cast<const char*>(residuals.data()),
                   residuals.size()*sizeof(double));

    dump.close();

    // quick report (on last chunk)
    std::cout << "Wrote residuals to " << args.dump_ckks << "\n";
    std::cout << "Example MAE (last block): " << meanAbs(residuals) << "\n";
    return 0;
}
catch(const std::exception& e){
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
