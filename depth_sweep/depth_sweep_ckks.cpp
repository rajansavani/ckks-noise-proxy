#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "openfhe.h"
#include "openfhe/pke/scheme/ckksrns/gen-cryptocontext-ckksrns.h"
using namespace lbcrypto;

static double polyval(const std::vector<double>& c, double x){
    double a = 0.0; for(auto it=c.rbegin(); it!=c.rend(); ++it) a = a*x + *it; return a;
}
static double meanAbs(const std::vector<double>& v){
    double s=0.0; for(double e: v) s+=std::abs(e); return s/v.size();
}
static inline double f_true(double x){ return std::exp(x); }  // target op

// pre-made polynomial coeffs for approximating exp(x) on [-4,4] with increasing degree (0 to 10)
static std::vector<std::vector<double>> make_poly_bank(){
    return {
        {1.0,1.0},
        {1.0,1.0,0.5},
        {1.0,1.0,0.5,1.0/6.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0,1.0/120.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0,1.0/120.0,1.0/720.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0,1.0/120.0,1.0/720.0,1.0/5040.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0,1.0/120.0,1.0/720.0,1.0/5040.0,1.0/40320.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0,1.0/120.0,1.0/720.0,1.0/5040.0,1.0/40320.0,1.0/362880.0},
        {1.0,1.0,0.5,1.0/6.0,1.0/24.0,1.0/120.0,1.0/720.0,1.0/5040.0,1.0/40320.0,1.0/362880.0,1.0/3628800.0}
    };
}

// helper to make a CKKS context with appropriate parameters for a given multiplicative depth
static CryptoContext<DCRTPoly> makeContext(uint32_t depth){
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(depth);
    p.SetScalingModSize(50);
    p.SetRingDim(1 << 16);
    p.SetSecurityLevel(HEStd_128_classic);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

int main(int argc, char** argv) try {
    // parse args
    int minDeg = (argc>1)? std::stoi(argv[1]) : 2;
    int maxDeg = (argc>2)? std::stoi(argv[2]) : 10;
    int runs   = (argc>3)? std::stoi(argv[3]) : 3;

    // pre-make polys + add some extra depth for safety (since the "depth" of a poly is really a lower bound on the true multiplicative depth needed)
    const uint32_t EXTRA_DEPTH = 2;
    auto polys = make_poly_bank();

    std::mt19937_64 rng(0);
    std::uniform_real_distribution<> ud(-4.0, 4.0);

    std::cout << "degree,depth,slots,enc_ms,eval_ms,dec_ms,total_ms,ckks_mae,poly_mae\n";

    // loop over poly degrees
    for(int deg=minDeg; deg<=maxDeg && deg<(int)polys.size(); ++deg){
        // make context + keys
        const auto& coeff = polys[deg];
        uint32_t depth = (uint32_t)coeff.size() - 1 + EXTRA_DEPTH;

        auto cc = makeContext(depth);
        size_t N = cc->GetRingDimension()/2;

        auto kp = cc->KeyGen();
        cc->EvalMultKeyGen(kp.secretKey);

        // make some random inputs + get poly approximation + get true values
        std::vector<double> x(N), y_poly(N), y_true(N);
        for(size_t i=0;i<N;++i){ x[i]=ud(rng); y_poly[i]=polyval(coeff,x[i]); y_true[i]=f_true(x[i]); }

        double enc_ms=0, eval_ms=0, dec_ms=0, total_ms=0;
        double mae_ckks=0, mae_poly=0;

        // repeat multiple times for more stable timing
        for(int r=0;r<runs;++r){
            // start timing + encrypt + eval poly + decrypt + end timing
            auto t0=std::chrono::high_resolution_clock::now();
            Plaintext ptIn = cc->MakeCKKSPackedPlaintext(x);
            auto t1=std::chrono::high_resolution_clock::now();
            auto ctIn = cc->Encrypt(kp.publicKey, ptIn);
            auto t2=std::chrono::high_resolution_clock::now();
            auto ctOut= cc->EvalPoly(ctIn, coeff);
            auto t3=std::chrono::high_resolution_clock::now();
            Plaintext ptDec;
            cc->Decrypt(kp.secretKey, ctOut, &ptDec);
            ptDec->SetLength(N);
            auto t4=std::chrono::high_resolution_clock::now();

            const auto& dec = ptDec->GetRealPackedValue();
            std::vector<double> fheErr(N), approxErr(N);
            for(size_t i=0;i<N;++i){ fheErr[i]=dec[i]-y_poly[i]; approxErr[i]=y_poly[i]-y_true[i]; }

            mae_ckks += meanAbs(fheErr);
            mae_poly += meanAbs(approxErr);

            enc_ms   += std::chrono::duration<double,std::milli>(t2-t1).count();
            eval_ms  += std::chrono::duration<double,std::milli>(t3-t2).count();
            dec_ms   += std::chrono::duration<double,std::milli>(t4-t3).count();
            total_ms += std::chrono::duration<double,std::milli>(t4-t0).count();
        }

        mae_ckks/=runs; mae_poly/=runs;
        enc_ms/=runs; eval_ms/=runs; dec_ms/=runs; total_ms/=runs;

        std::cout << deg << "," << depth << "," << N << ","
                  << enc_ms << "," << eval_ms << "," << dec_ms << ","
                  << total_ms << "," << mae_ckks << "," << mae_poly << "\n";
    }
    return 0;
}
catch(const std::exception& e){
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
