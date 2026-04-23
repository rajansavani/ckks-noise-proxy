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

// helpers for true function, polynomial evaluation, error metrics, and dumping binary data
static inline double exp_taylor_pow2(double x, int r){
    // (1 + x / 2^r)^(2^r), valid for x<=0
    double base = 1.0 + x / std::pow(2.0, r);
    double y = base;
    int pow2 = 1<<r;
    for(int i=1;i<pow2;i++) y *= base;
    return y;
}

static inline double softmax_reference(const std::vector<double>& row){
    // return sum(row) just to consume? We need per element err
    return 0.0;
}

static std::vector<double> softmax_vec_ref(const std::vector<double>& row){
    double m = *std::max_element(row.begin(), row.end());
    std::vector<double> exps(row.size());
    double s = 0.0;
    for(size_t i=0;i<row.size();++i){
        double v = std::exp(row[i]-m);
        exps[i]=v; s+=v;
    }
    for(double& v:exps) v/=s;
    return exps;
}

static double meanAbs(const std::vector<double>& v){
    double s=0; for(double z:v) s+=std::abs(z); return s/v.size();
}

struct PickedParams { int r; int d; double polyMae; };

// grid search to pick exp approximation params based on sample of data
PickedParams auto_pick_rd(const std::vector<double>& samples, int rowSize,
                          double m, double targetMae=1e-2){
    // sample first few rows
    size_t rows = samples.size()/rowSize;
    size_t testRows = std::min<size_t>(200, rows);
    std::vector<double> test(samples.begin(), samples.begin()+testRows*rowSize);
    // search small ranges
    int bestR=2, bestD=1; double bestMae=1e9;
    for(int r=2;r<=6;r++){
        // only exp approx uses r; division d try 1..3
        for(int d=1; d<=3; d++){
            std::vector<double> errs;
            errs.reserve(test.size());
            for(size_t rr=0; rr<testRows; ++rr){
                std::vector<double> row(rowSize);
                std::copy_n(test.begin()+rr*rowSize,rowSize,row.begin());
                // approx max ~ m (given)
                std::vector<double> num(rowSize);
                for(size_t i=0;i<rowSize;++i){
                    double x = row[i]-m; if(x>0) x=0; // clamp >0 to 0 (exp defined)
                    num[i] = exp_taylor_pow2(x, r);
                }
                double denom = 0; for(double v:num) denom+=v;
                // Goldschmidt d iters
                double F = 1.0/denom; // seed
                for(int it=0; it<d; ++it){
                    double D = denom*F;
                    F *= (2.0 - D);
                }
                for(size_t i=0;i<rowSize;++i){
                    double approx = num[i]*F;
                    double truth = std::exp(row[i]-m);
                    // real denom
                    // just compute softmax ref: using earlier row
                }
                // use real ref softmax
                auto ref = softmax_vec_ref(row);
                for(size_t i=0;i<rowSize;++i){
                    double x = row[i]-m; if(x>0) x=0;
                    double numi = exp_taylor_pow2(x,r);
                    double denomA=0; for(double v:num) denomA+=v;
                    double F0 = 1.0/denomA;
                    double Fcur = F0;
                    for(int it=0; it<d; ++it){
                        double D = denomA*Fcur;
                        Fcur *= (2.0 - D);
                    }
                    double approx = numi*Fcur;
                    errs.push_back(approx - ref[i]);
                }
            }
            double mae = meanAbs(errs);
            if(mae < bestMae){
                bestMae = mae; bestR = r; bestD = d;
            }
        }
    }
    return {bestR,bestD,bestMae};
}

// compute exp approximation and Goldschmidt division in CKKS, return MAE vs reference
Ciphertext<DCRTPoly> he_exp_approx(CryptoContext<DCRTPoly> cc,
                                   const Ciphertext<DCRTPoly>& x,
                                   int r){
    double scale = std::pow(2.0, r);
    auto x_scaled = cc->EvalMult(x, 1.0/scale);
    auto base = cc->EvalAdd(1.0, x_scaled); // 1 + x/2^r
    Ciphertext<DCRTPoly> y = base;
    int times = (1<<r) - 1;
    for(int i=0;i<times;i++){
        y = cc->EvalMult(y, base);
    }
    return y;
}

// Goldschmidt division: return A/B given ciphertexts A,B and number of iterations d
Ciphertext<DCRTPoly> he_div_goldschmidt(CryptoContext<DCRTPoly> cc,
                                        const Ciphertext<DCRTPoly>& A,
                                        const Ciphertext<DCRTPoly>& B,
                                        int d){
    // start with plaintext seed F0 = 1/approx(B) (estimate 1/64 here since B is sum of exp of ~[-5,0])
    auto F = cc->EvalMult(B, 0.0); // zero ciphertext same size
    double F0 = 1.0/64.0;
    F = cc->EvalAdd(F, F0);

    auto N = A;
    auto D = B;

    for(int i=0; i<d; ++i){
        // twoMinus = 2 - D*F
        auto DF = cc->EvalMult(D, F);
        auto twoMinus = cc->EvalAdd(2.0, cc->EvalMult(DF, -1.0));
        F = cc->EvalMult(F, twoMinus);
        N = cc->EvalMult(N, twoMinus);
        D = cc->EvalMult(D, twoMinus);
    }
    return N;
}

// helper to make a CKKS context with fixed parameters for softmax eval
CryptoContext<DCRTPoly> makeContext(uint32_t depth, uint32_t ringDim){
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(depth);
    p.SetRingDim(ringDim);
    p.SetScalingModSize(50);
    p.SetSecurityLevel(HEStd_128_classic);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    return cc;
}

struct Stats { double ckksMae; double polyMae; };

int main(int argc, char** argv){
    std::string infile="softmax_inputs.bin";
    std::string dumpC="", dumpP="";
    double a=-5.36, b=4.26, xmax=4.26;
    size_t rowSize=64;

    for(int i=1;i<argc;++i){
        if(!std::strcmp(argv[i],"--in") && i+1<argc) infile=argv[++i];
        else if(!std::strcmp(argv[i],"--dump-ckks") && i+1<argc) dumpC=argv[++i];
        else if(!std::strcmp(argv[i],"--dump-poly") && i+1<argc) dumpP=argv[++i];
        else if(!std::strcmp(argv[i],"--a") && i+1<argc) a=std::atof(argv[++i]);
        else if(!std::strcmp(argv[i],"--b") && i+1<argc) b=std::atof(argv[++i]);
        else if(!std::strcmp(argv[i],"--xmax") && i+1<argc) xmax=std::atof(argv[++i]);
        else if(!std::strcmp(argv[i],"--rowsize") && i+1<argc) rowSize=std::stoul(argv[++i]);
    }

    // load
    std::ifstream f(infile, std::ios::binary);
    if(!f){ std::cerr<<"cannot open "<<infile<<"\n"; return 1;}
    f.seekg(0,std::ios::end);
    size_t n = f.tellg()/sizeof(double);
    f.seekg(0);
    std::vector<double> vals(n);
    f.read(reinterpret_cast<char*>(vals.data()), n*sizeof(double));
    std::cout<<"Loaded "<<n<<" pre-softmax scores\n";

    // filter to [a,b]
    std::vector<double> filtered;
    filtered.reserve(n);
    for(double v:vals) if(v>=a && v<=b) filtered.push_back(v);
    std::cout<<"Kept "<<filtered.size()<<" in ["<<a<<","<<b<<"]\n";

    // auto pick r,d
    auto picked = auto_pick_rd(filtered, rowSize, xmax, 1e-2);
    std::cout<<"Auto-picked r="<<picked.r<<" d="<<picked.d<<" (poly-MAE≈"<<picked.polyMae<<")\n";

    // chunk rows
    size_t rows = filtered.size()/rowSize;
    filtered.resize(rows*rowSize);
    uint32_t ringDim = 65536; // standard
    uint32_t depthNeeded =  (1<<picked.r) + picked.d + 4;
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> kp;
    bool ok=false;
    for(int tries=0; tries<3 && !ok; ++tries){
        try{
            cc = makeContext(depthNeeded, ringDim);
            kp = cc->KeyGen();
            cc->EvalMultKeyGen(kp.secretKey);
            cc->EvalSumKeyGen(kp.secretKey);
            cc->EvalAtIndexKeyGen(kp.secretKey, {1,-1,(int)rowSize,-(int)rowSize});
            ok=true;
        }catch(const std::exception& e){
            ringDim <<=1;
        }
    }
    if(!ok){ std::cerr<<"Failed to create context\n"; return 1;}

    size_t slotCap = cc->GetRingDimension()/2;
    size_t rowsPerCipher = slotCap / rowSize;
    if(rowsPerCipher==0){ std::cerr<<"rowSize too big for packing\n"; return 1; }

    std::vector<double> polyErrs, ckksErrs;
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<size_t> dist(0, rows-1);
    size_t evalRows = std::min<size_t>(400, rows);

    for(size_t batch=0; batch<evalRows; ){
        size_t take = std::min(rowsPerCipher, evalRows - batch);
        std::vector<double> pack(take*rowSize);
        for(size_t rIdx=0;rIdx<take;++rIdx){
            // random row
            size_t rr = dist(rng);
            std::copy_n(filtered.begin()+rr*rowSize,rowSize, pack.begin()+rIdx*rowSize);
        }
        // ground truth
        std::vector<double> truth(pack.size());
        for(size_t rIdx=0;rIdx<take;++rIdx){
            std::vector<double> row(rowSize);
            std::copy_n(pack.begin()+rIdx*rowSize,rowSize,row.begin());
            auto ref = softmax_vec_ref(row);
            std::copy(ref.begin(), ref.end(), truth.begin()+rIdx*rowSize);
        }
        // polynomial approx (plaintext)
        std::vector<double> poly(pack.size());
        for(size_t rIdx=0;rIdx<take;++rIdx){
            double m = xmax; // using global
            std::vector<double> num(rowSize);
            for(size_t i=0;i<rowSize;++i){
                double x = pack[rIdx*rowSize+i] - m;
                if(x>0) x=0;
                num[i] = exp_taylor_pow2(x, picked.r);
            }
            double denom=0; for(double v:num) denom+=v;
            double F=1.0/denom;
            for(int it=0; it<picked.d; ++it){
                double D=denom*F;
                F *= (2.0 - D);
            }
            for(size_t i=0;i<rowSize;++i){
                poly[rIdx*rowSize+i] = num[i]*F;
            }
        }

        // HE path, pack as plaintext and encrypt
        Plaintext pt = cc->MakeCKKSPackedPlaintext(pack);
        auto ct = cc->Encrypt(kp.publicKey, pt);
        auto xmax_ct = cc->EvalMult(ct, 0.0); // zero
        xmax_ct = cc->EvalAdd(xmax_ct, xmax);
        auto diff = cc->EvalSub(ct, xmax_ct);
        auto num_ct = he_exp_approx(cc, diff, picked.r);
        auto denom_ct = cc->EvalSum(num_ct, rowSize);
        auto prob_ct = he_div_goldschmidt(cc, num_ct, denom_ct, picked.d);

        Plaintext dec;
        cc->Decrypt(kp.secretKey, prob_ct, &dec);
        dec->SetLength(pack.size());
        auto decv = dec->GetRealPackedValue();

        for(size_t i=0;i<pack.size();++i){
            ckksErrs.push_back(decv[i] - poly[i]);
            polyErrs.push_back(poly[i] - truth[i]);
        }
        batch += take;
    }

    double maeC = meanAbs(ckksErrs);
    double maeP = meanAbs(polyErrs);
    std::cout<<"ckks-MAE="<<maeC<<"  poly-MAE="<<maeP<<"\n";

    if(!dumpC.empty()){
        std::ofstream(dumpC, std::ios::binary).write(reinterpret_cast<const char*>(ckksErrs.data()),
                                                     ckksErrs.size()*sizeof(double));
    }
    if(!dumpP.empty()){
        std::ofstream(dumpP, std::ios::binary).write(reinterpret_cast<const char*>(polyErrs.data()),
                                                     polyErrs.size()*sizeof(double));
    }
    return 0;
}
