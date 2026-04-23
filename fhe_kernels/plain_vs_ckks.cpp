#include <openfhe.h>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

using namespace lbcrypto;
using Clock = std::chrono::high_resolution_clock;

// naïve N×N plaintext matmul (triple loop)
double plainMatmul(const std::vector<float>& A,
                   const std::vector<float>& B,
                   size_t N)
{
    std::vector<float> C(N * N, 0.f);
    auto t0 = Clock::now();
    for (size_t i = 0; i < N; ++i)
        for (size_t k = 0; k < N; ++k) {
            float aik = A[i*N + k];
            for (size_t j = 0; j < N; ++j)
                C[i*N + j] += aik * B[k*N + j];
        }
    auto t1 = Clock::now();
    return std::chrono::duration<double,std::milli>(t1 - t0).count();
}

// CKKS slot-wise multiply timing (single ciphertext)
double ckksSlotMult(const std::vector<float>& A,
                    const std::vector<float>& B,
                    size_t ringDim = 16384,
                    uint32_t scaleBits = 59)
{
    CCParams<CryptoContextCKKSRNS> params;
    params.SetRingDim(ringDim);
    params.SetScalingModSize(scaleBits);
    params.SetMultiplicativeDepth(1);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKESchemeFeature::PKE);
    cc->Enable(PKESchemeFeature::LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    std::vector<double> aD(A.begin(), A.end());
    std::vector<double> bD(B.begin(), B.end());

    auto ptA = cc->MakeCKKSPackedPlaintext(aD);
    auto ptB = cc->MakeCKKSPackedPlaintext(bD);
    auto ctA = cc->Encrypt(keys.publicKey, ptA);
    auto ctB = cc->Encrypt(keys.publicKey, ptB);

    auto t0 = Clock::now();
    auto ctC = cc->EvalMult(ctA, ctB);
    cc->ModReduceInPlace(ctC);
    auto t1 = Clock::now();
    return std::chrono::duration<double,std::milli>(t1 - t0).count();
}

int main(int argc, char** argv)
{
    size_t N = (argc > 1) ? std::stoul(argv[1]) : 64;  // default N=64 for slot fit
    const int trials = 1000;

    std::cout << "Averaging over " << trials
              << " runs for " << N << " × " << N << " matrices\n\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> A(N * N), B(N * N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    double pt_sum = 0, ckks_sum = 0;
    for (int t = 0; t < trials; ++t) {
        pt_sum   += plainMatmul(A, B, N);
        ckks_sum += ckksSlotMult(A, B);
    }
    double pt_ms = pt_sum / trials;
    double ckks_ms = ckks_sum / trials;

    std::cout << "Plaintext (avg):  " << pt_ms << " ms\n";
    std::cout << "CKKS EvalMult (avg):  " << ckks_ms << " ms\n";
    std::cout << "\nAverage slowdown ≈ " << (ckks_ms / pt_ms) << " ×\n";
    return 0;
}
