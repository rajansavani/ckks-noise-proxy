# CKKS Noise Proxy

A proxy for evaluating how fully homomorphic encryption degrades large language model
accuracy, without actually running encrypted inference end to end.

Encrypted LLM inference under CKKS is slow. A single benchmark like lm-eval-harness
on GPT-2 would take about a week if every forward pass ran fully encrypted. This repo
works around that by measuring the noise CKKS introduces once, storing those measurements,
and replaying the noise inside a normal plaintext model during evaluation. The result
is that a benchmark that would take a week runs in about three hours on a single GPU,
while still reflecting the fidelity loss the encrypted version would experience.

This was my summer 2025 research at Illinois Institute of Technology's Big Data X REU.
The work was accepted and presented at eScience 2025. The poster and extended abstract
are in `paper/`.

## What's in here

The repo is organized around the pipeline described in the paper.

`activation_collection/` runs GPT-2 with random inputs, captures the values that flow
into each non-linear layer (GeLU, LayerNorm, Softmax), and saves them. These samples
are needed so the CKKS measurements use realistic input distributions.

`poly_approximation/` contains the scripts that generate the GeLU polynomial
approximation error plots. The approximations themselves come from Dong et al. (2023);
this directory just visualizes how well they fit.

`fhe_kernels/` contains the C++ code that actually runs CKKS. For each non-linear
layer, it encrypts the collected activations, evaluates the polynomial approximation
homomorphically, decrypts, and records two residuals per sample: the polynomial
approximation error and the CKKS encryption error. Results get written to binary files.

`depth_sweep/` sweeps Chebyshev polynomial degrees 2 through 9 for the Softmax
exponential and records both the approximation error and the CKKS evaluation latency
at each degree. This is what produces the tradeoff curve in Figure 2 of the paper.

`noise_injection_benchmark/` is the payoff. It loads the stored residuals, injects
them into a normal GPT-2 via forward hooks, and runs lm-eval-harness on eight NLP tasks.
Compared to running fully encrypted inference, this is about 100x faster and the
accuracy drops match closely enough to guide design decisions.

`results/` contains the benchmark outputs from the paper and the latency/error
measurements from the depth sweep.

## Running it

The Python side runs natively on Windows, macOS, or Linux with Python 3.10 or 3.11.

```
git clone https://github.com/rajansavani/ckks-noise-proxy.git
cd ckks-noise-proxy
python -m venv .venv
# activate the venv (.venv/Scripts/Activate.ps1 on Windows, source .venv/bin/activate elsewhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For a PyTorch build with CUDA, grab the appropriate wheel command from pytorch.org
instead of the CPU line above.

To regenerate the benchmark plots from the saved results without rerunning anything:

```
cd noise_injection_benchmark
python simulate_noise.py --load-dir ../results/benchmark_run
```

To run a fresh benchmark you need the residual files in `residual_bins/`. Those are
produced by the C++ kernels, which is the part that needs OpenFHE. A Dockerfile is
included at the repo root that builds OpenFHE v1.2.4 and compiles the kernels.

```
docker build -t ckks-noise-proxy .
docker run --rm -it -v ${PWD}:/workspace ckks-noise-proxy
```

Inside the container, the compiled executables are at `/build/fhe_kernels/build/`
and `/build/depth_sweep/build/`. The activation files produced by
`activation_collection/` need to be converted from .npy to .bin using
`fhe_kernels/convert_npy_to_bin.py` before the kernels can read them.

## Limitations

The proxy models per-layer noise only. It does not account for bootstrapping overhead
or network transfer, both of which can dominate real end-to-end latency. Residuals
were collected from random token inputs, which may not match the distribution of
task-specific inputs. Latency numbers for full CKKS inference are extrapolated from
small pilot runs rather than measured directly. These are discussed in more detail
in the extended abstract.

## Authors

Rajan Savani (Illinois Institute of Technology), rsavani@hawk.illinoistech.edu  
Anwar Benhnini (Illinois Institute of Technology), abenhnini@hawk.illinoistech.edu  
André Bauer (Illinois Institute of Technology, mentor), abauer7@illinoistech.edu

## Citation

Savani, R., Benhnini, A., and Bauer, A. "Finding the Sweet Spot: Speed vs. Fidelity
in Encrypted LLMs via Proxy Simulation." IEEE International Conference on e-Science, 2025.
https://ieeexplore.ieee.org/abstract/document/11181475

## Acknowledgments

This work is supported in part by the National Science Foundation OAC-2150500 award.
