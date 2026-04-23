FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENFHE_VERSION=v1.2.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libomp-dev \
    autoconf \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${OPENFHE_VERSION} \
      https://github.com/openfheorg/openfhe-development.git /tmp/openfhe \
    && cd /tmp/openfhe \
    && mkdir build && cd build \
    && cmake -DBUILD_EXAMPLES=OFF \
             -DBUILD_UNITTESTS=OFF \
             -DBUILD_BENCHMARKS=OFF \
             -DWITH_OPENMP=ON \
             .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/openfhe

WORKDIR /build/fhe_kernels
COPY fhe_kernels/CMakeLists.txt fhe_kernels/*.cpp ./
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

WORKDIR /build/depth_sweep
COPY depth_sweep/CMakeLists.txt depth_sweep/*.cpp ./
RUN mkdir build && cd build && cmake .. && make -j$(nproc)

WORKDIR /workspace
CMD ["/bin/bash"]