cmake_minimum_required(VERSION 3.15)
project(ckks_noise_proxy_depth_sweep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)

link_directories(/usr/local/lib)

add_executable(depth_sweep_ckks depth_sweep_ckks.cpp)
target_link_libraries(depth_sweep_ckks PRIVATE OPENFHEcore OPENFHEpke)
