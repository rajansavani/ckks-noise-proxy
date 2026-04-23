cmake_minimum_required(VERSION 3.15)
project(ckks_noise_proxy_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe
)

link_directories(/usr/local/lib)

add_executable(eval_gelu               eval_gelu.cpp)
add_executable(eval_layernorm_inv_sqrt eval_layernorm_inv_sqrt.cpp)
add_executable(eval_softmax            eval_softmax.cpp)
add_executable(measure_linear_noise    measure_linear_noise.cpp)
add_executable(plain_vs_ckks           plain_vs_ckks.cpp)

target_link_libraries(eval_gelu               PRIVATE OPENFHEcore OPENFHEpke)
target_link_libraries(eval_layernorm_inv_sqrt PRIVATE OPENFHEcore OPENFHEpke)
target_link_libraries(eval_softmax            PRIVATE OPENFHEcore OPENFHEpke)
target_link_libraries(measure_linear_noise    PRIVATE OPENFHEcore OPENFHEpke)
target_link_libraries(plain_vs_ckks           PRIVATE OPENFHEcore OPENFHEpke)
