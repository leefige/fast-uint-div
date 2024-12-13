enable_language(CUDA)

find_package(CUDAToolkit 12 REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xptxas -warn-lmem-usage")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xptxas -warn-spills")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -expt-relaxed-constexpr")

add_executable(uint-div-cuda-test test_u32div.cu)
target_include_directories(uint-div-cuda-test
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(MSVC)
  target_link_options(uint-div-cuda-test
    PRIVATE "/NODEFAULTLIB:libcmt.lib")
endif()

add_executable(uint-div-cuda-bench bench_u32div.cu)
target_include_directories(uint-div-cuda-bench
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if(MSVC)
  target_link_options(uint-div-cuda-bench
    PRIVATE "/NODEFAULTLIB:libcmt.lib")
endif()
