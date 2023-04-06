cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(Erosion LANGUAGES CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build: Debug, Release" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

find_package(CUDA REQUIRED)

## build "cpu" part
# cpu part
set(CMAKE_CXX_COMPILER "g++")
set(ErosionCPUSrcs
  srcs/cpu.cpp
)
add_library(ErosionCPU OBJECT
  ${ErosionCPUSrcs}
)
target_include_directories(ErosionCPU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(ErosionCPU
  PROPERTIES
    CXX_STANDARD 11
)

## build "GPU One Step ErosionKernel" case
# kernel part to library
set(CMAKE_CXX_COMPILER "nvcc")
set(ErosionGPUSrcs
  srcs/gpu.cu
)
add_library(ErosionGPU OBJECT
  ${ErosionGPUSrcs}
)
target_include_directories(ErosionGPU PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
)
target_compile_options(ErosionGPU PRIVATE
  "-arch=compute_75"  # nvidia 2080 ti
  "-code=sm_75"
)
option(SHOW_MEMORY "Show memory uasge for kernel function" OFF)
if(SHOW_MEMORY)
  target_compile_options(ErosionGPU PRIVATE
    "--ptxas-options=-v"
  )
endif()
target_link_options(ErosionGPU PRIVATE
  "-L/usr/local/cuda/lib64"
  "-lcudart"
)

## build program(device + host)
set(CMAKE_CXX_COMPILER "g++")
set(ErosionCaseSrcs
  srcs/main.cpp
)
add_executable(ErosionCase
  ${ErosionCaseSrcs}
  $<TARGET_OBJECTS:ErosionCPU>
  $<TARGET_OBJECTS:ErosionGPU>
)
set_target_properties(ErosionCase
  PROPERTIES
    CXX_STANDARD 11
)
target_include_directories(ErosionCase PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include
)
target_link_options(ErosionCase PRIVATE
  "-L/usr/local/cuda/lib64"
  "-lcudart"
)