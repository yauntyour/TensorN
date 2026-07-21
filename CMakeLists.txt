cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(FetchContent)

project(TensorN
    VERSION 1.0.0
    DESCRIPTION "A C++ Tensor Library with CUDA and OpenBlas support"
    LANGUAGES CXX
)

# ============================================================================
# Options
# ============================================================================
option(TENSORN_ENABLE_CUDA "Enable CUDA support" ON)
option(TENSORN_ENABLE_OPENBLAS "Enable OpenBlas support" ON)
option(TENSORN_BUILD_EXAMPLES "Build example programs" ON)
option(TENSORN_BUILD_BENCHMARKS "Build benchmark programs" ON)

# ============================================================================
# C++ Standard
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# CUDA Configuration
# ============================================================================
if(TENSORN_ENABLE_CUDA)
    # CUDA requires MSVC on Windows
    if(WIN32 AND NOT MSVC)
        message(WARNING "CUDA on Windows requires MSVC compiler. Disabling CUDA support.")
        set(TENSORN_ENABLE_CUDA OFF)
    else()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        
        # Check CUDA version (require 13.3 or compatible)
        if(CUDAToolkit_VERSION VERSION_LESS "13.3")
            message(WARNING "CUDA ${CUDAToolkit_VERSION} found, but CUDA 13.3 is recommended")
        endif()
        
        # CUDA architecture settings
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
        endif()
        
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# ============================================================================
# OpenBlas Configuration
# ============================================================================
if(TENSORN_ENABLE_OPENBLAS)
    # Add cmake module path
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    
    find_package(OpenBLAS QUIET)
    
    if(NOT OpenBLAS_FOUND)
        message(WARNING "OpenBlas not found. OpenBlas support will be disabled.")
        set(TENSORN_ENABLE_OPENBLAS OFF)
    else()
        message(STATUS "Found OpenBlas: ${OpenBLAS_LIBRARIES}")
        if(OpenBLAS_VERSION)
            message(STATUS "OpenBlas version: ${OpenBLAS_VERSION}")
        endif()
    endif()
endif()

# ============================================================================
# nlohmann/json (header-only, fetched automatically)
# ============================================================================
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ============================================================================
# ZLIB Configuration (required for cnpy) - fetched to avoid MinGW/MSVC conflicts
# ============================================================================
FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        v1.3.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(zlib)

# ============================================================================
# Main Library Target (Header-only)
# ============================================================================
add_library(TensorN INTERFACE)
add_library(TensorN::TensorN ALIAS TensorN)

target_include_directories(TensorN INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(TensorN INTERFACE cxx_std_17)

target_link_libraries(TensorN INTERFACE nlohmann_json::nlohmann_json zlibstatic)

# ============================================================================
# CUDA Source Files
# ============================================================================
if(TENSORN_ENABLE_CUDA)
    set(TENSORN_CUDA_SOURCES
        core/CUDA/cuda_tensor.cu
        core/CUDA/elementwise.cu
        core/CUDA/reduction.cu
        core/CUDA/matmul.cu
        core/CUDA/convolution.cu
    )
    
    # Create CUDA static library
    add_library(TensorN_CUDA STATIC ${TENSORN_CUDA_SOURCES})
    
    target_include_directories(TensorN_CUDA PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(TensorN_CUDA PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cufft
        nlohmann_json::nlohmann_json
    )
    
    target_compile_definitions(TensorN_CUDA PUBLIC TENSORN_CUDA_AVAILABLE)
    
    set_target_properties(TensorN_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(TensorN_CUDA PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    
    # Link CUDA library to main target
    target_link_libraries(TensorN INTERFACE TensorN_CUDA)
endif()

# ============================================================================
# OpenBlas Configuration
# ============================================================================
if(TENSORN_ENABLE_OPENBLAS AND OpenBLAS_FOUND)
    target_include_directories(TensorN INTERFACE ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(TensorN INTERFACE ${OpenBLAS_LIBRARIES})
    target_compile_definitions(TensorN INTERFACE TENSORN_HAS_OPENBLAS=1)
    
    # Also link to CUDA library if enabled
    if(TENSORN_ENABLE_CUDA)
        target_include_directories(TensorN_CUDA PUBLIC ${OpenBLAS_INCLUDE_DIRS})
        target_link_libraries(TensorN_CUDA PUBLIC ${OpenBLAS_LIBRARIES})
    endif()
endif()

# ============================================================================
# Examples
# ============================================================================
if(TENSORN_BUILD_EXAMPLES)
    set(TENSORN_EXAMPLES
        example/exp1.cpp
        example/exp2.cpp
        example/exp3.cpp
        example/exp4.cpp
        example/exp5.cpp
    )
    
    foreach(EXAMPLE_SOURCE ${TENSORN_EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE TensorN)
        
        # Set output directory
        set_target_properties(${EXAMPLE_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples"
        )
    endforeach()
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(TENSORN_BUILD_BENCHMARKS)
    add_executable(TensorN_Benchmark benchmark/benchmark.cpp)
    target_link_libraries(TensorN_Benchmark PRIVATE TensorN)
    
    set_target_properties(TensorN_Benchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(DIRECTORY core/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/TensorN/core
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(FILES TensorN.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/TensorN
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "TensorN Configuration Summary:")
message(STATUS "  Version:            ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Support:       ${TENSORN_ENABLE_CUDA}")
if(TENSORN_ENABLE_CUDA)
    message(STATUS "  CUDA Version:       ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  OpenBlas Support:   ${TENSORN_ENABLE_OPENBLAS}")
message(STATUS "  Build Examples:     ${TENSORN_BUILD_EXAMPLES}")
message(STATUS "  Build Benchmarks:   ${TENSORN_BUILD_BENCHMARKS}")
message(STATUS "")