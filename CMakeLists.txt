cmake_minimum_required(VERSION 3.20)
project(TensorN
    VERSION 0.1.0
    DESCRIPTION "A safer, more efficient, and more universal native C++ Tensor library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options ----
option(TENSORN_BUILD_CUDA "Build CUDA accelerated operators" ON)
option(TENSORN_BUILD_OPENBLAS "Build OpenBLAS accelerated CPU operators" ON)
option(TENSORN_BUILD_EXAMPLES "Build example programs" ON)
option(TENSORN_BUILD_TESTS "Build tests" ON)
option(TENSORN_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# ---- Header-only library target ----
add_library(tensorn INTERFACE)
target_include_directories(tensorn INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(tensorn INTERFACE cxx_std_23)

# ---- OpenBLAS CPU backend ----
if(TENSORN_BUILD_OPENBLAS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS IMPORTED_TARGET openblas)
    endif()

    if(NOT OPENBLAS_FOUND)
        find_library(OPENBLAS_LIBRARY NAMES openblas libopenblas openblas64)
        find_path(OPENBLAS_INCLUDE_DIR NAMES cblas.h openblas/cblas.h)
        if(OPENBLAS_LIBRARY AND OPENBLAS_INCLUDE_DIR)
            set(OPENBLAS_FOUND TRUE)
            set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
            set(OPENBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
        endif()
    endif()

    if(OPENBLAS_FOUND)
        message(STATUS "OpenBLAS found: ${OPENBLAS_LIBRARIES}")
    else()
        message(WARNING "OpenBLAS not found. BLAS-accelerated CPU ops will use pure C++ fallback.")
    endif()
endif()

# ---- CUDA library target ----
if(TENSORN_BUILD_CUDA)
    enable_language(CUDA OPTIONAL)
    if(CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        add_library(tensorn_cuda STATIC
            core/CUDA/cuda_tensor.cu
            core/CUDA/elementwise.cu
            core/CUDA/reduction.cu
            core/CUDA/matmul.cu
            core/CUDA/convolution.cu
        )
        target_include_directories(tensorn_cuda PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(tensorn_cuda PUBLIC
            tensorn
            CUDA::cublas
            CUDA::cudart
        )
        target_compile_options(tensorn_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_target_properties(tensorn_cuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "70;75;80;86"
        )
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA compiler not found. CUDA acceleration disabled.")
        set(TENSORN_BUILD_CUDA OFF CACHE BOOL "" FORCE)
    endif()
endif()

# ---- Dependencies ----
include(FetchContent)

# nlohmann/json (required for static.hpp JSON I/O)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# zlib (required for cnpy NPZ support)
find_package(ZLIB REQUIRED)

# ---- Examples ----
if(TENSORN_BUILD_EXAMPLES)
    set(EXAMPLE_SOURCES exp1.cpp exp2.cpp exp3.cpp exp5.cpp)
    foreach(src ${EXAMPLE_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE tensorn nlohmann_json::nlohmann_json ZLIB::ZLIB)
    endforeach()
endif()

# ---- Tests ----
if(TENSORN_BUILD_TESTS)
    enable_testing()

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(tensorn_tests tests/test_main.cpp)
    target_link_libraries(tensorn_tests PRIVATE
        tensorn
        nlohmann_json::nlohmann_json
        ZLIB::ZLIB
        Catch2::Catch2WithMain
    )

    include(Catch)
    catch_discover_tests(tensorn_tests)

    # CUDA tests
    if(TENSORN_BUILD_CUDA)
        add_executable(tensorn_cuda_tests tests/test_cuda.cpp)
        target_link_libraries(tensorn_cuda_tests PRIVATE
            tensorn_cuda
            Catch2::Catch2WithMain
        )
        catch_discover_tests(tensorn_cuda_tests)
    endif()
endif()

# ---- Benchmarks ----
if(TENSORN_BUILD_BENCHMARKS)
    # Original combined benchmark (Native + BLAS + CUDA)
    add_executable(benchmark benchmark.cpp)
    target_link_libraries(benchmark PRIVATE tensorn)

    # MSVC O3 optimization flags
    if(MSVC)
        target_compile_options(benchmark PRIVATE /O2 /Ot /Ob2 /Oi /GL)
        target_link_options(benchmark PRIVATE /LTCG)
    endif()

    if(TENSORN_BUILD_CUDA)
        target_compile_definitions(benchmark PRIVATE TENSORN_CUDA_AVAILABLE)
        target_link_libraries(benchmark PRIVATE tensorn_cuda)
        find_package(CUDAToolkit REQUIRED)
        target_include_directories(benchmark PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(benchmark PRIVATE CUDA::cublas CUDA::cudart)
    endif()

    if(TENSORN_BUILD_OPENBLAS AND OPENBLAS_FOUND AND NOT MSVC)
        if(TARGET PkgConfig::OpenBLAS)
            target_link_libraries(benchmark PRIVATE PkgConfig::OpenBLAS)
        elseif(DEFINED OPENBLAS_LIBRARIES)
            target_link_libraries(benchmark PRIVATE ${OPENBLAS_LIBRARIES})
        endif()
    endif()

    # ---- Native-only baseline (no BLAS, no CUDA) ----
    add_executable(benchmark_native benchmark_native.cpp)
    target_link_libraries(benchmark_native PRIVATE tensorn)

    # ---- Native vs OpenBLAS (MSYS2 g++ O3 controlled comparison) ----
    add_executable(benchmark_blas benchmark_blas.cpp)
    target_link_libraries(benchmark_blas PRIVATE tensorn)

    if(TENSORN_BUILD_OPENBLAS AND OPENBLAS_FOUND)
        if(DEFINED OPENBLAS_INCLUDE_DIRS)
            target_include_directories(benchmark_blas PRIVATE ${OPENBLAS_INCLUDE_DIRS})
        endif()
        if(TARGET PkgConfig::OpenBLAS)
            target_link_libraries(benchmark_blas PRIVATE PkgConfig::OpenBLAS)
        elseif(DEFINED OPENBLAS_LIBRARIES)
            target_link_libraries(benchmark_blas PRIVATE ${OPENBLAS_LIBRARIES})
        endif()
    endif()

    # ---- Native vs CUDA (MSVC O3 controlled comparison, no BLAS) ----
    if(TENSORN_BUILD_CUDA)
        add_executable(benchmark_cuda benchmark_cuda.cpp)
        target_compile_definitions(benchmark_cuda PRIVATE TENSORN_CUDA_AVAILABLE)
        target_link_libraries(benchmark_cuda PRIVATE tensorn tensorn_cuda)
        find_package(CUDAToolkit REQUIRED)
        target_include_directories(benchmark_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(benchmark_cuda PRIVATE CUDA::cublas CUDA::cudart)
        set_target_properties(benchmark_cuda PROPERTIES
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()
endif()
