cmake_minimum_required(VERSION 3.20)
project(TensorN
    VERSION 0.1.0
    DESCRIPTION "A safer, more efficient, and more universal native C++ Tensor library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Header-only library target ----
add_library(tensorn INTERFACE)
target_include_directories(tensorn INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(tensorn INTERFACE cxx_std_26)

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
option(TENSORN_BUILD_EXAMPLES "Build example programs" ON)
if(TENSORN_BUILD_EXAMPLES)
    set(EXAMPLE_SOURCES exp1.cpp exp2.cpp exp3.cpp exp4.cpp exp5.cpp)
    foreach(src ${EXAMPLE_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE tensorn nlohmann_json::nlohmann_json ZLIB::ZLIB)
    endforeach()
endif()

# ---- Tests ----
option(TENSORN_BUILD_TESTS "Build tests" ON)
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
endif()
