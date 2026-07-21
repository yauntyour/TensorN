# FindOpenBLAS.cmake
# Find OpenBlas library
#
# This module defines:
#  OpenBLAS_FOUND        - True if OpenBlas is found
#  OpenBLAS_INCLUDE_DIRS - Include directories for OpenBlas
#  OpenBLAS_LIBRARIES    - Libraries to link against
#  OpenBLAS_VERSION      - Version of OpenBlas (if found)

# Try to find OpenBlas using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_OpenBLAS QUIET openblas)
endif()

# Find include directory
find_path(OpenBLAS_INCLUDE_DIR
    NAMES cblas.h openblas/cblas.h
    HINTS
        ${PC_OpenBLAS_INCLUDEDIR}
        ${PC_OpenBLAS_INCLUDE_DIRS}
        $ENV{OpenBLAS_HOME}/include
        $ENV{OPENBLAS_HOME}/include
        $ENV{VCPKG_ROOT}/installed/x64-windows/include
    PATHS
        C:/vcpkg/installed/x64-windows/include
        D:/x64/vcpkg/installed/x64-windows/include
        /usr/include
        /usr/local/include
        /opt/OpenBLAS/include
        /opt/homebrew/opt/openblas/include
        C:/OpenBLAS/include
        C:/Program\ Files/OpenBLAS/include
)

# Find library
find_library(OpenBLAS_LIBRARY
    NAMES openblas libopenblas
    HINTS
        ${PC_OpenBLAS_LIBDIR}
        ${PC_OpenBLAS_LIBRARY_DIRS}
        $ENV{OpenBLAS_HOME}/lib
        $ENV{OPENBLAS_HOME}/lib
        $ENV{VCPKG_ROOT}/installed/x64-windows/lib
    PATHS
        C:/vcpkg/installed/x64-windows/lib
        D:/x64/vcpkg/installed/x64-windows/lib
        /usr/lib
        /usr/local/lib
        /opt/OpenBLAS/lib
        /opt/homebrew/opt/openblas/lib
        C:/OpenBLAS/lib
        C:/Program\ Files/OpenBLAS/lib
)

# Try to find version
if(OpenBLAS_INCLUDE_DIR)
    # Try to read version from openblas_config.h
    if(EXISTS "${OpenBLAS_INCLUDE_DIR}/openblas_config.h")
        file(READ "${OpenBLAS_INCLUDE_DIR}/openblas_config.h" _openblas_config)
        string(REGEX MATCH "OPENBLAS_VERSION[ \t]+([0-9]+\\.[0-9]+\\.[0-9]+)" _match "${_openblas_config}")
        if(_match)
            set(OpenBLAS_VERSION "${CMAKE_MATCH_1}")
        endif()
    endif()
    
    # Alternative: try to read from cblas.h
    if(NOT OpenBLAS_VERSION AND EXISTS "${OpenBLAS_INCLUDE_DIR}/cblas.h")
        file(READ "${OpenBLAS_INCLUDE_DIR}/cblas.h" _cblas_h)
        string(REGEX MATCH "OPENBLAS_VERSION[ \t]+([0-9]+\\.[0-9]+\\.[0-9]+)" _match "${_cblas_h}")
        if(_match)
            set(OpenBLAS_VERSION "${CMAKE_MATCH_1}")
        endif()
    endif()
endif()

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
    REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
    VERSION_VAR OpenBLAS_VERSION
)

if(OpenBLAS_FOUND)
    set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
    set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
    
    # Create imported target
    if(NOT TARGET OpenBLAS::OpenBLAS)
        add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
        set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
            IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
        )
    endif()
endif()

# Mark as advanced
mark_as_advanced(OpenBLAS_INCLUDE_DIR OpenBLAS_LIBRARY)