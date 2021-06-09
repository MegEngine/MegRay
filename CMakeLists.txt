cmake_minimum_required(VERSION 3.13)
project(MegRay)

cmake_policy(SET CMP0077 NEW)

enable_language(CXX)

include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(MEGRAY_WITH_NCCL "Enable NCCL Backend" ON)
option(MEGRAY_WITH_UCX "Enable UCX Backend (Not recommend)" OFF)
option(MEGRAY_WITH_SHM "Enable SHM Backend" ON)
option(MEGRAY_WITH_TEST "Enable test for MegRay." OFF)
set(MEGRAY_LOG_LEVEL 2 CACHE STRING "MegRay log level")
set(MEGRAY_CUDA_ARCHITECTURES "" CACHE STRING "MegRay CUDA Architectures")
set(MEGRAY_CUDA_GENCODE "" CACHE STRING "Overwrite -gencode specifications for CUDA")

# if RCCL_PATH is specified, force open MEGRAY_WITH_RCCL
cmake_dependent_option(MEGRAY_WITH_RCCL "Build with rccl" OFF "NOT DEFINED RCCL_PATH;NOT DEFINED ENV{RCCL_PATH}" ON)

if(MEGRAY_WITH_NCCL OR MEGRAY_WITH_UCX OR MEGRAY_WITH_SHM)
    set(MEGRAY_WITH_CUDA ON)
else()
    set(MEGRAY_WITH_CUDA OFF)
endif()

set(MEGRAY_WITH_HIP ${MEGRAY_WITH_RCCL})

configure_file("${PROJECT_SOURCE_DIR}/include/megray/config.h.in" "${PROJECT_BINARY_DIR}/include/megray/config.h")

if(${MEGRAY_WITH_CUDA} AND ${MEGRAY_WITH_HIP})
    message(FATAL_ERROR "Coexistence of CUDA and HIP is not supported yet")
endif()

if(${MEGRAY_WITH_CUDA})
    find_package(CUDA)
    enable_language(CUDA)
    get_filename_component(CUDA_HOME ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_HOME ${CUDA_HOME} DIRECTORY)

    if ("${MEGRAY_CUDA_ARCHITECTURES}" STREQUAL "" AND "${MEGRAY_CUDA_GENCODE}" STREQUAL "")
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.1.0")
            set(MEGRAY_CUDA_ARCHITECTURES_DEFAULT 61-real 70-real 75-real 80-real 86)
        elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11.0.0")
            set(MEGRAY_CUDA_ARCHITECTURES_DEFAULT 61-real 70-real 75-real 80)
        elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "10.0.0")
            set(MEGRAY_CUDA_ARCHITECTURES_DEFAULT 52-real 60-real 61-real 70-real 75)
        elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "9.0.0")
            set(MEGRAY_CUDA_ARCHITECTURES_DEFAULT 52-real 60-real 61-real 70)
        else()
            set(MEGRAY_CUDA_ARCHITECTURES_DEFAULT 35-real 52-real 60-real 61)
        endif()
        set(MEGRAY_CUDA_ARCHITECTURES ${MEGRAY_CUDA_ARCHITECTURES_DEFAULT} CACHE STRING "" FORCE)
    else()
        set(MEGRAY_CUDA_ARCHITECTURES OFF CACHE STRING "" FORCE)
    endif()
endif()

if(${MEGRAY_WITH_HIP})
    if(NOT DEFINED HIP_PATH)
        set(DEFAULT_HIP_PATH "/opt/rocm/hip")
        if (DEFINED $ENV{HIP_PATH})
            set(DEFAULT_HIP_PATH $ENV{HIP_PATH})
        endif()
        set(HIP_PATH ${DEFAULT_HIP_PATH} CACHE STRING "")
    endif()

    list(APPEND CMAKE_PREFIX_PATH
                ${HIP_PATH}/../
                ${HIP_PATH}
                ${HIP_PATH}/../hcc)
    find_package(HIP REQUIRED)

    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} "--cpp_config"
        OUTPUT_VARIABLE HIP_COMPILE_OPTIONS
    )

    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} "--compiler"
        OUTPUT_VARIABLE HIP_COMPILER
    )

separate_arguments(HIP_COMPILE_OPTIONS)

    if(${HIP_COMPILER} STREQUAL "hcc")
        set(HIP_LIBRARY hip_hcc)
    elseif(${HIP_COMPILER} STREQUAL "clang")
        set(HIP_LIBRARY amdhip64)
    else()
        message(FATAL_ERROR "Unknown HIP compiler ${HIP_COMPILER}")
    endif()
endif()

add_subdirectory(src)

if(${MEGRAY_WITH_TEST})
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif()
