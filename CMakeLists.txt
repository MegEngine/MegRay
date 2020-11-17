cmake_minimum_required(VERSION 3.9.0)
project(MegRay)

cmake_policy(SET CMP0077 NEW)

enable_language(CXX)

include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(MEGRAY_WITH_NCCL "" ON)
option(MEGRAY_WITH_UCX "" OFF)
option(MEGRAY_WITH_RCCL "" OFF)
option(MEGRAY_WITH_TEST "Enable test for MegRay." OFF)
set(MEGRAY_LOG_LEVEL 2 CACHE STRING "MegRay log level")

if(MEGRAY_WITH_NCCL OR MEGRAY_WITH_UCX)
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
    enable_language(CUDA)
    get_filename_component(CUDA_HOME ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_HOME ${CUDA_HOME} DIRECTORY)

    set(MEGRAY_CUDA_GENCODE "" CACHE STRING "Overwrite -gencode specifications for CUDA")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_75,code=compute_75")
    elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "9.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "9.0.0")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_70,code=compute_70")
    else()
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_35,code=sm_35")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
        set(MEGRAY_CUDA_GENCODE "${MEGRAY_CUDA_GENCODE} -gencode arch=compute_61,code=compute_61")
    endif()
endif()

if(${MEGRAY_WITH_HIP})
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "")
    endif()

    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH /opt/rocm)
    endif()

    list(APPEND CMAKE_PREFIX_PATH
                ${ROCM_PATH}
                ${ROCM_PATH}/hip
                ${ROCM_PATH}/hcc)
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
