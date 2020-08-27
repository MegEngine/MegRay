cmake_minimum_required(VERSION 3.9.0)
project(MegRay)

enable_language(CXX)

include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(MEGRAY_WITH_NCCL "" ON)
option(MEGRAY_WITH_UCX "" ON)
option(MEGRAY_WITH_RCCL "" OFF)
option(MEGRAY_WITH_TEST "Enable test for MegRay." OFF)

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
    if(NOT DEFINED ROCM_HOME)
        set(ROCM_HOME $ENV{ROCM_HOME})
    endif()

    if(NOT DEFINED ROCM_HOME)
        set(ROCM_HOME /opt/rocm)
    endif()

    list(APPEND CMAKE_PREFIX_PATH
                ${ROCM_HOME}
                ${ROCM_HOME}/hip
                ${ROCM_HOME}/hcc)
    find_package(HIP REQUIRED)

    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} "--cpp_config"
        OUTPUT_VARIABLE HIP_CONFIG_RESULT
    )

    string(REPLACE " " ";" HIP_COMPILE_OPTIONS ${HIP_CONFIG_RESULT})
endif()

add_subdirectory(src)

if(${MEGRAY_WITH_TEST})
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif()