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

if(${MEGRAY_WITH_CUDA})
    enable_language(CUDA)
    get_filename_component(CUDA_HOME ${CMAKE_CUDA_COMPILER} DIRECTORY)
    get_filename_component(CUDA_HOME ${CUDA_HOME} DIRECTORY)
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