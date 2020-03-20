cmake_minimum_required(VERSION 3.9.0)
project(MegRay VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(NOT MSVC)
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Dqc <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> Dq  <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>")
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found in PATH")
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(${CMAKE_THREAD_LIBS_INIT} STREQUAL "-pthread" AND MGE_WITH_CUDA)
    set_property(TARGET Threads::Threads
        PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>"
        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
link_directories(${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})

set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall,-Wextra,-Werror -Xfatbin -compress-all")
set(CMAKE_CUDA_HOST_COMPILER $(CMAKE_CXX_COMPILER))

find_program(CCACHE_BIN ccache)
if(CCACHE_BIN)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_BIN})
    if(MGE_WITH_CUDA AND NOT ${CMAKE_VERSION} VERSION_LESS "3.10.0")
        message("-- Using ccache as CMAKE_CUDA_COMPILER_LAUNCHER")
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_BIN})
    endif()
endif()

option(MEGRAY_CUDA_USE_STATIC "Enable MegEngine CUDA static linking." ON)
set(MEGRAY_CUDA_GENCODE "" CACHE STRING "Overwrite -gencode specifications for CUDA")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMEGDNN_THREADS_512=0")
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

if(MEGRAY_CUDA_USE_STATIC)
    list(APPEND CUDA_LIBS cudart_static)
else()
    list(APPEND CUDA_LIBS cudart)
endif()

include(cmake/ucx.cmake)
include(cmake/nccl.cmake)

option(MEGRAY_TEST "Enable test for MegRay." OFF)
if(MEGRAY_TEST)
    include(cmake/gtest.cmake)
    add_subdirectory(test)
endif()

include_directories("${CMAKE_CURRRENT_BINARY_DIR}/src")
file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE CUDA_SRC_FILES "src/*.cu")

add_library(megray STATIC ${SRC_FILES} ${CUDA_SRC_FILES})
target_link_libraries(megray libucx libnccl ${CUDA_LIBS})
target_include_directories(megray INTERFACE src)
target_compile_options(megray PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>"
    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-unused-parameter>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-varible>"
    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-unused-variable>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-result>"
    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-unused-result>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-but-set-variable>"
    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-unused-but-set-variable>"
)
