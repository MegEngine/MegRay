if(DEFINED RCCL_PATH)
    set(RCCL_DIR "${RCCL_PATH}")
elseif(DEFINED ENV{RCCL_PATH})
    set(RCCL_DIR "$ENV{RCCL_PATH}")
endif()

if(DEFINED RCCL_DIR AND NOT EXISTS "${RCCL_DIR}/include/rccl.h")
    message(WARNING "Cannot find RCCL in ${RCCL_DIR}, ignore RCCL_PATH")
    unset(RCCL_DIR)
endif()

if(DEFINED RCCL_DIR)
    file(READ "${RCCL_DIR}/include/rccl.h" RCCL_HEADER_CONTENT)
    string(REGEX MATCH "#define NCCL_VERSION_CODE [0-9]+" RCCL_VERSION_CODE "${RCCL_HEADER_CONTENT}")
    separate_arguments(RCCL_VERSION_CODE)
    list(GET RCCL_VERSION_CODE 2 RCCL_VERSION_CODE)
    if(${RCCL_VERSION_CODE} LESS 2700)
        message(WARNING "rccl version less than 2.7, ignore RCCL_PATH")
        unset(RCCL_DIR)
    endif()
endif()

if(NOT DEFINED RCCL_DIR)
    set(RCCL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/rccl CACHE STRING "rccl directory")
    set(RCCL_BUILD_DIR ${PROJECT_BINARY_DIR}/third_party/rccl/build)
    set(RCCL_LIBS ${RCCL_BUILD_DIR}/librccl.so)
    set(RCCL_INCLUDE_DIR ${RCCL_BUILD_DIR})
    message(STATUS "RCCL_PATH not avaliable, build from source")
else()
    set(RCCL_LIBS ${RCCL_DIR}/lib/librccl.so)
    set(RCCL_INCLUDE_DIR ${RCCL_DIR}/include)
endif()

if(${CMAKE_GENERATOR} STREQUAL "Unix Makefiles")
    set(MAKE_COMMAND "$(MAKE)")
else()
    set(MAKE_COMMAND make)
endif()

if(NOT DEFINED RCCL_DIR)
    add_custom_command(
        OUTPUT ${RCCL_LIBS}
        COMMAND CXX=${HIP_HIPCC_EXECUTABLE} ${CMAKE_COMMAND} -S ${RCCL_SOURCE_DIR} -B ${RCCL_BUILD_DIR} -DCMAKE_CXX_COMPILER_FORCED=ON
        COMMAND ${MAKE_COMMAND} -C ${RCCL_BUILD_DIR}
        WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
        VERBATIM)
    file(MAKE_DIRECTORY ${RCCL_INCLUDE_DIR})
endif()

add_custom_target(rccl DEPENDS ${RCCL_LIBS})
add_library(librccl SHARED IMPORTED GLOBAL)
add_dependencies(librccl rccl)
set_target_properties(
    librccl PROPERTIES
    IMPORTED_LOCATION ${RCCL_LIBS}
)
target_include_directories(librccl INTERFACE ${RCCL_INCLUDE_DIR} ${HIP_INCLUDE_DIRS})
target_link_directories(librccl INTERFACE ${HIP_LIB_INSTALL_DIR})
target_link_libraries(librccl INTERFACE ${CMAKE_DL_LIBS} rt numa ${HIP_LIBRARY})
target_compile_options(librccl
    INTERFACE ${HIP_COMPILE_OPTIONS})
