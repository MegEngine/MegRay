include(ExternalProject)

option(UCX_WITH_GDRCOPY "Build ucx with gdrcopy" OFF)

if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    set(MAKE_COMMAND make)
else()
    set(MAKE_COMMAND "$(MAKE)")
endif()

get_filename_component(CUDA_HOME ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_HOME ${CUDA_HOME} DIRECTORY)

if(UCX_WITH_GDRCOPY)
    set(GDRCOPY_DIR ${PROJECT_SOURCE_DIR}/third_party/gdrcopy)
    set(GDRCOPY_BUILD_DIR ${PROJECT_BINARY_DIR}/third_party/gdrcopy)
    set(GDRCOPY_LIB ${GDRCOPY_BUILD_DIR}/lib64/libgdrapi.so)
    ExternalProject_add(
        gdrcopy
        SOURCE_DIR ${GDRCOPY_DIR}
        PREFIX ${GDRCOPY_BUILD_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${MAKE_COMMAND} -C ${GDRCOPY_DIR} lib CUDA_HOME=${CUDA_HOME}
        INSTALL_COMMAND ${MAKE_COMMAND} -C ${GDRCOPY_DIR} lib_install PREFIX="" DESTDIR=${GDRCOPY_BUILD_DIR}
        BUILD_BYPRODUCTS ${GDRCOPY_LIB}
    )

    ExternalProject_Add_Step(
        gdrcopy
        clean
        COMMAND make clean
        WORKING_DIRECTORY <SOURCE_DIR>
        DEPENDEES install
    )

    set(GDRCOPY_INC ${GDRCOPY_BUILD_DIR}/include)
    file(MAKE_DIRECTORY ${GDRCOPY_INC})

    add_library(libgdrcopy SHARED IMPORTED GLOBAL)
    add_dependencies(libgdrcopy gdrcopy)
    set_target_properties(
        libgdrcopy PROPERTIES
        IMPORTED_LOCATION ${GDRCOPY_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${GDRCOPY_INC}
    )

    list(APPEND UCX_CONFIGURE_ARGS --with-gdrcopy=${GDRCOPY_BUILD_DIR})
endif()

set(UCX_DIR ${PROJECT_SOURCE_DIR}/third_party/ucx)
set(UCX_BUILD_DIR ${PROJECT_BINARY_DIR}/third_party/ucx)

file(STRINGS ${UCX_DIR}/configure.ac UCX_SO_LINE REGEX "\\[libucx_so_version\\]")
string(REGEX MATCH "([0-9]+):([0-9]+):([0-9]+)" _ ${UCX_SO_LINE})
set(UCX_SO_VER_MAJOR ${CMAKE_MATCH_1})
set(UCX_SO_VER_MINOR ${CMAKE_MATCH_2})
set(UCX_SO_VER_PATCH ${CMAKE_MATCH_3})

set(UCP_LIB ${UCX_BUILD_DIR}/lib/libucp.so.${UCX_SO_VER_MAJOR})
set(UCT_LIB ${UCX_BUILD_DIR}/lib/libuct.so.${UCX_SO_VER_MAJOR})
set(UCS_LIB ${UCX_BUILD_DIR}/lib/libucs.so.${UCX_SO_VER_MAJOR})
set(UCM_LIB ${UCX_BUILD_DIR}/lib/libucm.so.${UCX_SO_VER_MAJOR})

ExternalProject_add(
    ucx
    SOURCE_DIR ${UCX_DIR}
    PREFIX ${UCX_BUILD_DIR}
    CONFIGURE_COMMAND ${UCX_DIR}/configure --enable-mt --with-pic --disable-static
         --with-cuda=${CUDA_HOME} --disable-numa --prefix=${UCX_BUILD_DIR} ${UCX_CONFIGURE_ARGS}
    BUILD_COMMAND ${MAKE_COMMAND} all
    INSTALL_COMMAND ${MAKE_COMMAND} install
    BUILD_BYPRODUCTS ${UCP_LIB} ${UCT_LIB} ${UCS_LIB} ${UCM_LIB}
)

ExternalProject_Add_Step(
    ucx
    autogen
    COMMAND ./autogen.sh
    WORKING_DIRECTORY <SOURCE_DIR>
    DEPENDERS configure
)

if(UCX_WITH_GDRCOPY)
    add_dependencies(ucx gdrcopy)
endif()

set(UCX_INC ${UCX_BUILD_DIR}/include)
file(MAKE_DIRECTORY ${UCX_INC})

add_library(libucp SHARED IMPORTED GLOBAL)
add_dependencies(libucp ucx)
set_target_properties(
    libucp PROPERTIES
    IMPORTED_LOCATION ${UCP_LIB}
)

add_library(libuct SHARED IMPORTED GLOBAL)
add_dependencies(libuct ucx)
set_target_properties(
    libuct PROPERTIES
    IMPORTED_LOCATION ${UCT_LIB}
)

add_library(libucs SHARED IMPORTED GLOBAL)
add_dependencies(libucs ucx)
set_target_properties(
    libucs PROPERTIES
    IMPORTED_LOCATION ${UCS_LIB}
)

add_library(libucm SHARED IMPORTED GLOBAL)
add_dependencies(libucm ucx)
set_target_properties(
    libucm PROPERTIES
    IMPORTED_LOCATION ${UCM_LIB}
)

add_library(libucx INTERFACE)
target_link_libraries(libucx INTERFACE libucp libuct libucs libucm)
if(UCX_WITH_GDRCOPY)
    target_link_libraries(libucx INTERFACE libgdrcopy)
endif()
target_include_directories(libucx INTERFACE ${UCX_INC})
