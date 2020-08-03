set(NCCL_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl CACHE STRING "nccl directory")
set(NCCL_BUILD_DIR ${PROJECT_BINARY_DIR}/third_party/nccl/build)
set(NCCL_LIBS ${NCCL_BUILD_DIR}/lib/libnccl_static.a)

enable_language(CUDA)

get_filename_component(NCCL_BUILD_DIR ${NCCL_BUILD_DIR} ABSOLUTE)

if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    set(MAKE_COMMAND make)
else()
    set(MAKE_COMMAND "$(MAKE)")
endif()

add_custom_command(
    OUTPUT ${NCCL_LIBS}
    COMMAND ${MAKE_COMMAND} -C src ${NCCL_BUILD_DIR}/include/nccl.h BUILDDIR=${NCCL_BUILD_DIR}
    COMMAND ${MAKE_COMMAND} src.build NVCC_GENCODE=${MEGRAY_CUDA_GENCODE} BUILDDIR=${NCCL_BUILD_DIR} CUDA_HOME=${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}
    WORKING_DIRECTORY ${NCCL_DIR}
    VERBATIM
)

file(MAKE_DIRECTORY ${NCCL_BUILD_DIR}/include)
add_custom_target(nccl DEPENDS ${NCCL_LIBS})
add_library(libnccl STATIC IMPORTED GLOBAL)
add_dependencies(libnccl nccl)
set_target_properties(
    libnccl PROPERTIES
    IMPORTED_LOCATION ${NCCL_LIBS}
)
target_include_directories(libnccl INTERFACE "${NCCL_BUILD_DIR}/include")
target_link_libraries(libnccl INTERFACE cudart_static ${CMAKE_DL_LIBS} rt)
