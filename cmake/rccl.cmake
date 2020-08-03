set(RCCL_DIR ${PROJECT_SOURCE_DIR}/third_party/rccl CACHE STRING "rccl directory")
set(RCCL_BUILD_DIR ${PROJECT_BINARY_DIR}/third_party/rccl/build)
set(RCCL_LIBS ${RCCL_BUILD_DIR}/librccl.so)

if(${CMAKE_GENERATOR} STREQUAL "Unix Makefiles")
    set(MAKE_COMMAND "$(MAKE)")
else()
    set(MAKE_COMMAND make)
endif()

add_custom_command(
    OUTPUT ${RCCL_LIBS}
    COMMAND CXX=${HIP_HIPCC_EXECUTABLE} cmake -S ${RCCL_DIR} -B ${RCCL_BUILD_DIR}
    COMMAND ${MAKE_COMMAND} -C ${RCCL_BUILD_DIR}
    WORKING_DIRECTORY ${RCCL_DIR}
    VERBATIM)

file(MAKE_DIRECTORY ${RCCL_BUILD_DIR}/include)

add_custom_target(rccl DEPENDS ${RCCL_LIBS})
add_library(librccl SHARED IMPORTED GLOBAL)
add_dependencies(librccl rccl)
set_target_properties(
    librccl PROPERTIES
    IMPORTED_LOCATION ${RCCL_LIBS}
)
target_include_directories(librccl INTERFACE "${RCCL_BUILD_DIR};${HIP_INCLUDE_DIRS}")
target_link_directories(librccl INTERFACE ${HIP_LIB_INSTALL_DIR})
target_link_libraries(librccl INTERFACE ${CMAKE_DL_LIBS} rt numa)
target_compile_options(librccl
    INTERFACE ${HIP_CONFIG_RESULT_LIST})
