file(GLOB SRC_FILES "*.cpp")

include(rccl)

add_library(megray_rccl OBJECT)
target_sources(megray_rccl PRIVATE ${SRC_FILES})
target_link_libraries(megray_rccl PUBLIC megray_core librccl)
target_include_directories(megray_rccl PUBLIC ${MEGRAY_INCLUDE_DIRS})
target_include_directories(megray_rccl PUBLIC ${HIP_INCLUDE_DIRS})
target_compile_options(megray_rccl PUBLIC ${HIP_COMPILE_OPTIONS})
target_link_directories(megray_rccl PUBLIC ${HIP_LIB_INSTALL_DIR} ${RCCL_BUILD_DIR})
