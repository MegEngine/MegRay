file(GLOB SRC_FILES "*.cpp")
file(GLOB CUDA_SRC_FILES "*.cu")

find_package(ucx)

add_library(megray_ucx STATIC)
set_target_properties(megray_ucx PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_sources(megray_ucx PRIVATE ${SRC_FILES} ${CUDA_SRC_FILES})
target_include_directories(megray_ucx PUBLIC ${PROJECT_BINARY_DIR}/include)
target_include_directories(megray_ucx PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(megray_ucx PUBLIC "${CUDA_HOME}/include")
target_link_libraries(megray_ucx INTERFACE megray_core libucx ${CUDA_LIBRARIES})
