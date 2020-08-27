file(GLOB SRC_FILES "*.cpp")
file(GLOB CUDA_SRC_FILES "*.cu")

include(ucx)

add_library(megray_ucx OBJECT)
set_target_properties(megray_ucx PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_sources(megray_ucx PRIVATE ${SRC_FILES} ${CUDA_SRC_FILES})
target_include_directories(megray_ucx PUBLIC ${MEGRAY_INCLUDE_DIRS})
target_include_directories(megray_ucx PUBLIC "${CUDA_HOME}/include")
target_link_libraries(megray_ucx PUBLIC libucx megray_core ${CUDA_LIBRARIES})
