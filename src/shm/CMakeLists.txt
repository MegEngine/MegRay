file(GLOB SRC_FILES "*.cpp")
file(GLOB CUDA_SRC_FILES "*.cu")

add_library(megray_shm OBJECT)
set_target_properties(megray_shm PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_sources(megray_shm PRIVATE ${SRC_FILES} ${CUDA_SRC_FILES})
target_include_directories(megray_shm PUBLIC ${MEGRAY_INCLUDE_DIRS})
target_include_directories(megray_shm PUBLIC "${CUDA_HOME}/include")
target_link_libraries(megray_shm PUBLIC megray_core ${CUDA_LIBRARIES})
