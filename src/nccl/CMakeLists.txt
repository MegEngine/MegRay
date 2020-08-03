file(GLOB SRC_FILES "*.cpp")

include(nccl)

add_library(megray_nccl STATIC)
target_sources(megray_nccl PRIVATE ${SRC_FILES})
target_include_directories(megray_nccl PUBLIC ${PROJECT_BINARY_DIR}/include)
target_include_directories(megray_nccl PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(megray_nccl PRIVATE libnccl INTERFACE megray_core)
