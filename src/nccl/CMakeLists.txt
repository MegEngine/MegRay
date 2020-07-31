file(GLOB SRC_FILES "*.cpp")

find_package(nccl)

add_library(megray_nccl STATIC)
target_sources(megray_nccl PRIVATE ${SRC_FILES})
target_include_directories(megray_nccl PUBLIC ${PROJECT_BINARY_DIR}/include)
target_include_directories(megray_nccl PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(megray_nccl INTERFACE megray_core PRIVATE libnccl)
