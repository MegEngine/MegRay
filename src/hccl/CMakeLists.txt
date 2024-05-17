file(GLOB SRC_FILES "*.cpp")

add_library(megray_hccl OBJECT)
target_sources(megray_hccl PRIVATE ${SRC_FILES})
target_link_libraries(megray_hccl PUBLIC libhccl libascendcl megray_core)
