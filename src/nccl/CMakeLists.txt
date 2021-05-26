file(GLOB SRC_FILES "*.cpp")

include(nccl)

add_library(megray_nccl OBJECT)
target_sources(megray_nccl PRIVATE ${SRC_FILES})
target_link_libraries(megray_nccl PUBLIC nccl_static megray_core)
