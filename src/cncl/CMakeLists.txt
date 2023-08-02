file(GLOB SRC_FILES "*.cpp")

add_library(megray_cncl OBJECT)
target_sources(megray_cncl PRIVATE ${SRC_FILES})
target_link_libraries(megray_cncl PUBLIC libcnrt libcncl megray_core)
