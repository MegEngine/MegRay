file(GLOB SRC_FILES "*.cpp")

add_library(megray_core OBJECT)
target_sources(megray_core PRIVATE ${SRC_FILES})
target_include_directories(megray_core PUBLIC ${MEGRAY_INCLUDE_DIRS})
