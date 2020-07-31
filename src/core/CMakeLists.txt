file(GLOB SRC_FILES "*.cpp")

add_library(megray_core STATIC)
target_sources(megray_core PRIVATE ${SRC_FILES})
target_include_directories(megray_core PUBLIC ${PROJECT_BINARY_DIR}/include)
target_include_directories(megray_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
