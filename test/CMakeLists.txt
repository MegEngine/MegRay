add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/gtest" "third_party/gtest")

file(GLOB SOURCES "*.cpp")
add_executable(megray_test ${SOURCES})

target_link_libraries(megray_test megray gtest ${CMAKE_DL_LIBS})
