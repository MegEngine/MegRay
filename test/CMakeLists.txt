file(GLOB_RECURSE SOURCES "*.cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
add_executable(megray_test ${SOURCES})
target_link_libraries(megray_test megray libnccl libucp cudart gtest pthread dl rt)
