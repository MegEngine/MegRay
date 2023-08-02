add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/gtest" "third_party/gtest")

set(UTIL_SRCS "test_cuda.cpp" "test_hip.cpp" "test_cambricon.cpp")
set(TEST_SRCS "${UTIL_SRCS}" "test_main.cpp" "test_opr.cpp" "test_server_client.cpp" "test_shm.cpp")
set(PERF_SRCS "${UTIL_SRCS}" "performance.cpp")

add_executable(megray_test ${TEST_SRCS})
add_executable(performance ${PERF_SRCS})

target_link_libraries(megray_test megray gtest ${CMAKE_DL_LIBS})
target_link_libraries(performance megray pthread ${CMAKE_DL_LIBS})
