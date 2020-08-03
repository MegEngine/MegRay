file(GLOB SRC_FILES "*.cpp")

add_library(megray STATIC)

target_sources(megray PRIVATE ${SRC_FILES})
target_include_directories(megray PUBLIC ${PROJECT_BINARY_DIR}/include)
target_include_directories(megray PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(core)

target_link_libraries(megray INTERFACE megray_core)

if(${MEGRAY_WITH_NCCL})
    add_subdirectory(nccl)
    target_link_libraries(megray INTERFACE megray_nccl)
endif()

if(${MEGRAY_WITH_UCX})
    add_subdirectory(ucx)
    target_link_libraries(megray INTERFACE megray_ucx)
endif()

if(${MEGRAY_WITH_RCCL})
    add_subdirectory(rccl)
    target_link_libraries(megray INTERFACE megray_rccl)
endif()