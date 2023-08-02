file(GLOB SRC_FILES "*.cpp")

add_library(megray STATIC)

list(APPEND MEGRAY_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/include ${PROJECT_SOURCE_DIR}/include)

target_sources(megray PRIVATE ${SRC_FILES})
target_include_directories(megray PUBLIC ${MEGRAY_INCLUDE_DIRS})

add_subdirectory(core)

target_link_libraries(megray PUBLIC megray_core)

if(${MEGRAY_WITH_NCCL})
    add_subdirectory(nccl)
    target_link_libraries(megray PUBLIC megray_nccl)
endif()

if(${MEGRAY_WITH_UCX})
    add_subdirectory(ucx)
    target_link_libraries(megray PUBLIC megray_ucx)
endif()

if(${MEGRAY_WITH_RCCL})
    add_subdirectory(rccl)
    target_link_libraries(megray PUBLIC megray_rccl)
endif()

if(${MEGRAY_WITH_SHM})
    add_subdirectory(shm)
    target_link_libraries(megray PUBLIC megray_shm)
endif()

if(${MEGRAY_WITH_CNCL})
    add_subdirectory(cncl)
    target_link_libraries(megray PUBLIC megray_cncl)
endif()


