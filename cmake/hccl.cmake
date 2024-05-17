find_library(
  HCCL_LIBRARY
  NAMES libhccl.so
  PATHS ${ALTER_LD_LIBRARY_PATHS} "$ENV{ASCEND_TOOLKIT_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
  HINTS ${ALTER_LIBRARY_PATHS}
  PATH_SUFFIXES lib lib64
  DOC "HCCL library."
)

if(HCCL_LIBRARY STREQUAL "HCCL_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find HCCL Library")
endif()

get_filename_component(__found_hccl_root "${HCCL_LIBRARY}/../.." REALPATH)
find_path(
  HCCL_INCLUDE_DIR
  NAMES hccl/hccl.h
  HINTS "$ENV{ASCEND_TOOLKIT_HOME}/include" ${__found_hccl_root}
  PATH_SUFFIXES include
  DOC "Path to HCCL include directory."
)

if(HCCL_INCLUDE_DIR STREQUAL "HCCL_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find HCCL Header, please set up ASCEND_TOOLKIT_HOME correctly")
endif()

add_library(libhccl SHARED IMPORTED)
set_target_properties(libhccl PROPERTIES IMPORTED_LOCATION ${HCCL_LIBRARY} INTERFACE_INCLUDE_DIRECTORIES ${HCCL_INCLUDE_DIR})

message(STATUS "Found HCCL HEADER: ${HCCL_INCLUDE_DIR}")
message(STATUS "Found HCCL LIBRARY: ${HCCL_LIBRARY}")
