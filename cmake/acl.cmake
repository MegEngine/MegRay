
find_library(
  ACL_LIBRARY
  NAMES libascendcl.so
  PATHS ${ALTER_LD_LIBRARY_PATHS} "$ENV{ASCEND_TOOLKIT_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
  HINTS ${ALTER_LIBRARY_PATHS}
  PATH_SUFFIXES lib lib64
  DOC "ASCENDCL library."
)

if(ACL_LIBRARY STREQUAL "ACL_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find ASCENDCL Library")
endif()

get_filename_component(__found_acl_root "${ACL_LIBRARY}/../.." REALPATH)
find_path(
  ACL_INCLUDE_DIR
  NAMES acl/acl.h
  HINTS "$ENV{ASCEND_TOOLKIT_HOME}/include" ${__found_acl_root}
  PATH_SUFFIXES include
  DOC "Path to ACL include directory."
)

if(ACL_INCLUDE_DIR STREQUAL "ACL_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find ACL Header, please set up ASCEND_TOOLKIT_HOME correctly")
endif()

add_library(libascendcl SHARED IMPORTED)
set_target_properties(libascendcl PROPERTIES IMPORTED_LOCATION ${ACL_LIBRARY} INTERFACE_INCLUDE_DIRECTORIES ${ACL_INCLUDE_DIR})

message(STATUS "Found ACL HEADER: ${ACL_INCLUDE_DIR}")
message(STATUS "Found ACL LIBRARY: ${ACL_LIBRARY}")
