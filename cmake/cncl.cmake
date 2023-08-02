find_library(
  CNCL_LIBRARY
  NAMES libcncl.so
  PATHS ${ALTER_LD_LIBRARY_PATHS} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
  HINTS ${ALTER_LIBRARY_PATHS}
  PATH_SUFFIXES lib lib64
  DOC "CNCL library.")

if(CNCL_LIBRARY STREQUAL "CNCL_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find CNCL Library")
endif()

get_filename_component(__found_cncl_root "${CNCL_LIBRARY}/../.." REALPATH)
find_path(
  CNCL_INCLUDE_DIR
  NAMES cncl.h
  HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cncl_root}
  PATH_SUFFIXES include
  DOC "Path to CNCL include directory.")

if(CNCL_INCLUDE_DIR STREQUAL "CNCL_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find CNCL Library")
endif()

file(STRINGS "${CNCL_INCLUDE_DIR}/cncl.h" CNCL_MAJOR
     REGEX "^#define CNCL_MAJOR [0-9]+.*$")
file(STRINGS "${CNCL_INCLUDE_DIR}/cncl.h" CNCL_MINOR
     REGEX "^#define CNCL_MINOR [0-9]+.*$")
file(STRINGS "${CNCL_INCLUDE_DIR}/cncl.h" CNCL_PATCH
     REGEX "^#define CNCL_PATCHLEVEL [0-9]+.*$")

string(REGEX REPLACE "^#define CNCL_MAJOR ([0-9]+).*$" "\\1" CNCL_VERSION_MAJOR
                     "${CNCL_MAJOR}")
string(REGEX REPLACE "^#define CNCL_MINOR ([0-9]+).*$" "\\1" CNCL_VERSION_MINOR
                     "${CNCL_MINOR}")
string(REGEX REPLACE "^#define CNCL_PATCHLEVEL ([0-9]+).*$" "\\1" CNCL_VERSION_PATCH
                     "${CNCL_PATCH}")
set(CNCL_VERSION_STRING
    "${CNCL_VERSION_MAJOR}.${CNCL_VERSION_MINOR}.${CNCL_VERSION_PATCH}")

add_library(libcncl SHARED IMPORTED)

set_target_properties(
  libcncl PROPERTIES IMPORTED_LOCATION ${CNCL_LIBRARY} INTERFACE_INCLUDE_DIRECTORIES
                                                       ${CNCL_INCLUDE_DIR})

message(
  STATUS "Found CNCL: ${__found_cncl_root} (found version: ${CNCL_VERSION_STRING})")

