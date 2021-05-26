set(NCCL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl)
set(NCCL_BINARY_DIR ${PROJECT_BINARY_DIR}/third_party/nccl)

enable_language(CUDA)

# common cu flags/include directories
set(NCCL_CUDA_FLAGS -maxrregcount=96 -rdc=true)
if (DEFINED MEGRAY_CUDA_GENCODE)
    set(NCCL_CUDA_FLAGS ${NCCL_CUDA_FLAGS} "SHELL:${MEGRAY_CUDA_GENCODE}")
endif()

add_library(nccl_device_interface INTERFACE)
target_compile_options(nccl_device_interface INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${NCCL_CUDA_FLAGS}>)
target_include_directories(nccl_device_interface INTERFACE "${NCCL_SOURCE_DIR}/src/include" "${NCCL_BINARY_DIR}/include" "${CUDA_HOME}/include")

# generate device objs
set(NCCL_DEVICE_OBJS "")
set(BASES "sendrecv;all_reduce;all_gather;broadcast;reduce;reduce_scatter")
set(OPS "sum;prod;min;max")
set(DTS "i8;u8;i32;u32;i64;u64;f16;f32;f64")
foreach(BASE IN ITEMS ${BASES})
    set(OPN 0)
    foreach(OP IN ITEMS ${OPS})
        set(DTN 0)
        foreach(DT IN ITEMS ${DTS})
            set(NCCL_DEVICE_OBJ nccl_device_${BASE}_${OP}_${DT})
            add_library(${NCCL_DEVICE_OBJ} OBJECT "${NCCL_SOURCE_DIR}/src/collectives/device/${BASE}.cu")
            target_compile_definitions(${NCCL_DEVICE_OBJ} PRIVATE NCCL_OP=${OPN} NCCL_TYPE=${DTN})
            target_link_libraries(${NCCL_DEVICE_OBJ} PRIVATE nccl_device_interface)
            set(NCCL_DEVICE_OBJS ${NCCL_DEVICE_OBJS} ${NCCL_DEVICE_OBJ})
            math(EXPR DTN "${DTN}+1")
        endforeach(DT)
        math(EXPR OPN "${OPN}+1")
    endforeach(OP)
endforeach(BASE)

add_library(nccl_device_functions OBJECT "${NCCL_SOURCE_DIR}/src/collectives/device/functions.cu")
target_link_libraries(nccl_device_functions PRIVATE nccl_device_interface)

set(NCCL_DEVICE_OBJS ${NCCL_DEVICE_OBJS} nccl_device_functions)

# compile host codes and apply device-link
file(GLOB_RECURSE NCCL_HOST_SRCS CONFIGURE_DEPENDS "${NCCL_SOURCE_DIR}/src/*.cc")
add_library(nccl_static STATIC ${NCCL_HOST_SRCS})
target_include_directories(nccl_static
    PUBLIC "${NCCL_BINARY_DIR}/include" "${CUDA_HOME}/include"
    PRIVATE "${NCCL_SOURCE_DIR}/src/include"
)
target_link_libraries(nccl_static PUBLIC cudart_static ${CMAKE_DL_LIBS} rt)
target_link_libraries(nccl_static PRIVATE ${NCCL_DEVICE_OBJS})
target_link_libraries(nccl_static PRIVATE nccl_device_interface)

set_target_properties(nccl_static ${NCCL_DEVICE_OBJS} PROPERTIES
        CXX_VISIBILITY_PRESET hidden)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
    set_target_properties(nccl_static ${NCCL_DEVICE_OBJS} PROPERTIES
            CUDA_ARCHITECTURES "${MEGRAY_CUDA_ARCHITECTURES}")
endif()
set_target_properties(nccl_static PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# parse version info
file(READ "${NCCL_SOURCE_DIR}/makefiles/version.mk" NCCL_VERSION_FILE)

string(REGEX REPLACE "\n" ";" NCCL_VERSION_FILE ${NCCL_VERSION_FILE})

foreach(PAIR IN ITEMS ${NCCL_VERSION_FILE})
    if(NOT ${PAIR} MATCHES ":=")
        continue()
    endif()
    string(REGEX REPLACE ":=" ";" PAIR "${PAIR}")
    list(GET PAIR 0 KEY)
    list(GET PAIR 1 VALUE)
    string(STRIP "${KEY}" KEY)
    string(STRIP "${VALUE}" VALUE)
    if(NOT ${KEY} MATCHES "NCCL_")
        continue()
    endif()
    set(${KEY} "${VALUE}")
endforeach()

math(EXPR NCCL_VERSION "${NCCL_MAJOR}*1000 + ${NCCL_MINOR}*100 + ${NCCL_PATCH}" OUTPUT_FORMAT DECIMAL) 

# read header input
file(READ "${NCCL_SOURCE_DIR}/src/nccl.h.in" NCCL_HEADER)

# replace version info
string(REGEX REPLACE "\\\$\{nccl\:Major\}" "${NCCL_MAJOR}" NCCL_HEADER "${NCCL_HEADER}")
string(REGEX REPLACE "\\\$\{nccl\:Minor\}" "${NCCL_MINOR}" NCCL_HEADER "${NCCL_HEADER}")
string(REGEX REPLACE "\\\$\{nccl\:Patch\}" "${NCCL_PATCH}" NCCL_HEADER "${NCCL_HEADER}")
string(REGEX REPLACE "\\\$\{nccl\:Suffix\}" "${NCCL_SUFFIX}" NCCL_HEADER "${NCCL_HEADER}")
string(REGEX REPLACE "\\\$\{nccl\:Version\}" "${NCCL_VERSION}" NCCL_HEADER "${NCCL_HEADER}")

# write header back
file(WRITE "${NCCL_BINARY_DIR}/include/nccl.h.in" "${NCCL_HEADER}")
configure_file("${NCCL_BINARY_DIR}/include/nccl.h.in" "${NCCL_BINARY_DIR}/include/nccl.h" COPYONLY)
