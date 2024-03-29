set(NCCL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl)
set(NCCL_BINARY_DIR ${PROJECT_BINARY_DIR}/third_party/nccl)

enable_language(CUDA)

if (MEGRAY_BUILD_NCCL_WITH_EXTERNAL_SCRIPT)
    set(NCCL_BUILD_DIR ${NCCL_BINARY_DIR}/build)
    set(NCCL_INCLUDE_DIRS ${NCCL_BUILD_DIR}/include)
    set(NCCL_LIBRARIES ${NCCL_BUILD_DIR}/lib/libnccl_static.a)

    include(ProcessorCount)
    ProcessorCount(NCCL_BUILD_THREAD)
    if(NCCL_BUILD_THREAD EQUAL 0)
        set(NCCL_BUILD_THREAD 16)
    endif()
    message(STATUS "NCCL_BUILD_THREAD: ${NCCL_BUILD_THREAD}")

    if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
        set(MAKE_COMMAND make)
    else()
        set(MAKE_COMMAND "$(MAKE)")
    endif()

    if ("${MEGRAY_CUDA_GENCODE}" STREQUAL "")
        message(WARNING "MEGRAY_CUDA_GENCODE unset")
    endif()

    get_filename_component(NCCL_BUILD_DIR ${NCCL_BUILD_DIR} ABSOLUTE)
    file(MAKE_DIRECTORY ${NCCL_BUILD_DIR}/include)

    if("${MEGRAY_CUDA_GENCODE}" STREQUAL "" ) # hack for nccl 2-11-4 default nvcc will error
        add_custom_command(
            OUTPUT ${NCCL_LIBRARIES}
            COMMAND ${MAKE_COMMAND} -j${NCCL_BUILD_THREAD} -C src ${NCCL_BUILD_DIR}/include/nccl.h BUILDDIR=${NCCL_BUILD_DIR}
            COMMAND ${MAKE_COMMAND} -j${NCCL_BUILD_THREAD} src.build  BUILDDIR=${NCCL_BUILD_DIR} CUDA_HOME=${CUDA_HOME}
            WORKING_DIRECTORY ${NCCL_SOURCE_DIR}
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT ${NCCL_LIBRARIES}
            COMMAND ${MAKE_COMMAND} -j${NCCL_BUILD_THREAD} -C src ${NCCL_BUILD_DIR}/include/nccl.h BUILDDIR=${NCCL_BUILD_DIR}
            COMMAND ${MAKE_COMMAND} -j${NCCL_BUILD_THREAD} src.build NVCC_GENCODE=${MEGRAY_CUDA_GENCODE} BUILDDIR=${NCCL_BUILD_DIR} CUDA_HOME=${CUDA_HOME}
            WORKING_DIRECTORY ${NCCL_SOURCE_DIR}
            VERBATIM
        )
    endif()

    add_custom_target(nccl_build DEPENDS ${NCCL_LIBRARIES})
    add_library(nccl_static STATIC IMPORTED GLOBAL)
    add_dependencies(nccl_static nccl_build)
    set_target_properties(
        nccl_static PROPERTIES
        IMPORTED_LOCATION ${NCCL_LIBRARIES}
    )
    target_include_directories(nccl_static INTERFACE "${NCCL_INCLUDE_DIRS}" "${CUDA_HOME}/include")
    target_link_directories(nccl_static INTERFACE "${CUDA_HOME}/lib64")
    target_link_libraries(nccl_static INTERFACE cudart_static ${CMAKE_DL_LIBS} rt)
else()
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
    file(GLOB_RECURSE NCCL_HOST_SRCS "${NCCL_SOURCE_DIR}/src/*.cc")
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
endif()
