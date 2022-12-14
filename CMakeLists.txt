cmake_minimum_required(VERSION 3.2)
project(matxscript)

SET(CMAKE_LOCAL "${PROJECT_SOURCE_DIR}/cmake")
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_LOCAL}/)
SET(CMAKE_INSTALL_RPATH ".")

IF (DEFINED ENV{CUDA_HOME})
    MESSAGE(STATUS "USE CUDA CUDA_HOME: $ENV{CUDA_HOME}")
    SET(USE_CUDA $ENV{CUDA_HOME})
ENDIF ()

IF (USE_CUDA)
    INCLUDE(cmake/FindCUDA.cmake)
    find_cuda(${USE_CUDA})
    INCLUDE_DIRECTORIES(SYSTEM ${CUDA_INCLUDE_DIRS})
    SET(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-Wall -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
    MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
ENDIF ()

IF (ENABLE_RUNTIME_STRING_INVARIANT_CHECK)
    ADD_DEFINITIONS(-DMATXSCRIPT_RUNTIME_STRING_UNICODE_ENABLE_INVARIANT_CHECK)
ENDIF ()

IF (BUILD_MICRO_RUNTIME)
    ADD_DEFINITIONS(-DMATX_BUILD_MICRO_RUNTIME)
    ADD_DEFINITIONS(-DMATX_SUPPORT_ANDROID)
    ADD_DEFINITIONS(-DDISABLE_UNICODEDATA)
ELSE ()
    ADD_DEFINITIONS(-DMATX_RUNTIME_ENABLE_STRINGREF)
    # third-part
    FIND_PACKAGE(PCRE REQUIRED)
    INCLUDE_DIRECTORIES(${PCRE_INCLUDE_DIR})
    # jemalloc
    FIND_PACKAGE(Jemalloc)
    IF (JEMALLOC_INCLUDE_DIR)
        INCLUDE_DIRECTORIES(${JEMALLOC_INCLUDE_DIR})
        ADD_DEFINITIONS(-DUSE_JEMALLOC)
    ENDIF ()
ENDIF ()

IF (PCRE_FOUND)
    ADD_DEFINITIONS(-DMATX_ENABLE_PCRE_REGEX)
ENDIF ()

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(3rdparty/rapidjson/include)
INCLUDE_DIRECTORIES(3rdparty/filesystem/include)
INCLUDE_DIRECTORIES(3rdparty/alloc/include/brt)
INCLUDE_DIRECTORIES(3rdparty/cppjieba/include)
INCLUDE_DIRECTORIES(3rdparty/cppjieba/deps/limonp/include)

IF (DEFINED ENV{CFLAGS})
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} $ENV{CFLAGS}")
ENDIF ()
IF (DEFINED ENV{CXXFLAGS})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CXXFLAGS}")
ENDIF ()
IF (DEFINED ENV{CPPFLAGS})
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{CPPFLAGS}")
ENDIF ()
IF (NOT CMAKE_BUILD_TYPE)
    MESSAGE(STATUS "No build type selected, default to ReleaseWithDebugInfo")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -DNDEBUG")
ENDIF ()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -ftls-model=global-dynamic")
MESSAGE(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# cuda src
FILE(GLOB_RECURSE MATXSCRIPT_CUDA_SRCS RELATIVE ${CMAKE_SOURCE_DIR} "src/runtime/cuda/*.cc" "src/runtime/cuda/*.cpp" "src/runtime/cuda/*.h")
# allocator
FILE(GLOB_RECURSE MEM_ALLOC RELATIVE ${CMAKE_SOURCE_DIR} "3rdparty/alloc/*.cc")
FILE(GLOB_RECURSE MEM_CUDA_ALLOC RELATIVE ${CMAKE_SOURCE_DIR} "3rdparty/alloc/brt/core/device/cuda/*.cc")
# regex src
FILE(GLOB_RECURSE MATXSCRIPT_REGEX_HS RELATIVE ${CMAKE_SOURCE_DIR} "include/matxscript/runtime/regex/*.h")
FILE(GLOB_RECURSE MATXSCRIPT_REGEX_SRCS RELATIVE ${CMAKE_SOURCE_DIR} "src/runtime/regex/*.cc" "src/runtime/regex/*.cpp")
# runtime src
FILE(GLOB_RECURSE MATXSCRIPT_RUNTIME_HS RELATIVE ${CMAKE_SOURCE_DIR} "include/matxscript/runtime/*.h")
FILE(GLOB_RECURSE MATXSCRIPT_RUNTIME_SRCS RELATIVE ${CMAKE_SOURCE_DIR} "src/runtime/*.cc" "src/runtime/*.cpp" "src/runtime/*.h")
# pipeline src
FILE(GLOB_RECURSE MATXSCRIPT_PIPELINE_HS RELATIVE ${CMAKE_SOURCE_DIR} "include/matxscript/pipeline/*.h")
FILE(GLOB_RECURSE MATXSCRIPT_PIPELINE_SRCS RELATIVE ${CMAKE_SOURCE_DIR} "src/pipeline/*.cc" "src/pipeline/*.cpp" "src/pipeline/*.h")
# unicodedata src
FILE(GLOB_RECURSE MATXSCRIPT_UNICODEDATA_HS RELATIVE ${CMAKE_SOURCE_DIR} "include/matxscript/runtime/unicodelib/py_unicodedata.h")
FILE(GLOB_RECURSE MATXSCRIPT_UNICODEDATA_SRCS RELATIVE ${CMAKE_SOURCE_DIR} "src/runtime/unicodelib/py_unicodedata.cc")
# other src
FILE(GLOB_RECURSE MATXSCRIPT_OTHER_HS RELATIVE ${CMAKE_SOURCE_DIR} "include/matxscript/ir/*.h" "include/matxscript/server/*.h")
FILE(GLOB_RECURSE
        MATXSCRIPT_OTHER_SRCS
        RELATIVE
        ${CMAKE_SOURCE_DIR}
        "src/codegen/*.cc" "src/codegen/*.cpp" "src/codegen/*.h"
        "src/ir/*.cc" "src/ir/*.cpp" "src/ir/*.h"
        "src/printer/*.cc" "src/printer/*.cpp" "src/printer/*.h"
        "src/server/*.cc" "src/server/*.cpp" "src/server/*.h"
        "src/c_api/*.cc" "src/c_api/*.cpp" "src/c_api/*.h")

IF (BUILD_MICRO_RUNTIME)
    # generic src
    SET(MATXSCRIPT_HS ${MATXSCRIPT_RUNTIME_HS} ${MATXSCRIPT_PIPELINE_HS})
    SET(MATXSCRIPT_SRCS ${MATXSCRIPT_RUNTIME_SRCS} ${MATXSCRIPT_PIPELINE_SRCS})
    LIST(REMOVE_ITEM MATXSCRIPT_HS ${MATXSCRIPT_UNICODEDATA_HS})
    LIST(REMOVE_ITEM MATXSCRIPT_SRCS ${MATXSCRIPT_UNICODEDATA_SRCS})
ELSE ()
    # generic src
    SET(MATXSCRIPT_HS ${MATXSCRIPT_RUNTIME_HS} ${MATXSCRIPT_PIPELINE_HS} ${MATXSCRIPT_OTHER_HS})
    SET(MATXSCRIPT_SRCS ${MATXSCRIPT_RUNTIME_SRCS} ${MATXSCRIPT_PIPELINE_SRCS} ${MATXSCRIPT_OTHER_SRCS})
ENDIF ()

IF (NOT PCRE_FOUND)
    LIST(REMOVE_ITEM MATXSCRIPT_HS ${MATXSCRIPT_REGEX_HS})
    LIST(REMOVE_ITEM MATXSCRIPT_SRCS ${MATXSCRIPT_REGEX_SRCS})
ENDIF ()

LIST(REMOVE_ITEM MATXSCRIPT_SRCS ${MATXSCRIPT_CUDA_SRCS})
LIST(REMOVE_ITEM MEM_ALLOC ${MEM_CUDA_ALLOC})

ADD_LIBRARY(matx SHARED ${MATXSCRIPT_HS} ${MATXSCRIPT_SRCS} ${MEM_ALLOC})

IF (BUILD_MICRO_RUNTIME)
    SET_TARGET_PROPERTIES(matx PROPERTIES LINK_FLAGS_RELEASE -s)
ELSE ()
    TARGET_LINK_LIBRARIES(matx pthread)
ENDIF ()
TARGET_LINK_LIBRARIES(matx dl)
IF (PCRE_FOUND)
    TARGET_LINK_LIBRARIES(matx ${PCRE_LIBRARIES})
ENDIF ()

IF (USE_CUDA)
    ADD_LIBRARY(matx_cuda SHARED ${MATXSCRIPT_CUDA_SRCS} ${MEM_CUDA_ALLOC})
    TARGET_LINK_LIBRARIES(matx_cuda matx)
    # TARGET_LINK_LIBRARIES(matx_cuda ${CUDA_NVRTC_LIBRARY})
    TARGET_LINK_LIBRARIES(matx_cuda ${CUDA_CUDA_LIBRARY})
    TARGET_LINK_LIBRARIES(matx_cuda ${CUDA_CUDART_LIBRARY})
ENDIF ()

IF (JEMALLOC_LIBRARIES)
    TARGET_LINK_LIBRARIES(matx ${JEMALLOC_LIBRARIES})
ENDIF ()


SET(USE_LIBBACKTRACE OFF)
IF (DEFINED ENV{USE_LIBBACKTRACE})
    MESSAGE(STATUS "USE LIB_BACKTRACE: $ENV{USE_LIBBACKTRACE}")
    IF (CMAKE_SYSTEM_NAME MATCHES "Darwin" OR CMAKE_SYSTEM_NAME MATCHES "Linux")
        SET(USE_LIBBACKTRACE $ENV{USE_LIBBACKTRACE})
    ENDIF ()
ENDIF ()

IF (USE_LIBBACKTRACE)
    MESSAGE("Building with libbacktrace...")
    include(cmake/Libbacktrace.cmake)
    TARGET_LINK_LIBRARIES(matx libbacktrace)
    target_include_directories(matx PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/libbacktrace/include)
    target_compile_definitions(matx PUBLIC MATX_WITH_LIBBACKTRACE)
ENDIF ()

ENABLE_TESTING()
SET(BUILD_TESTING ON)
IF (DEFINED ENV{BUILD_TESTING})
    MESSAGE(STATUS "USE ENV BUILD_TESTING: $ENV{BUILD_TESTING}")
    SET(BUILD_TESTING $ENV{BUILD_TESTING})
ENDIF ()

IF (BUILD_TESTING)
    MESSAGE(STATUS "BUILD_TESTING: ON")
    ADD_SUBDIRECTORY(test/cc/)
ENDIF ()

SET(BUILD_BENCHMARK ON)
IF (DEFINED ENV{BUILD_BENCHMARK})
    MESSAGE(STATUS "USE ENV BUILD_BENCHMARK: $ENV{BUILD_BENCHMARK}")
    SET(BUILD_BENCHMARK $ENV{BUILD_BENCHMARK})
ENDIF ()

IF (BUILD_BENCHMARK)
    MESSAGE(STATUS "BUILD_BENCHMARK: ON")
    ADD_SUBDIRECTORY(tools)
ENDIF ()

# extensions
FILE(GLOB_RECURSE MATXSCRIPT_TEXT_SRCS RELATIVE ${CMAKE_SOURCE_DIR} "text/*.h" "text/*.cc" "text/*.cpp")
ADD_LIBRARY(matx_text_ops SHARED ${MATXSCRIPT_TEXT_SRCS})
TARGET_LINK_LIBRARIES(matx_text_ops matx)

IF (BUILD_MICRO_RUNTIME)
    INSTALL(DIRECTORY include DESTINATION matx/)
    INSTALL(TARGETS matx matx_text_ops
            RUNTIME DESTINATION matx/bin
            LIBRARY DESTINATION matx/lib
            ARCHIVE DESTINATION matx/lib)
ELSE ()
    INSTALL(DIRECTORY python/matx DESTINATION .)
    INSTALL(FILES python/requirements.txt DESTINATION .)
    INSTALL(FILES python/setup.py DESTINATION .)
    INSTALL(DIRECTORY include DESTINATION matx/)
    INSTALL(DIRECTORY 3rdparty/rapidjson/include DESTINATION matx/)
    INSTALL(DIRECTORY ${PCRE_INCLUDE_DIR} DESTINATION matx/)

    IF (USE_CUDA)
        SET(targets matx matx_text_ops matx_cuda)
    ELSE ()
        SET(targets matx matx_text_ops)
    ENDIF ()

    INSTALL(TARGETS ${targets}
            RUNTIME DESTINATION matx/bin
            LIBRARY DESTINATION matx/lib
            ARCHIVE DESTINATION matx/lib)
    SET(PCRE_TEMP_LIB ${PCRE_LIBRARIES})
    IF(NOT APPLE)
        SET(PCRE_INSTALL_LIBS ${PCRE_TEMP_LIB} "${PCRE_TEMP_LIB}.1")
    ENDIF()
    WHILE (IS_SYMLINK ${PCRE_TEMP_LIB})
        GET_FILENAME_COMPONENT(PCRE_TEMP_LIB_F ${PCRE_TEMP_LIB} REALPATH)
        SET(PCRE_INSTALL_LIBS ${PCRE_INSTALL_LIBS} ${PCRE_TEMP_LIB_F})
        SET(PCRE_TEMP_LIB ${PCRE_TEMP_LIB_F})
    ENDWHILE ()
    MESSAGE(STATUS "INSTALL PCRE LIBS: ${PCRE_INSTALL_LIBS}")
    INSTALL(FILES ${PCRE_INSTALL_LIBS} DESTINATION matx/lib)
ENDIF ()
