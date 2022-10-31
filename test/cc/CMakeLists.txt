cmake_minimum_required(VERSION 3.2)

# include path
INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(3rdparty/rapidjson/include)
INCLUDE_DIRECTORIES(3rdparty/filesystem/include)

# link path
LINK_DIRECTORIES(/usr/local/lib)

# test cc file
FILE(GLOB_RECURSE UNIT_TEST_CC RELATIVE ${CMAKE_SOURCE_DIR}/test/cc "*test*.cc")


# target
ADD_EXECUTABLE(cc_unit_test ${UNIT_TEST_CC})

TARGET_LINK_LIBRARIES(cc_unit_test matx)
TARGET_LINK_LIBRARIES(cc_unit_test pthread)
TARGET_LINK_LIBRARIES(cc_unit_test gtest)
TARGET_LINK_LIBRARIES(cc_unit_test gtest_main)
