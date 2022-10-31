cmake_minimum_required(VERSION 3.2)

# include path
INCLUDE_DIRECTORIES(/usr/local/include)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(3rdparty/rapidjson/include)

# link path
LINK_DIRECTORIES(/usr/local/lib)

# target
ADD_EXECUTABLE(echo_object_size echo_object_size.cc)
ADD_EXECUTABLE(test_memory_leak test_memory_leak.cc)

TARGET_LINK_LIBRARIES(echo_object_size matx)
TARGET_LINK_LIBRARIES(test_memory_leak matx)

# add test
ADD_TEST(echo_object_size echo_object_size)
