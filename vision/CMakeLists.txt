cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(BYTED_VISION LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14  -s -O3 -fPIC  -g -Wall -Wextra -Wno-unused-local-typedefs -Wno-unused-function -Wno-unused-parameter -ffunction-sections -fdata-sections -Wl,--gc-sections")

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0) 

#include opencv
find_package(OpenCV 3.4.8 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_INSTALL_PATH}/lib)
message("OpenCV_INSTALL_PATH: " ${OpenCV_INSTALL_PATH})
set(CV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_flann)

#include matx header
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)

#include byted vision cpu
include_directories(${CMAKE_CURRENT_LIST_DIR})


#add byted_vision cpu src
FILE(GLOB byted_vision_cc_file ${CMAKE_CURRENT_LIST_DIR}/ops/base/*.cc ${CMAKE_CURRENT_LIST_DIR}/ops/cpu/*.cc ${CMAKE_CURRENT_LIST_DIR}/utils/*.cc)


add_library(byted_vision_cpu_ops SHARED
    ${cv_cuda_c_file}
    ${byted_vision_cc_file}
)

target_link_libraries(byted_vision_cpu_ops PUBLIC ${CV_LIBS})
