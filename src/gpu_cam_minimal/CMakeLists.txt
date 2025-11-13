cmake_minimum_required(VERSION 3.10)
project(gpu_cam_minimal)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## Add optimization flags
add_compile_options(-O2)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

# CMake policy: make CheckInclude* honor CMAKE_REQUIRED_LIBRARIES (silence dev warning)
if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

find_package(ament_cmake_auto REQUIRED)

# Core OpenCV requirements
find_package(OpenCV REQUIRED COMPONENTS core videoio)

# Optional: detect OpenCV CUDA runtime libs we actually need when GPU path is enabled
find_library(OPENCV_CUDAWARPING_LIB opencv_cudawarping)
find_library(OPENCV_CUDAIMGPROC_LIB opencv_cudaimgproc)

# Jetson/NVDEC optional libs (link only when found)
find_library(LIB_V4L2 v4l2)
find_library(LIB_EGL EGL)
find_library(LIB_CUDA cuda)
find_library(LIB_NVBUF_SURFACE nvbufsurface
  PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu)

# Optional include paths to improve __has_include detection in source
find_path(CUDA_INCLUDE_DIR cuda.h
  PATHS /usr/local/cuda/include /usr/include
)
find_path(NVMMAPI_INCLUDE_DIR NvVideoDecoder.h
  PATHS /usr/src/jetson_multimedia_api/include /usr/include
)

ament_auto_find_build_dependencies()

include(CheckIncludeFileCXX)
check_include_file_cxx("opencv2/core/cuda.hpp" HAVE_OPENCV_CUDA)

add_executable(gpu_cam_minimal_node src/gpu_cam_minimal_node.cpp src/nvdec_mjpeg_decoder.cpp)

# Link OpenCV (core/videoio always)
target_link_libraries(gpu_cam_minimal_node ${OpenCV_LIBRARIES})

# Add optional CUDA OpenCV libs if present (only then define GPU macro)
if(OPENCV_CUDAWARPING_LIB AND OPENCV_CUDAIMGPROC_LIB)
  target_link_libraries(gpu_cam_minimal_node ${OPENCV_CUDAWARPING_LIB} ${OPENCV_CUDAIMGPROC_LIB})
  target_compile_definitions(gpu_cam_minimal_node PRIVATE GPU_CAM_HAS_OPENCV_CUDA=1)
  message(STATUS "gpu_cam_minimal: OpenCV CUDA detected (linking cudaimgproc,cudawarping)")
else()
  message(WARN "gpu_cam_minimal: OpenCV CUDA not fully available; building without CUDA color ops")
endif()

# Link Jetson/NVDEC
target_link_libraries(gpu_cam_minimal_node ${LIB_V4L2} ${LIB_EGL} ${LIB_CUDA} ${LIB_NVBUF_SURFACE})

ament_target_dependencies(gpu_cam_minimal_node armor_detector rclcpp sensor_msgs camera_info_manager)

# Link opencv
target_include_directories(gpu_cam_minimal_node PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(gpu_cam_minimal_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(CUDA_INCLUDE_DIR)
  target_include_directories(gpu_cam_minimal_node PRIVATE ${CUDA_INCLUDE_DIR})
endif()
if(NVMMAPI_INCLUDE_DIR)
  target_include_directories(gpu_cam_minimal_node PRIVATE ${NVMMAPI_INCLUDE_DIR})
endif()

# Install the node
install(TARGETS gpu_cam_minimal_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install shared resources if present
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs OR EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY docs config
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
  )
endif()

ament_auto_package()