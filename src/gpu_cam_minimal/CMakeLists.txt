cmake_minimum_required(VERSION 3.10)
project(gpu_cam_minimal)

# Options
option(GPU_CAM_USE_CUDA "Enable GPU upload via OpenCV CUDA" ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core videoio)

# Try to detect CUDA modules of OpenCV (optional)
set(HAVE_OPENCV_CUDA OFF)
if(GPU_CAM_USE_CUDA)
  # Some OpenCV builds do not provide separate imported targets per CUDA module.
  # We'll enable CUDA path at compile time if headers are available.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${OpenCV_LIBRARIES})
  check_cxx_source_compiles(
    "#include <opencv2/core/cuda.hpp>\nint main(){cv::cuda::GpuMat g;return 0;}" OPENCV_HAS_CUDA_NAMESPACE)
  if(OPENCV_HAS_CUDA_NAMESPACE)
    set(HAVE_OPENCV_CUDA ON)
  endif()
endif()

add_executable(gpu_cam_minimal_node src/gpu_cam_minimal_node.cpp)

if(HAVE_OPENCV_CUDA)
  target_compile_definitions(gpu_cam_minimal_node PRIVATE GPU_CAM_HAS_OPENCV_CUDA=1)
endif()

ament_target_dependencies(gpu_cam_minimal_node rclcpp sensor_msgs camera_info_manager image_transport cv_bridge)

target_include_directories(gpu_cam_minimal_node PRIVATE ${OpenCV_INCLUDE_DIRS})

# Try to include video_reader headers if available (optional)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../video_reader/include/video_reader/gpu_image.hpp)
  message(STATUS "gpu_cam_minimal: found local video_reader includes")
  target_include_directories(gpu_cam_minimal_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../video_reader/include)
endif()

target_link_libraries(gpu_cam_minimal_node ${OpenCV_LIBRARIES})

install(TARGETS gpu_cam_minimal_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()
