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

# Detect Jetson Multimedia API (headers only check)
set(JETSON_MMAPI_INCLUDE_DIR "/usr/src/jetson_multimedia_api/include")
set(JETSON_MMAPI_INCLUDE_DIR_ALT "/usr/src/tegra_multimedia_api/include")

if(EXISTS ${JETSON_MMAPI_INCLUDE_DIR}/NvVideoDecoder.h OR EXISTS ${JETSON_MMAPI_INCLUDE_DIR_ALT}/NvVideoDecoder.h)
  message(STATUS "gpu_cam_minimal: Jetson Multimedia API detected")
  add_definitions(-DGPU_CAM_HAS_JETSON_MMA)

  # Library search paths vary by JetPack/L4T version
  list(APPEND CMAKE_LIBRARY_PATH
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu/tegra
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
  )

  # Core MMAPI libs used by jetson_mjpeg_decoder.cpp
  find_library(NVBUFSURFACE_LIB nvbufsurface)
  find_library(NVBUFUTILS_LIB nvbuf_utils)
  find_library(NVBUFSURFTRANS_LIB nvbufsurftransform)
  find_library(EGL_LIB EGL)
  find_library(CUDAEGL_LIB cudaEGL)
  find_library(CUDART_LIB cudart)
  find_library(V4L2_LIB v4l2)
  find_library(PTHREAD_LIB pthread)

  # Link if found; some symbols are in nvbufsurface, nvbuf_utils
  target_link_libraries(gpu_cam_minimal_node PRIVATE
    ${NVBUFSURFACE_LIB}
    ${NVBUFUTILS_LIB}
    ${NVBUFSURFTRANS_LIB}
    ${EGL_LIB}
    ${CUDAEGL_LIB}
    ${CUDART_LIB}
    ${V4L2_LIB}
    ${PTHREAD_LIB}
  )
endif()

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

add_executable(gpu_cam_minimal_node
  src/gpu_cam_minimal_node.cpp
  src/jetson_mjpeg_decoder.cpp
)

if(HAVE_OPENCV_CUDA)
  target_compile_definitions(gpu_cam_minimal_node PRIVATE GPU_CAM_HAS_OPENCV_CUDA=1)
endif()

ament_target_dependencies(gpu_cam_minimal_node rclcpp sensor_msgs camera_info_manager image_transport cv_bridge)

target_include_directories(gpu_cam_minimal_node PRIVATE ${OpenCV_INCLUDE_DIRS} include ${JETSON_MMAPI_INCLUDE_DIR} ${JETSON_MMAPI_INCLUDE_DIR_ALT})

# Try to include armor_detector headers if available (for GpuImage type)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../armor_detector/include/armor_detector/gpu_image.hpp)
  message(STATUS "gpu_cam_minimal: found armor_detector includes")
  target_include_directories(gpu_cam_minimal_node PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../armor_detector/include)
endif()

target_link_libraries(gpu_cam_minimal_node PRIVATE ${OpenCV_LIBRARIES})

install(TARGETS gpu_cam_minimal_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()
