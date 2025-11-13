cmake_minimum_required(VERSION 3.10)
project(gpu_cam_minimal)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core videoio)

# Optional: detect OpenCV CUDA runtime libs
find_library(OPENCV_CUDAWARPING_LIB opencv_cudawarping)
find_library(OPENCV_CUDAIMGPROC_LIB opencv_cudaimgproc)

# Jetson/NVDEC libraries (from Rules.mk)
set(NVLIBS_DIR /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra /usr/lib/aarch64-linux-gnu)
find_library(LIB_V4L2 v4l2)
find_library(LIB_EGL EGL)
find_library(LIB_GLESv2 GLESv2)
find_library(LIB_X11 X11)
find_library(LIB_NVBUF_SURFACE nvbufsurface PATHS ${NVLIBS_DIR})
find_library(LIB_NVV4L2 nvv4l2 PATHS ${NVLIBS_DIR})
find_library(LIB_NVBUF_SURFTRANSFORM nvbufsurftransform PATHS ${NVLIBS_DIR})
find_library(LIB_NVJPEG nvjpeg PATHS ${NVLIBS_DIR})
find_library(LIB_NVOSD nvosd PATHS ${NVLIBS_DIR})
find_library(LIB_DRM drm PATHS ${NVLIBS_DIR})
find_library(LIB_CUDA cuda)
if (NOT LIB_NVBUF_SURFACE OR NOT LIB_NVV4L2 OR NOT LIB_NVJPEG OR NOT LIB_NVOSD)
  message(FATAL_ERROR "Could not find required Jetson NVDEC / Multimedia libraries.")
endif()
if (NOT LIB_V4L2 OR NOT LIB_EGL OR NOT LIB_GLESv2 OR NOT LIB_X11 OR NOT LIB_DRM)
  message(FATAL_ERROR "Could not find required system libraries.")
endif()
if (NOT LIB_CUDA)
  message(FATAL_ERROR "Could not find CUDA libraries.")
endif()
if (NOT LIB_NVBUF_SURFTRANSFORM)
  message(WARNING "Could not find nvbufsurftransform library; some features may be disabled.")
endif()

# Ensure libdrm is found
find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM REQUIRED libdrm)
include_directories(${DRM_INCLUDE_DIRS})
link_libraries(${DRM_LIBRARIES})

# include paths
find_path(CUDA_INCLUDE_DIR cuda.h
  PATHS /usr/local/cuda/include
)
# 添加 Jetson Multimedia API 源文件
file(GLOB NV_CLASSES_SRC
    /usr/src/jetson_multimedia_api/samples/common/classes/*.cpp
)
file(GLOB NV_CUDA_SRC
    /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda/*.cpp
    /usr/src/jetson_multimedia_api/samples/common/algorithm/cuda/*.cu
)

ament_auto_find_build_dependencies()

include(CheckIncludeFileCXX)
check_include_file_cxx("opencv2/core/cuda.hpp" HAVE_OPENCV_CUDA)


# Executable
set(PROJECT_SOURCES
  src/${PROJECT_NAME}_node.cpp
  src/nvdec_mjpeg_decoder.cpp
)
add_executable(${PROJECT_NAME}_node ${PROJECT_SOURCES} ${NV_CLASSES_SRC} ${NV_CUDA_SRC})

# Link OpenCV
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBRARIES})

# Link OpenCV CUDA if present
if(OPENCV_CUDAWARPING_LIB AND OPENCV_CUDAIMGPROC_LIB)
  target_link_libraries(${PROJECT_NAME}_node ${OPENCV_CUDAWARPING_LIB} ${OPENCV_CUDAIMGPROC_LIB})
  target_compile_definitions(${PROJECT_NAME}_node PRIVATE GPU_CAM_HAS_OPENCV_CUDA=1)
  message(STATUS "gpu_cam_minimal: OpenCV CUDA detected (linking cudaimgproc,cudawarping)")
else()
  message(WARNING "gpu_cam_minimal: OpenCV CUDA not fully available; building without CUDA color ops")
endif()

# Link Jetson NVDEC / Multimedia libraries (Rules.mk equivalent)
target_link_libraries(${PROJECT_NAME}_node
    ${LIB_V4L2}
    ${LIB_EGL}
    ${LIB_GLESv2}
    ${LIB_X11}
    ${LIB_NVBUF_SURFACE}
    ${LIB_NVBUF_SURFTRANSFORM}
    ${LIB_NVV4L2}
    ${LIB_NVJPEG}
    ${LIB_NVOSD}
    ${LIB_DRM}
    ${LIB_CUDA}
    pthread
    cudart
)

# Include directories
target_include_directories(${PROJECT_NAME}_node
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
if(CUDA_INCLUDE_DIR)
  target_include_directories(${PROJECT_NAME}_node PRIVATE ${CUDA_INCLUDE_DIR})
endif()
# Include Jetson Multimedia API headers
target_include_directories(${PROJECT_NAME}_node BEFORE PRIVATE
  /usr/src/jetson_multimedia_api/include
  /usr/src/jetson_multimedia_api/include/libjpeg-8b
  /usr/src/jetson_multimedia_api/samples/common/classes
)
target_include_directories(${PROJECT_NAME}_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ROS2 dependencies
ament_target_dependencies(${PROJECT_NAME}_node armor_detector rclcpp sensor_msgs camera_info_manager)

# Install the node
install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install shared resources if present
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/docs OR EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY docs config
    DESTINATION share/${PROJECT_NAME}
    OPTIONAL
  )
endif()

target_compile_options(${PROJECT_NAME}_node PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>   # 屏蔽 CUDA 文件编译器警告
    $<$<COMPILE_LANGUAGE:CXX>:-isystem /usr/src/jetson_multimedia_api/include -w>
)

ament_auto_package()
