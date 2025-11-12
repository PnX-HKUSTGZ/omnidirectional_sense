cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Warnings: apply only to C++ sources; NVCC does not accept bare -Werror
# We'll set C++ warnings later via target_compile_options

## 优化：按构建类型分别配置（避免全局 -O2 覆盖/冲突）
# C++ 优化在 Release/RelWithDebInfo 打开，Debug 保持易调试
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3")

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add third_party_install to CMAKE_PREFIX_PATH
get_filename_component(ABS_PREFIX_PREFIX "../../third_party_install" ABSOLUTE)
set(CMAKE_PREFIX_PREFIX ${ABS_PREFIX_PREFIX})
message(STATUS "ABS_PREFIX_PREFIX: ${ABS_PREFIX_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH ${ABS_PREFIX_PREFIX})

# 设置 OpenGL 策略以消除警告
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

#######################
## Find dependencies ##
#######################
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
set(OpenCV_DIR "/usr/local/include/opencv4")
find_package(OpenCV REQUIRED)

# 查找 CUDA (必需)
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # CUDA 优化选项（按构建类型）：
    # -O3：更激进优化；--use_fast_math：快速数学；-lineinfo：便于 Nsight 对应源码；
    # -Xptxas -v：显示寄存器/共享内存用量；-Xptxas -dlcm=ca：全局加载缓存策略（可A/B）。
    set(CUDA_RELEASE_FLAGS "-O3 --use_fast_math -lineinfo -Xptxas=-v,-dlcm=ca")
    set(CUDA_RELWITHDEBINFO_FLAGS "-O3 -lineinfo -Xptxas=-v")
    # Debug 不启用 -G（会关闭优化），保留默认以便调试
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} ${CUDA_RELEASE_FLAGS}")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} ${CUDA_RELWITHDEBINFO_FLAGS}")
    add_definitions(-DWITH_CUDA)
else()
    message(FATAL_ERROR "CUDA not found, CUDA is required for this project")
endif()

# 查找 TensorRT (必需)
# TensorRT 通常安装在 /usr/local/TensorRT-* 或 /usr/lib/x86_64-linux-gnu
set(TensorRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT installation directory")
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES include)
find_library(TensorRT_LIBRARY nvinfer
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)
find_library(TensorRT_ONNX_LIBRARY nvonnxparser
    HINTS ${TensorRT_ROOT} $ENV{TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    message(STATUS "TensorRT found!")
    message(STATUS "TensorRT include: ${TensorRT_INCLUDE_DIR}")
    message(STATUS "TensorRT library: ${TensorRT_LIBRARY}")
else()
    message(FATAL_ERROR "TensorRT not found! Please set TensorRT_ROOT or install TensorRT")
endif()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

# 显式将 CUDA 源文件添加到目标（确保由 NVCC 编译）
target_sources(${PROJECT_NAME}
  PRIVATE
    src/ai_kernels.cu
)

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME} PUBLIC 
  ${CUDA_LIBRARY_DIRS}
)

# 链接库
set(LINK_LIBS 
  ${OpenCV_LIBS} 
  ${CUDA_LIBRARIES} 
  ${TensorRT_LIBRARY}
  ${TensorRT_ONNX_LIBRARY}
  cudnn
)

# 设置 CUDA 架构 (可根据需要调整)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75 80 86)

# C++ warnings as errors; for CUDA, forward warnings to host compiler only
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-O3 --use_fast_math -lineinfo -Xptxas=-v>
)

target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()


endif()

#############
## Install ##
#############

ament_auto_package(
  USE_SCOPED_HEADER_INSTALL_DIR
  INSTALL_TO_SHARE
  model
)
