cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
add_definitions(-Wall -Werror)

## Add optimization flags
add_compile_options(-O2)

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
find_package(OpenCV REQUIRED)

# 查找 CUDA (可选)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # 添加 CUDA 编译选项
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
    add_definitions(-DWITH_CUDA)
else()
    message(STATUS "CUDA not found, building without CUDA support")
endif()

# 手动设置 OpenVINO 库路径
set(OPENVINO_INCLUDE_DIRS "${ABS_PREFIX_PREFIX}/include")
set(OPENVINO_LIBRARY_DIRS "${ABS_PREFIX_PREFIX}/lib")
set(OPENVINO_LIBRARIES openvino openvino_c)

# 检查 OpenVINO 库是否存在
if(NOT EXISTS "${ABS_PREFIX_PREFIX}/lib/libopenvino.so")
    message(WARNING "OpenVINO libraries not found at expected location")
endif()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${OpenCV_INCLUDE_DIRS}
  ${OPENVINO_INCLUDE_DIRS}
)

# 如果找到 CUDA，添加 CUDA 包含目录
if(CUDA_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

target_link_directories(${PROJECT_NAME} PUBLIC 
  ${OPENVINO_LIBRARY_DIRS}
)

# 链接库
set(LINK_LIBS ${OpenCV_LIBS} ${OPENVINO_LIBRARIES})

# 如果找到 CUDA，添加 CUDA 库
if(CUDA_FOUND)
    list(APPEND LINK_LIBS ${CUDA_LIBRARIES})
    # 设置 CUDA 架构 (可根据需要调整)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75 80 86)
endif()

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

  # Tests disabled due to missing dependencies
  # find_package(ament_cmake_gtest)
  # ament_add_gtest(test_node_startup test/test_node_startup.cpp)
  # target_link_libraries(test_node_startup ${PROJECT_NAME})
  # ament_add_gtest(test_number_cls test/test_number_cls.cpp)
  # target_link_libraries(test_number_cls ${PROJECT_NAME})

endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model
)
