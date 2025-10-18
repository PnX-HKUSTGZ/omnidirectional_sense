cmake_minimum_required(VERSION 3.8)
project(video_reader)

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

# Ensure docs directory exists to avoid install-time errors
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/docs")
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(camera_info_manager REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
    src/video_reader_node.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${camera_info_manager_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${camera_info_manager_LIBRARIES}
)

rclcpp_components_register_node(${PROJECT_NAME}
    PLUGIN video_reader::VideoReaderNode
    EXECUTABLE ${PROJECT_NAME}_node
)

#############
## Testing ##
#############

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
    INSTALL_TO_SHARE
    docs
    config
)