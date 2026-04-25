"""
merge_launch.py - 合并相机标定启动文件
支持V4L2摄像头、海康摄像头以及 gpu_cam_minimal，可通过 camera_type 参数选择
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare

# 复用 common 中的工具
sys.path.append(os.path.join(get_package_share_directory('rm_vision_bringup'), 'launch'))
from common import get_gpu_cam_node

def generate_launch_description():
    camera_type = LaunchConfiguration('camera_type')
    camera_params_file = LaunchConfiguration('camera_params_file')
    camera_name = LaunchConfiguration('camera_name')
    video_device = LaunchConfiguration('video_device')

    camera_type_arg = DeclareLaunchArgument(
        'camera_type',
        default_value='gpu_cam',
        description='相机类型: v4l2 | hik | gpu_cam',
    )

    camera_params_arg = DeclareLaunchArgument(
        'camera_params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('rm_vision_bringup'),
            'config',
            'cam_mid_params.yaml',
        ]),
        description='gpu_cam_minimal 模式使用的参数文件, 示例见 cam_*_params.yaml',
    )

    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='camera_1',
        description='相机名称，需与 camera_info 中的 camera_name 一致',
    )

    video_device_arg = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/camera_mid',
        description='V4L2 设备路径，gpu_cam_minimal/v4l2 模式都会用到',
    )
    
    v4l2_camera_node = Node(
        package='gpu_cam_minimal',
        executable='v4l2_camera_node',
        name='wide_camera_node',
        namespace='',
        output='screen',
        parameters=[
            # 基础配置
            {'video_device': video_device},
            {'output_encoding': 'rgb8'},
            {'image_size': [1280, 720]},
            {'framerate': 30.0},
            {'camera_name': 'wide_camera'},
            
            # 图像质量参数（确保在合理范围内）
            {'brightness': 50},      # 0-100范围
            {'contrast': 50},        # 0-100范围
            {'saturation': 60},      # 0-100范围
            {'sharpness': 50},       # 0-100范围
            
            # 自动控制
            {'exposure_auto': 1},    # 自动曝光
            {'focus_auto': 1},       # 自动对焦
        ],
        condition=IfCondition(PythonExpression(["'", camera_type, "' == 'v4l2'"])),
    )

    hik_camera_node = Node(
        package='hik_camera',
        executable='hik_camera_node',
        name='hik_camera',
        output='screen',
        parameters=[
            {'width': 1280},
            {'height': 720},
            {'fps': 30.0},
            {'exposure_time': 5000},
            {'gain': 16.0},
            {'pixel_format': 'rgb8'},
        ],
        condition=IfCondition(PythonExpression(["'", camera_type, "' == 'hik'"])),
    )

    gpu_cam_composable = get_gpu_cam_node(
        cam_id=1,
        name='gpu_cam_node_1',
        remappings=[('/image_gpu', '/image_raw'), ('/camera_info', '/camera_info')],
        frame_id='camera_1_optical_frame',
        camera_name='camera_1',
    )

    gpu_cam_container = ComposableNodeContainer(
        name='gpu_cam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        emulate_tty=True,
        composable_node_descriptions=[gpu_cam_composable],
        condition=IfCondition(PythonExpression(["'", camera_type, "' == 'gpu_cam'"])),
    )
    
    # 3. 相机标定节点 - 动态选择话题
    # 使用条件表达式的正确方法
    calibration_node = Node(
        package='camera_calibration',
        executable='cameracalibrator',
        name='camera_calibrator',
        output='screen',
        arguments=[
            '--size', '7x7',
            '--square', '0.03',
            '--pattern', 'circles'
        ],
        remappings=[
            ('image', '/image_raw'),
        ]
    )
    
    return LaunchDescription([
        camera_type_arg,
        camera_params_arg,
        camera_name_arg,
        video_device_arg,
        # 根据条件启动对应的相机节点
        v4l2_camera_node,
        hik_camera_node,
        gpu_cam_container,
        calibration_node,
    ])
