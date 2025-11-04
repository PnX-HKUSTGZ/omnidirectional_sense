from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gpu_cam_minimal',
            executable='gpu_cam_minimal_node',
            name='gpu_cam_minimal',
            output='screen',
            parameters=[{
                'device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'frame_id': 'camera',
                'image_topic': 'image_raw'
            }]
        )
    ])
