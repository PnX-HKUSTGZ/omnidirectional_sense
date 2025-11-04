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
                'camera_name': 'default_cam',
                'camera_info_url': '',
                'frame_id': 'default_cam',
                'framerate': 30.0,
                'image_width': 640,
                'image_height': 480,
                'video_device': '/dev/video0',
                'publish_mode': 'cpu',  # 'cpu' or 'gpu'
                'pixel_format': 'mjpeg'
            }]
        )
    ])
