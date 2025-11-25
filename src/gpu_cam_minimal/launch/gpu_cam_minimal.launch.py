from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    shared_params = PathJoinSubstitution([
        FindPackageShare('gpu_cam_minimal'),
        'config',
        'camera_controls.yaml'
    ])

    return LaunchDescription([
        Node(
            package='gpu_cam_minimal',
            executable='gpu_cam_minimal_node',
            name='gpu_cam_minimal',
            output='screen',
            parameters=[
                shared_params,
                {
                'camera_name': 'cam_0',
                'camera_info_url': 'file:///home/ori/rmvision/omnidirectional_sense/src/gpu_cam_minimal/config/camera_info.yaml',
                'frame_id': 'cam_0',
                'framerate': 30.0,
                'image_width': 1920,
                'image_height': 1080,
                'video_device': '/dev/video0',
                'publish_mode': 'gpu',  # 'cpu' or 'gpu'
                'pixel_format': 'mjpeg', 
                'debug': True,
                }
            ]
        )
    ])
