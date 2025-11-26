from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    shared_params = PathJoinSubstitution([
        FindPackageShare('gpu_cam_minimal'),
        'config',
        'camera_controls.yaml'
    ])

    node_params = [
        shared_params,
        {
            'camera_name': 'cam_0',
            'camera_info_url': 'file:///home/ori/rmvision/omnidirectional_sense/src/gpu_cam_minimal/config/camera_info.yaml',
            'frame_id': 'cam_0',
            'framerate': 30.0,
            'image_width': 1920,
            'image_height': 1080,
            'video_device': '/dev/video1',
            'debug': True,
        }
    ]

    composable = ComposableNode(
        package='gpu_cam_minimal',
        plugin='GpuCamMinimalNode',
        name='gpu_cam_minimal',
        parameters=node_params,
    )

    container = ComposableNodeContainer(
        name='gpu_cam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='screen',
        composable_node_descriptions=[composable],
        emulate_tty=True,
    )

    return LaunchDescription([container])
