import os
import sys
from ament_index_python.packages import get_package_share_directory
sys.path.append(os.path.join(get_package_share_directory('rm_vision_bringup'), 'launch'))


def generate_launch_description():

    from common import node_params, launch_params, create_robot_state_publisher, serial_driver_node, video_reader_shared_params, armor_detector_shared_params
    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription
    def get_video_reader_node(package, plugin, name='video_reader_node', remappings=None, frame_id='camera_optical_frame', camera_name='camera'):
        # 使用共享的 video_reader 参数，并允许传入每个实例的 frame_id 与 camera_name
        return ComposableNode(
            package=package,
            plugin=plugin,
            name=name,
            parameters=[video_reader_shared_params, {'frame_id': frame_id, 'camera_name': camera_name}],
            remappings=remappings or [],
            extra_arguments=[{'use_intra_process_comms': True}]
        )

    def get_camera_detector_container(*nodes, container_name='camera_detector_container'):
        node_list = list(nodes)
        workspace_root =os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(get_package_share_directory('rm_vision_bringup'))))))
        third_party_lib_path = os.path.join(workspace_root, 'third_party_install', 'lib')
        
        container = ComposableNodeContainer(
            name=container_name,
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=node_list,
            output='both',
            emulate_tty=True,
            additional_env={'LD_LIBRARY_PATH': third_party_lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')},
            ros_arguments=['--ros-args', ],
        )
        return TimerAction(
            period=2.0,
            actions=[container],
        )

    def make_armor_detector_node(name='armor_detector', remappings=None):
        # 使用共享的 armor_detector 参数
        return ComposableNode(
            package='armor_detector',
            plugin='rm_auto_aim::ArmorDetectorNode',
            name=name,
            parameters=[armor_detector_shared_params, {'use_ai_detector': True}],
            remappings=remappings or [],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
    
    image_nodes = []
    detector_nodes = []
    cam_detectors = []
    robot_state_publishers = []  # 添加列表来存储所有的 robot_state_publisher
    
    for i in range(4):
        node_name = f"video_reader_node_{i}"
        det_name = f"armor_detector_{i}"
        # Per-instance topic remappings
        cam_ns = f"/cam{i}"
        frame_id = f"camera_{i}_optical_frame"  # 每个摄像头的独立 frame_id
        image_remaps = [
            ('/image_gpu', f'{cam_ns}/image_gpu'),
            ('/camera_info', f'{cam_ns}/camera_info'),
        ]
        detector_remaps = [
            # subscriptions
            ('/image_gpu', f'{cam_ns}/image_gpu'),
            ('/camera_info', f'{cam_ns}/camera_info'),
            # publications
            ('/detector/armors', f'{cam_ns}/detector/armors'),
            ('/detector/cars', f'{cam_ns}/detector/cars'),
            ('/detector/marker', f'{cam_ns}/detector/marker'),
            ('/detector/result_img', f'{cam_ns}/detector/result_img'),
        ]
        if launch_params['video_play']:
            image_node = get_video_reader_node('video_reader', 'video_reader::VideoReaderNode', name=node_name, remappings=image_remaps, frame_id=frame_id, camera_name=f'camera_{i}')
        else:
            # camera package does not exist, use video_reader as fallback
            print("hik_camera package does not exist, use video_reader as fallback")
            image_node = get_video_reader_node('video_reader', 'video_reader::VideoReaderNode', name=node_name, remappings=image_remaps, frame_id=frame_id, camera_name=f'camera_{i}')
        armor_detector_node = make_armor_detector_node(name=det_name, remappings=detector_remaps)
        container_name = f"camera_detector_container_{i}"
        cam_detector = get_camera_detector_container(image_node, armor_detector_node, container_name=container_name)
        
        # 为每个摄像头创建对应的 robot_state_publisher
        robot_state_pub = create_robot_state_publisher(i)
        
        image_nodes.append(image_node)
        detector_nodes.append(armor_detector_node)
        cam_detectors.append(cam_detector)
        robot_state_publishers.append(robot_state_pub)


    delay_serial_node = TimerAction(
        period=1.5,
        actions=[serial_driver_node],
    )


    if launch_params['unit_test']:
        launch_description_list = [
            *robot_state_publishers, 
            *cam_detectors,
            delay_serial_node,
        ]
    else:
        launch_description_list = [
            *robot_state_publishers, 
            *cam_detectors,
        ]

    return LaunchDescription(launch_description_list)
