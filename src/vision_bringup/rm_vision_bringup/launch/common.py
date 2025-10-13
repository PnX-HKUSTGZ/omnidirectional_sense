import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

launch_params = yaml.safe_load(open(os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'launch_params.yaml')))

robot_description = Command(['xacro ', os.path.join(
    get_package_share_directory('rm_gimbal_description'), 'urdf', 'rm_gimbal.urdf.xacro'),
    ' xyz:=', launch_params['odom2camera']['xyz'], ' rpy:=', launch_params['odom2camera']['rpy']])

robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description,
                 'publish_frequency': 1000.0}]
)

node_params = os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'node_params.yaml')

tracker_node = Node(
    package='armor_tracker',
    executable='armor_tracker_node',
    output='both',
    emulate_tty=True,
    parameters=[node_params],
    ros_arguments=['--log-level', 'armor_tracker:='+launch_params['tracker_log_level']],
)

ballistic_node = Node(
    package='ballistic_calculation',
    executable='ballistic_calculation_node',
    output='screen',
    emulate_tty=True,
    parameters=[
        node_params
    ],
    
)

video_reader_node = Node(
    package='video_reader',
    executable='video_reader_node',
    output='screen',
    emulate_tty=True,
    parameters=[
        node_params
    ],
)

rune_solver_node = Node(
    package='rune_solver',
    executable='rune_solver_node',
    output='screen',
    emulate_tty=True,
    parameters=[
        node_params
    ],
)
recorder_node = Node(
    package='topic_recorder',
    executable='topic_recorder_node',
    name='topic_recorder_node',
    output='screen',
    emulate_tty=True,
    parameters=[{'config_path': os.path.join(
                                get_package_share_directory('rm_vision_bringup'), 'config', 'topic_record_params.yaml')}],
)