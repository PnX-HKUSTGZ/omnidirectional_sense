import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

launch_params = yaml.safe_load(open(os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'launch_params.yaml')))

# 原始的 node params 文件路径（保持兼容）
node_params = os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'node_params.yaml')

# 解析 node_params.yaml，提取通用的参数字典，便于将同一份参数应用到多个节点实例上
_raw_node_params = yaml.safe_load(open(node_params)) if os.path.exists(node_params) else {}

# helper to extract ros__parameters dict safely from possible keys with or without leading '/'
def _extract(params, key):
    if not params:
        return {}
    for k in (key, f"/{key}"):
        v = params.get(k)
        if isinstance(v, dict) and 'ros__parameters' in v:
            return v['ros__parameters']
    return {}

# 共享参数:video_reader 和 armor_detector(其他节点可按需添加)
video_reader_shared_params = _extract(_raw_node_params, 'video_reader_node')
armor_detector_shared_params = _extract(_raw_node_params, 'armor_detector')
usb_cam_shared_params = _extract(_raw_node_params, 'usb_cam_node')

# 为每个摄像头创建独立的 robot_state_publisher
# 所有摄像头共享 base_link 和 gimbal_link，但各自有独立的 camera_link
def create_robot_state_publisher(cam_id):
    robot_description = Command(['xacro ', os.path.join(
        get_package_share_directory('rm_gimbal_description'), 'urdf', 'rm_gimbal.urdf.xacro'),
        ' xyz:=', launch_params[f'odom2camera_{cam_id}']['xyz'], 
        ' rpy:=', launch_params[f'odom2camera_{cam_id}']['rpy'],
        ' camera_name:=', f'camera_{cam_id}'])
    
    return Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name=f'robot_state_publisher_{cam_id}',
        parameters=[{'robot_description': robot_description,
                     'publish_frequency': 1000.0}],
    )
serial_driver_node = Node(
    package='rm_serial_driver',
    executable='virtual_serial_node',
    name='virtual_serial',
    output='both',
    emulate_tty=True,
    parameters=[node_params],
    ros_arguments=['--ros-args', '-p', 'has_rune:=true' if launch_params['rune'] else 'has_rune:=false'],
)