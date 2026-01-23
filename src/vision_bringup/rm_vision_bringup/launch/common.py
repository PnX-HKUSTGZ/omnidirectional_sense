import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.actions import Node

launch_params = yaml.safe_load(open(os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'launch_params.yaml')))

_config_dir = os.path.join(get_package_share_directory('rm_vision_bringup'), 'config')
camera_param_files = launch_params.get('camera_param_files', {})

# 原始的 node params 文件路径（保持兼容）
node_params = os.path.join(
    get_package_share_directory('rm_vision_bringup'), 'config', 'node_params.yaml')

# 解析 node_params.yaml，提取通用的参数字典，便于将同一份参数应用到多个节点实例上
_raw_node_params = yaml.safe_load(open(node_params)) if os.path.exists(node_params) else {}

def _load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data else {}

def _camera_param_path(cam_id):
    key = cam_id if cam_id in camera_param_files else str(cam_id)
    fname = camera_param_files.get(key)
    if not fname:
        return None
    return os.path.join(_config_dir, fname)

def load_cam_params(cam_id):
    path = _camera_param_path(cam_id)
    if not path:
        return {}
    return _load_yaml(path)

# helper to extract ros__parameters dict safely from possible keys with or without leading '/'
def _extract(params, key):
    if not params:
        return {}
    for k in (key, f"/{key}"):
        v = params.get(k)
        if isinstance(v, dict) and 'ros__parameters' in v:
            return v['ros__parameters']
    return {}

# 共享参数: video_reader / armor_detector / gpu_cam_minimal (其他节点可按需添加)
video_reader_shared_params = _extract(_raw_node_params, 'video_reader_node')
armor_detector_shared_params = _extract(_raw_node_params, 'armor_detector')
gpu_cam_shared_params = _extract(_raw_node_params, 'gpu_cam_minimal')

def video_reader_params_for(cam_id):
    return _extract(load_cam_params(cam_id), 'video_reader_node')

def gpu_cam_params_for(cam_id):
    return _extract(load_cam_params(cam_id), 'gpu_cam_minimal')

# 为每个摄像头创建独立的 robot_state_publisher
# 所有摄像头共享 odom_omni 和 omni_gimbal_link，但各自有独立的 camera_link
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