# rm_gimbal_description
RoboMaster 视觉自瞄系统所需的 URDF

<img src="docs/rm_vision.svg" alt="rm_vision" width="200" height="200">

该项目为 [rm_vision](https://github.com/chenjunnn/rm_vision) 的子模块

## 坐标系定义

单位和方向请参考 https://www.ros.org/reps/rep-0103.html

odom: 以云台中心为原点的惯性系

gimbal_joint: 描述云台到Odom系旋转关系，包括pitch和yaw，由电控发送获得

camera_joint: 表述相机到云台系的变换关系

camera_optical_joint: 表述以 z 轴为前方的相机坐标系转换为 x 轴为前方的相机坐标系的旋转关系

## 使用方法

修改 [urdf/rm_gimbal.urdf.xacro](urdf/rm_gimbal.urdf.xacro) 中的 `gimbal_camera_transfrom` 

xyz 与 rpy 对应机器人云台上相机到云台中心的平移与旋转关系，可以由机械图纸测量得到，或在机器人上直接测量
