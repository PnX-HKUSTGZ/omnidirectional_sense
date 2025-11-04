# gpu_cam_minimal

Minimal ROS 2 node that:
- Captures frames from a video device using OpenCV (CAP_V4L2).
- Uploads each frame to GPU via OpenCV CUDA (if available at build time).
- Publishes the frame as `sensor_msgs/Image` with encoding `bgr8`.

No pixel format conversions are performed in the code.

## Parameters
- `device_id` (int, default 0): Video device index (e.g., `/dev/video0`).
- `width` (int, default 640)
- `height` (int, default 480)
- `fps` (int, default 30)
- `frame_id` (string, default `camera`)
- `image_topic` (string, default `image_raw`)

## Run
Launch:

```bash
ros2 launch gpu_cam_minimal gpu_cam_minimal.launch.py
```

Or run directly:

```bash
ros2 run gpu_cam_minimal gpu_cam_minimal_node --ros-args -p device_id:=0 -p width:=1280 -p height:=720 -p fps:=30
```

Notes:
- If OpenCV was built without CUDA, the node still runs and publishes images, but logs a warning and skips GPU upload.
- The published `sensor_msgs/Image` uses `bgr8` as provided by OpenCV `VideoCapture` without explicit conversion in this node.
