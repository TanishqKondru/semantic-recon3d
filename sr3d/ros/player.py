"""Replays a converted stereo scene as a robot would publish it: rectified images, camera info, and the
kinematic camera pose, all with the same stamp. The pose goes out both as a stamped topic
(/stereo/left/pose, like a robot's measured Cartesian pose) and on /tf (world -> camera_left) for RViz.

    python -m sr3d.ros.player --ros-args -p scene:=data/scenes/stereomis_P2_1_tools -p rate:=15.0

`lockstep:=true` publishes the next frame only after the perception node has answered the previous one
(every frame processed, e.g. to record a bag); otherwise frames go out at `rate` and the node keeps up or drops.
"""
from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, Range
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ..data.scene import Scene
from .msgs import camera_info, image_msg, pose_msg, transform_msg


class StereoPlayer(Node):
    def __init__(self):
        super().__init__("stereo_player")
        self.declare_parameter("scene", "data/scenes/stereomis_P2_1_tools")
        self.declare_parameter("rate", 15.0)
        self.declare_parameter("lockstep", False)
        self.declare_parameter("loop", False)
        self.declare_parameter("start", 0)
        self.declare_parameter("stop", -1)        # exclusive; -1 = end of scene
        self.scene = Scene(self.get_parameter("scene").value)
        self.lockstep = self.get_parameter("lockstep").value
        self.loop = self.get_parameter("loop").value
        m = self.scene.meta
        self.pub = {k: self.create_publisher(t, f"/stereo/{k}", 2) for k, t in
                    (("left/image_rect", Image), ("right/image_rect", Image),
                     ("left/camera_info", CameraInfo), ("right/camera_info", CameraInfo),
                     ("left/pose", PoseStamped))}
        stop = self.get_parameter("stop").value
        idx = range(self.get_parameter("start").value, len(self.scene) if stop < 0 else stop)
        self.frames = [self.scene[i] for i in idx]    # decode up front so PNG reads do not add jitter
        self.tf = TransformBroadcaster(self)
        self.info = (m.width, m.height, m.baseline_m or 0.0)
        right = np.eye(4)
        right[0, 3] = self.info[2]    # rectified pair: the right camera sits +baseline along x
        # scene_view: z-up frame at the mean camera pose (x right, y forward, z = image up) for RViz's orbit view
        poses = np.stack([f.T_wc for f in self.frames])
        u, _, vt = np.linalg.svd(poses[:, :3, :3].sum(0))
        R = u @ vt                                    # chordal mean rotation
        view = np.eye(4)
        view[:3, :3] = np.column_stack([R[:, 0], R[:, 2], -R[:, 1]])
        view[:3, 3] = poses[:, :3, 3].mean(0)
        now = self.get_clock().now().to_msg()
        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform([transform_msg(right, Header(stamp=now, frame_id="camera_left"), "camera_right"),
                                      transform_msg(view, Header(stamp=now, frame_id="world"), "scene_view")])
        self.i = 0
        self.sent = None    # time the unanswered lockstep frame went out
        if self.lockstep:
            self.create_subscription(Range, "/sr3d/min_distance", self._answered, 10)
            self.create_timer(0.5, self._kick)
        else:
            self.create_timer(1.0 / self.get_parameter("rate").value, self._publish)
        self.get_logger().info(f"{m.name}: frames {idx.start}-{idx.stop - 1}, lockstep={self.lockstep}")

    def _answered(self, _msg):
        self.sent = None
        self._publish()

    def _kick(self):
        """Start once the perception node is up; resend a frame it dropped (e.g. before it had camera_info)."""
        if self.count_publishers("/sr3d/min_distance") == 0:
            return
        if self.sent is None and self.i == 0:
            self._publish()
        elif self.sent is not None and (self.get_clock().now() - self.sent).nanoseconds > 5e9:
            self.i -= 1
            self._publish()

    def _publish(self):
        if self.i >= len(self.frames):
            if not self.loop:
                self.get_logger().info("end of scene")
                raise SystemExit
            self.i = 0
        f = self.frames[self.i]
        w, h, B = self.info
        stamp = self.get_clock().now().to_msg()
        world = Header(stamp=stamp, frame_id="world")
        left, right = Header(stamp=stamp, frame_id="camera_left"), Header(stamp=stamp, frame_id="camera_right")
        self.tf.sendTransform(transform_msg(f.T_wc, world, "camera_left"))
        self.pub["left/pose"].publish(pose_msg(f.T_wc, world))
        self.pub["left/camera_info"].publish(camera_info(self.scene.K, w, h, left))
        self.pub["right/camera_info"].publish(camera_info(self.scene.K, w, h, right, B))
        self.pub["right/image_rect"].publish(image_msg(f.right, right))
        self.pub["left/image_rect"].publish(image_msg(f.rgb, left))
        self.i += 1
        if self.lockstep:
            self.sent = self.get_clock().now()


def main():
    rclpy.init()
    node = StereoPlayer()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt):
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
