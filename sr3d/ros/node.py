"""ROS 2 perception node: stereo images + kinematic camera pose in, instrument-to-tissue proximity out.

Subscribes   /stereo/{left,right}/image_rect (rgb8), /stereo/left/pose (kinematic camera pose, world frame),
             all with the same stamp; /stereo/right/camera_info
Publishes    /sr3d/min_distance     sensor_msgs/Range, closest instrument-tissue distance (+inf: no instrument)
             /sr3d/tool_tips        geometry_msgs/PoseArray, instrument tips (world)
             /sr3d/proximity        visualization_msgs/MarkerArray, tips + closest-point lines + distance text
             /sr3d/semantic_cloud   sensor_msgs/PointCloud2, current frame x y z rgb label confidence (camera_left)
             /sr3d/instrument_cloud sensor_msgs/PointCloud2, current instrument surface points (world)
             /sr3d/tissue_map       sensor_msgs/PointCloud2, fused tissue surface with colour (world), on refresh

Image queues are one deep, so when perception is slower than the camera the node works on the newest frame
and drops the rest instead of falling behind. The pose arrives as a topic synchronised with the images rather
than through a tf2 lookup: under GPU load the tf listener thread fell ~0.5 s behind and lookups failed. Per-frame timings go to `log` (CSV) for the latency report.

    python -m sr3d.ros.node --ros-args -p segmenter:=outputs/segmentation/segformer_b2_fluor/best
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped
from message_filters import Subscriber, TimeSynchronizer
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, Range
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from .msgs import (
    distance_msg,
    image_array,
    pose_matrix,
    proximity_markers,
    semantic_cloud,
    tips_msg,
    xyz_cloud,
)
from .pipeline import ProximityPipeline

STAGES = ("depth", "segmentation", "fusion", "proximity")


def ms(later: Time, earlier: Time) -> float:
    return (later - earlier).nanoseconds / 1e6


class ProximityNode(Node):
    def __init__(self):
        super().__init__("sr3d_proximity")
        p = {k: self.declare_parameter(k, v).value for k, v in {
            "segmenter": "outputs/segmentation/segformer_b2_fluor/best", "raft_iters": 12, "map_every": 10,
            "backend": "pytorch", "engines": "outputs/deploy", "precision": "fp16",
            "live_tissue": True, "live_dilate_px": 12, "erode_px": 1,
            "world_frame": "world", "cloud_stride": 2, "map_stride": 1,
            "log": "outputs/ros/node_timing.csv"}.items()}
        self.p = p
        self.K = self.baseline = None
        self.pipe: ProximityPipeline | None = None

        if p["backend"] == "tensorrt":                       # heavy imports after rclpy is up
            from ..deploy.engines import TRTSegmenter, TRTStereo
            eng = Path(p["engines"])
            self.raft = TRTStereo(eng / f"raft_{p['precision']}.engine")
            self.seg = TRTSegmenter(eng / f"segformer_{p['precision']}.engine")
        else:
            from ..depth.raft import RAFTStereo
            from ..segmentation.model import InstrumentSegmenter
            self.raft = RAFTStereo(iters=p["raft_iters"])
            self.seg = InstrumentSegmenter(p["segmenter"])

        self.create_subscription(CameraInfo, "/stereo/right/camera_info", self._info, 1)
        qos = QoSProfile(depth=1)
        self.sync = TimeSynchronizer([Subscriber(self, Image, "/stereo/left/image_rect", qos_profile=qos),
                                      Subscriber(self, Image, "/stereo/right/image_rect", qos_profile=qos),
                                      Subscriber(self, PoseStamped, "/stereo/left/pose", qos_profile=qos)], 2)
        self.sync.registerCallback(self._frame)

        self.pub_d = self.create_publisher(Range, "/sr3d/min_distance", 10)
        self.pub_tips = self.create_publisher(PoseArray, "/sr3d/tool_tips", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/sr3d/proximity", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, "/sr3d/semantic_cloud", 2)
        self.pub_tool_cloud = self.create_publisher(PointCloud2, "/sr3d/instrument_cloud", 2)
        self.pub_map = self.create_publisher(PointCloud2, "/sr3d/tissue_map", 1)

        Path(p["log"]).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(p["log"], "w", newline="")  # noqa: SIM115 - open for the node's lifetime
        self.log = csv.writer(self.log_file)
        self.log.writerow(["frame", "stamp", "queue_ms", *[f"{s}_ms" for s in STAGES], "compute_ms", "publish_ms",
                           "latency_ms", "distance_mm", "tools", "map_points"])
        self.n = 0
        self.get_logger().info(f"{p['backend']} models loaded, waiting for camera_info")

    def _info(self, msg: CameraInfo):
        if self.pipe is not None:
            return
        self.K = np.array(msg.k).reshape(3, 3)
        self.baseline = -msg.p[3] / msg.p[0]
        fx, B = self.K[0, 0], self.baseline
        self.pipe = ProximityPipeline(self.K, lambda l, r: self.raft.depth(l, r, fx, B), self.seg.prob,
                                      map_every=self.p["map_every"], live_tissue=self.p["live_tissue"],
                                      live_dilate_px=self.p["live_dilate_px"], erode_px=self.p["erode_px"])
        self.get_logger().info(f"camera: fx {fx:.1f} px, baseline {B * 1e3:.2f} mm")

    def _frame(self, left: Image, right: Image, pose: PoseStamped):
        t_recv = self.get_clock().now()
        if self.pipe is None:
            return
        stamp = Time.from_msg(left.header.stamp)
        T_wc = pose_matrix(pose)
        rgb_l, rgb_r = image_array(left), image_array(right)
        r = self.pipe.step(rgb_l, rgb_r, T_wc)

        t0 = time.perf_counter()
        world = Header(stamp=left.header.stamp, frame_id=self.p["world_frame"])
        self.pub_d.publish(distance_msg(r.distance, left.header))
        t_done = self.get_clock().now()
        self.pub_tips.publish(tips_msg(r.tools, world))
        self.pub_markers.publish(proximity_markers(r.tools, world))
        self.pub_cloud.publish(semantic_cloud(r.depth, rgb_l, r.prob, self.K, left.header, self.p["cloud_stride"]))
        tool_pts = [t.points for t in r.tools]
        self.pub_tool_cloud.publish(xyz_cloud(np.concatenate(tool_pts) if tool_pts else np.zeros((0, 3)), world))
        if r.map_refreshed and self.pipe.map is not None and self.pub_map.get_subscription_count():
            pts, rgb = self.pipe.tsdf.points(self.pipe.min_weight, colors=True)
            k = self.p["map_stride"]
            self.pub_map.publish(xyz_cloud(pts[::k], world, rgb[::k]))
        publish_ms = 1000 * (time.perf_counter() - t0)

        self.log.writerow([self.n, f"{stamp.nanoseconds / 1e9:.6f}", f"{ms(t_recv, stamp):.2f}",
                           *[f"{r.ms[s]:.2f}" for s in STAGES], f"{r.ms['total']:.2f}", f"{publish_ms:.2f}",
                           f"{ms(t_done, stamp):.2f}", f"{r.distance * 1e3:.3f}", len(r.tools), r.map_points])
        self.n += 1
        if self.n % 25 == 0:
            self.log_file.flush()
            self.get_logger().info(f"frame {self.n}: d = {r.distance * 1e3:.2f} mm, compute {r.ms['total']:.0f} ms, "
                                   f"latency {ms(t_done, stamp):.0f} ms")

    def destroy_node(self):
        self.log_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = ProximityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
