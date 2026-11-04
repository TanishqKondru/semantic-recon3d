"""numpy <-> ROS 2 message conversions (no cv_bridge, so the node only needs rclpy and the standard msgs)."""
from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField, Range
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from .proximity import Tool


def image_msg(rgb: np.ndarray, header: Header) -> Image:
    h, w = rgb.shape[:2]
    return Image(header=header, height=h, width=w, encoding="rgb8", is_bigendian=0, step=w * 3,
                 data=np.ascontiguousarray(rgb, np.uint8).tobytes())


def image_array(msg: Image) -> np.ndarray:
    if msg.encoding != "rgb8":
        raise ValueError(f"expected rgb8, got {msg.encoding}")
    return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[:, :msg.width]


def camera_info(K: np.ndarray, w: int, h: int, header: Header, baseline_m: float = 0.0) -> CameraInfo:
    """Rectified camera; for the right camera pass the baseline so P carries Tx = -fx * B (ROS convention)."""
    P = np.zeros((3, 4))
    P[:, :3] = K
    P[0, 3] = -K[0, 0] * baseline_m
    return CameraInfo(header=header, width=w, height=h, distortion_model="plumb_bob", d=[0.0] * 5,
                      k=K.ravel().tolist(), r=np.eye(3).ravel().tolist(), p=P.ravel().tolist())


def transform_msg(T: np.ndarray, header: Header, child: str) -> TransformStamped:
    from scipy.spatial.transform import Rotation
    msg = TransformStamped(header=header, child_frame_id=child)
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, T[:3, 3])
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = x, y, z, w
    return msg


def pose_msg(T: np.ndarray, header: Header) -> PoseStamped:
    t = transform_msg(T, header, "").transform
    msg = PoseStamped(header=header)
    msg.pose.position = Point(x=t.translation.x, y=t.translation.y, z=t.translation.z)
    msg.pose.orientation = t.rotation
    return msg


def pose_matrix(msg: PoseStamped) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    p, q = msg.pose.position, msg.pose.orientation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = p.x, p.y, p.z
    return T


def transform_matrix(msg: TransformStamped) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    t, q = msg.transform.translation, msg.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = t.x, t.y, t.z
    return T


def _pack_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 -> RViz's packed-float rgb."""
    c = rgb_u8.astype(np.uint32)
    return ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)


_SEMANTIC_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z", "rgb", "label", "confidence"))]
_XYZ_FIELDS = _SEMANTIC_FIELDS[:3]
_XYZRGB_FIELDS = _SEMANTIC_FIELDS[:4]


def _cloud(header: Header, fields: list[PointField], data: np.ndarray) -> PointCloud2:
    data = np.ascontiguousarray(data, np.float32)
    return PointCloud2(header=header, height=1, width=len(data), fields=fields, is_bigendian=False,
                       point_step=4 * data.shape[1], row_step=4 * data.size, data=data.tobytes(), is_dense=True)


def semantic_cloud(depth: np.ndarray, rgb: np.ndarray, prob: np.ndarray, K: np.ndarray, header: Header,
                   stride: int = 2) -> PointCloud2:
    """Camera-frame cloud of the current frame: x y z rgb label (0 tissue, 1 instrument) confidence."""
    d = depth[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    z = d[v, u]
    u, v = u * stride, v * stride
    packed = _pack_rgb(rgb[v, u])
    p = prob[v, u]
    label = (p > 0.5).astype(np.float32)
    data = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, packed, label,
                     np.where(label > 0, p, 1 - p)], 1)
    return _cloud(header, _SEMANTIC_FIELDS, data)


def xyz_cloud(points: np.ndarray, header: Header, rgb: np.ndarray | None = None) -> PointCloud2:
    """Plain cloud; `rgb` (N, 3) in [0, 1] adds colour."""
    if rgb is None:
        return _cloud(header, _XYZ_FIELDS, points)
    packed = _pack_rgb(np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8))
    return _cloud(header, _XYZRGB_FIELDS, np.column_stack([points, packed]))


def distance_msg(d: float, header: Header, max_m: float = 0.2) -> Range:
    """Minimum instrument-to-tissue distance as a stamped range; +inf when there is no instrument (REP 117)."""
    return Range(header=header, radiation_type=Range.INFRARED, field_of_view=0.0, min_range=0.0, max_range=max_m,
                 range=float(d) if np.isfinite(d) else float("inf"))


def tips_msg(tools: list[Tool], header: Header) -> PoseArray:
    msg = PoseArray(header=header)
    for t in tools:
        if t.tip is not None:
            pose = Pose()
            pose.position = Point(x=float(t.tip[0]), y=float(t.tip[1]), z=float(t.tip[2]))
            pose.orientation.w = 1.0
            msg.poses.append(pose)
    return msg


def _colour(d_m: float) -> ColorRGBA:
    if d_m < 2e-3:
        return ColorRGBA(r=0.85, g=0.1, b=0.1, a=1.0)
    if d_m < 5e-3:
        return ColorRGBA(r=0.95, g=0.6, b=0.1, a=1.0)
    return ColorRGBA(r=0.2, g=0.8, b=0.3, a=1.0)


def proximity_markers(tools: list[Tool], header: Header) -> MarkerArray:
    """Tip spheres, a line from each instrument's closest point to the tissue, and the distance as text."""
    out = MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)])
    for k, t in enumerate(tools):
        if t.tip is not None:
            m = Marker(header=header, ns="tips", id=k, type=Marker.SPHERE, action=Marker.ADD)
            m.pose.position = Point(x=float(t.tip[0]), y=float(t.tip[1]), z=float(t.tip[2]))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.5e-3
            m.color = ColorRGBA(r=0.9, g=0.2, b=0.9, a=1.0)
            out.markers.append(m)
        if t.closest_tool is None or not np.isfinite(t.distance):
            continue
        a, b = t.closest_tool, t.closest_tissue
        line = Marker(header=header, ns="closest", id=k, type=Marker.LINE_LIST, action=Marker.ADD,
                      points=[Point(x=float(a[0]), y=float(a[1]), z=float(a[2])),
                              Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))])
        line.pose.orientation.w = 1.0
        line.scale.x = 0.6e-3
        line.color = _colour(t.distance)
        text = Marker(header=header, ns="distance", id=k, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                      text=f"{t.distance * 1e3:.1f} mm")
        away = (a - b) / max(np.linalg.norm(a - b), 1e-9)    # label sits on the far side of the instrument
        pos = a + 4e-3 * away
        text.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        text.pose.orientation.w = 1.0
        text.scale.z = 3e-3
        text.color = line.color
        out.markers += [line, text]
    return out
