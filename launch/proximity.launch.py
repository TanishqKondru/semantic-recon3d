"""Stereo player + proximity node (+ RViz). Run from the repo root inside the sr3d-ros env:

    ros2 launch launch/proximity.launch.py rate:=15.0 rviz:=true
    ros2 launch launch/proximity.launch.py lockstep:=true          # process every frame
    ros2 launch launch/proximity.launch.py tool_radius_mm:=0.0     # visible instrument surface only (no shape prior)
"""
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as Arg


def generate_launch_description():
    args = {"scene": "data/scenes/stereomis_P2_1_tools", "rate": "15.0", "lockstep": "false",
            "segmenter": "outputs/segmentation/segformer_b2_fluor/best", "raft_iters": "12",
            "log": "outputs/ros/node_timing.csv", "rviz": "false", "start": "0", "stop": "-1",
            "backend": "pytorch", "live_dilate_px": "12", "erode_px": "1",
            "tool_radius_mm": "4.0", "tool_model": "tiered", "gpu_nn": "true"}
    py = [sys.executable, "-m"]
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in args.items()],
        ExecuteProcess(cmd=[*py, "sr3d.ros.node", "--ros-args", "-p", ["segmenter:=", Arg("segmenter")],
                            "-p", ["raft_iters:=", Arg("raft_iters")], "-p", ["log:=", Arg("log")],
                            "-p", ["backend:=", Arg("backend")], "-p", ["live_dilate_px:=", Arg("live_dilate_px")],
                            "-p", ["erode_px:=", Arg("erode_px")],
                            "-p", ["tool_radius_mm:=", Arg("tool_radius_mm")], "-p", ["tool_model:=", Arg("tool_model")],
                            "-p", ["gpu_nn:=", Arg("gpu_nn")]],
                       name="sr3d_proximity", output="screen"),
        ExecuteProcess(cmd=[*py, "sr3d.ros.player", "--ros-args", "-p", ["scene:=", Arg("scene")],
                            "-p", ["rate:=", Arg("rate")], "-p", ["lockstep:=", Arg("lockstep")],
                            "-p", ["start:=", Arg("start")], "-p", ["stop:=", Arg("stop")]],
                       name="stereo_player", output="screen", on_exit=Shutdown()),
        ExecuteProcess(cmd=["rviz2", "-d", "launch/proximity.rviz"], condition=IfCondition(Arg("rviz")),
                       name="rviz", output="log"),
    ])
