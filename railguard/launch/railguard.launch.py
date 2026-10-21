"""Launch with mandatory initial_speed:=<m/s>; optional bag selects the GT seed interval."""
import os
import math
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_nodes(context):
    config = LaunchConfiguration('config').perform(context)
    topic = LaunchConfiguration('input_topic').perform(context)
    speed = LaunchConfiguration('initial_speed').perform(context)
    bag = LaunchConfiguration('bag').perform(context).strip().strip('/').split('/')[-1]
    odometry = LaunchConfiguration('odometry').perform(context).strip().lower() not in ('0', 'false', 'no', 'off')
    if not odometry:
        override = {'/**': {'ros__parameters': {'use_odometry': False, **({'input_topic': topic} if topic else {})}}}
        f = tempfile.NamedTemporaryFile('w', suffix='.yaml', prefix='railguard_override_', delete=False)
        yaml.safe_dump(override, f); f.close()
        return [Node(package='railguard', executable='railguard_node', parameters=[config, f.name], output='screen')]
    if not speed or not math.isfinite(float(speed)) or float(speed) < 0:
        raise RuntimeError('initial_speed:=<m/s> (or RAILGUARD_INITIAL_SPEED) is required; use 0 for a stationary start')
    # Node-specific sections of the YAML take precedence over wildcard ('/**') overrides whatever the file order, so the
    # overrides are written as node-specific sections of a second parameter file.
    override = {}
    if topic:
        override['/**'] = {'ros__parameters': {'input_topic': topic}}
    odom = {}
    if bag:
        odom['bag'] = bag
    if speed:
        odom['initial_speed_m_s'] = float(speed)
    override['railguard_odometry'] = {'ros__parameters': odom}
    params = [config]
    if override:
        f = tempfile.NamedTemporaryFile('w', suffix='.yaml', prefix='railguard_override_', delete=False)
        yaml.safe_dump(override, f); f.close(); params.append(f.name)
    return [
        Node(package='railguard', executable='railguard_odometry', parameters=params, output='screen'),
        Node(package='railguard', executable='railguard_node', parameters=params, output='screen'),
    ]


def generate_launch_description():
    default = os.path.join(get_package_share_directory('railguard'), 'config', 'railguard.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default),
        DeclareLaunchArgument('bag', default_value=os.environ.get('RAILGUARD_BAG', ''), description='optional recording name: selects the GT initialization interval, not speed'),
        DeclareLaunchArgument('initial_speed', default_value=os.environ.get('RAILGUARD_INITIAL_SPEED', ''), description='required KISS-ICP initial speed, m/s (including known bags)'),
        DeclareLaunchArgument('input_topic', default_value=os.environ.get('RAILGUARD_INPUT_TOPIC', ''), description='PointCloud2 topic of the bag'),
        DeclareLaunchArgument('odometry', default_value=os.environ.get('RAILGUARD_ODOMETRY', 'true'), description='false = detection node only, single-scan mode'),
        OpaqueFunction(function=launch_nodes),
    ])
