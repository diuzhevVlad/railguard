"""ros2 launch railguard railguard.launch.py [config:=...] [bag:=new_data | initial_speed:=16.0] [input_topic:=/lidar_points] [odometry:=true]

KISS-ICP must be seeded like the GT odometry of the recording: bag:=<directory name> selects the preset (speed and seed
interval) from obstacle_app/gt_odometry.py, the same table the web app uses; for a recording without a preset give
initial_speed:=<m/s>. One of the two is required when odometry is on. odometry:=false starts the detection node alone in
single-scan mode (no KISS-ICP, no temporal confirmation: the immediate rule only). input_topic defaults to the YAML value."""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from obstacle_app.gt_odometry import GT_ODOMETRY_PRESETS
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
    if bag and bag not in GT_ODOMETRY_PRESETS and not speed:
        raise RuntimeError(f'no GT odometry preset for bag {bag!r}: give initial_speed:=<m/s> (RAILGUARD_INITIAL_SPEED) as well')
    if not bag and not speed:
        raise RuntimeError('the KISS-ICP seed must match the recording: give bag:=<recording directory name> (RAILGUARD_BAG) '
                           'for a known recording, or initial_speed:=<m/s> (RAILGUARD_INITIAL_SPEED); see README')
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
        DeclareLaunchArgument('bag', default_value=os.environ.get('RAILGUARD_BAG', ''), description='recording directory name: selects the GT odometry preset'),
        DeclareLaunchArgument('initial_speed', default_value=os.environ.get('RAILGUARD_INITIAL_SPEED', ''), description='KISS-ICP initial speed, m/s (required without a preset)'),
        DeclareLaunchArgument('input_topic', default_value=os.environ.get('RAILGUARD_INPUT_TOPIC', ''), description='PointCloud2 topic of the bag'),
        DeclareLaunchArgument('odometry', default_value=os.environ.get('RAILGUARD_ODOMETRY', 'true'), description='false = detection node only, single-scan mode'),
        OpaqueFunction(function=launch_nodes),
    ])
