"""ros2 launch railguard railguard.launch.py [config:=/path/to/railguard.yaml]"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default = os.path.join(get_package_share_directory('railguard'), 'config', 'railguard.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default),
        Node(package='railguard', executable='railguard_node', parameters=[LaunchConfiguration('config')], output='screen'),
    ])
