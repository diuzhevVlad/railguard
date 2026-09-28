#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /opt/railguard/ws/install/setup.bash
exec "$@"
