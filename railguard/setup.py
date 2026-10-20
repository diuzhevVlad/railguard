from glob import glob

from setuptools import setup

package_name = 'railguard'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'kiss-icp>=1.3,<2', 'typer[all]>=0.6,<0.10'],
    zip_safe=True,
    maintainer='RailGuard team',
    maintainer_email='vladdiuzhev@gmail.com',
    description='RailGuard: track centerline to 200 m from a single metro-tunnel LiDAR scan.',
    license='Proprietary',
    entry_points={'console_scripts': ['railguard_node = railguard.node:main',
                                        'railguard_odometry = railguard.odometry_node:main']},
)
