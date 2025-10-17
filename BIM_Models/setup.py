from setuptools import setup
import os
from glob import glob

package_name = 'BIM_Models'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files if you have any
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include config files if you have any
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maintainer',
    maintainer_email='you@example.com',
    description='Python utilities and scripts for BIM models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Add your nodes here in format:
            # 'node_executable_name = package_name.script_name:main'
            # Example:
            'ros_vlm_node = BIM_Models.ros2_vlm:main',
        ],
    },
)