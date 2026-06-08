import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mlp_astar_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        (
            os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py')),
        ),
        (
            os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz')),
        ),
        (
            os.path.join('share', package_name, 'maps'),
            glob(os.path.join('maps', '*')),
        ),
        (
            os.path.join('share', package_name, 'data'),
            glob(os.path.join('data', '*')),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artem',
    maintainer_email='rtmkrk@gmail.com',
    description='MLP-assisted A* planner for ROS2',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'points = mlp_astar_planner.points:main',
            'generate_dataset = mlp_astar_planner.dataset_generator:main',
            'train_mlp = mlp_astar_planner.train_mlp:main',
            'astar_mlp = mlp_astar_planner.astar_mlp:main',
            'astar_classic = mlp_astar_planner.astar_classic:main',
            'astar_hybrid_refine = mlp_astar_planner.astar_hybrid_refine:main',
            'mlp_gradient_only = mlp_astar_planner.mlp_gradient_only:main',
            'benchmark_methods = mlp_astar_planner.benchmark_methods:main',
        ],
    },
)