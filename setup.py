from setuptools import find_packages, setup

package_name = 'mlp_astar_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/mlp_astar_planner']),
        ('share/mlp_astar_planner', ['package.xml']),
        ('share/mlp_astar_planner/rviz', ['rviz/mlp_astar.rviz']),
        ('share/mlp_astar_planner/launch', ['launch/mlp_astar.launch.py']),
        ('share/mlp_astar_planner/maps', [
            'maps/map.yaml',
            'maps/map.pgm',
            'maps/map_small.yaml',
            'maps/map_small.pgm',
            'maps/my_map.yaml',
            'maps/my_map.pgm',
        ]),
        ('share/mlp_astar_planner/launch', [
            'launch/mlp_astar.launch.py',
            'launch/astar_comparision.launch.py',
        ]),
        ('share/mlp_astar_planner/rviz', [
            'rviz/mlp_astar.rviz',
            'rviz/astar_comparision.rviz',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artem',
    maintainer_email='rtmkrk@gmail.com',
    description='TODO: Package description',
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
        ],
    },
)
