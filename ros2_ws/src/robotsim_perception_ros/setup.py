from glob import glob

from setuptools import setup

package_name = "robotsim_perception_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.rviz")),
    ],
    install_requires=["setuptools"],
    # colcon 은 test 의존성에 pytest 가 있어야 pytest 러너를 고른다 (없으면 unittest -> 0 tests).
    # setuptools 72+ 에서 tests_require 가 제거됐으므로 extras_require['test'] 를 쓴다.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Gwangyeol Kim",
    maintainer_email="aivdlrx@gmail.com",
    description="ROS2 wrapper for robotsim_perception (ToF box detection -> pick poses in base_link)",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "perception_node = robotsim_perception_ros.perception_node:main",
        ],
    },
)
