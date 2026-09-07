from glob import glob

from setuptools import setup

package_name = "twin_bridge"

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
    extras_require={"test": ["pytest"]},      # colcon 이 pytest 러너를 고르게
    zip_safe=True,
    maintainer="Gwangyeol Kim",
    maintainer_email="aivdlrx@gmail.com",
    description="Bridges /robot/target_poses to the MuJoCo cell twin (tools/twin_server.py) and reports execution results",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "twin_bridge = twin_bridge.bridge_node:main",
        ],
    },
)
