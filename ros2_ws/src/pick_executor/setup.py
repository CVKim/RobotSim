from glob import glob

from setuptools import setup

package_name = "pick_executor"

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
    extras_require={"test": ["pytest"]},      # colcon 이 pytest 러너를 고르게 (robotsim_perception_ros 와 동일한 이유)
    zip_safe=True,
    maintainer="Gwangyeol Kim",
    maintainer_email="aivdlrx@gmail.com",
    description="pick poses -> 3-point robot trajectory, with perception re-capture loop",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "pick_executor = pick_executor.pick_executor_node:main",
        ],
    },
)
