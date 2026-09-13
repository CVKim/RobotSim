cmake_minimum_required(VERSION 3.8)
project(robotsim_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/plan_compare.py DESTINATION lib/${PROJECT_NAME})

ament_package()
