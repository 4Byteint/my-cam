#!/bin/bash
docker run -it --rm \
  --net=host \
  -v ~/Documents/my-cam/auto_project/ROS:/workspace \
  ros:humble-ros-base


# if you want to start the container, "bash start_ros_container.sh"
