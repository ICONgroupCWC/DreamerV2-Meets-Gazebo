#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()
windows = {
    "front": "/cam_front/world_cam/image_raw",
    "back":  "/cam_back/world_cam/image_raw",
    "left":  "/cam_left/world_cam/image_raw",
    "right": "/cam_right/world_cam/image_raw",
    "top":   "/cam_top/world_cam/image_raw",
}

def make_callback(name):
    def callback(msg):
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow(name, img)
        cv2.waitKey(1)
    return callback

rospy.init_node("multi_camera_viewer")

for name, topic in windows.items():
    rospy.Subscriber(topic, Image, make_callback(name))

rospy.spin()
