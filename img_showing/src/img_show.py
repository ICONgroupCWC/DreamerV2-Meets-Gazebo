#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np

def callback(msg):
    # Convert ROS CompressedImage message to OpenCV image
    np_arr = np.frombuffer(msg.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Show image
    cv2.imshow("Result Image", image)
    cv2.waitKey(1)  # Required to keep window responsive

def main():
    rospy.init_node('result_img_viewer', anonymous=True)
    rospy.Subscriber("/result_img", CompressedImage, callback)
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
