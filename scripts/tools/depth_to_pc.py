#!/usr/bin/env python

import rospy
import ros_numpy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge

def save_pointcloud_as_pcd(points, filename="output_pointcloud.pcd"):
    """
    Save a point cloud to a .pcd file
    """
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
    with open(filename, "w") as f:
        f.write(header)
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    rospy.loginfo(f"Point cloud saved to {filename}")


def depth_to_pointcloud(depth_image, fx, fy, cx, cy):
    """
    Convert depth image to a point cloud
    """
    height, width = depth_image.shape
    points = []
    for v in range(height):
        for u in range(width):
            z = depth_image[v, u]
            if z == 0 or z != z:  # Ignore invalid depth values (0 or NaN)
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
    return points


def process_single_depth_message():
    rospy.init_node("single_depth_to_pointcloud", anonymous=True)
    bridge = CvBridge()

    # Wait for a single depth image message
    rospy.loginfo("Waiting for a depth image message...")
    depth_msg = rospy.wait_for_message("/camera/depth_cam/image_raw", Image)
    rospy.loginfo("Received depth image message.")

    # Wait for a single camera info message
    rospy.loginfo("Waiting for camera info message...")
    cam_info = rospy.wait_for_message("/zed/zed_node/rgb/rect/camera_info", CameraInfo)
    rospy.loginfo("Received camera info message.")

    # Extract camera intrinsics
    fx = cam_info.K[0]
    fy = cam_info.K[4]
    cx = cam_info.K[2]
    cy = cam_info.K[5]

    # Convert depth image to numpy array
    depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    # Convert depth to point cloud
    points = depth_to_pointcloud(depth_image, fx, fy, cx, cy)

    # Save point cloud as .pcd
    save_pointcloud_as_pcd(points, filename="output_pointcloud.pcd")

    rospy.loginfo("Point cloud generation complete. Exiting.")


if __name__ == "__main__":
    try:
        process_single_depth_message()
    except rospy.ROSInterruptException:
        pass
