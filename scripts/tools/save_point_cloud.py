import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import ros_numpy  # Import ros_numpy for easier handling of PointCloud2


def pointcloud_callback(msg):
    # Convert the PointCloud2 message to a NumPy structured array
    pc = ros_numpy.numpify(msg)
    pc = ros_numpy.point_cloud2.split_rgb_field(pc)  # Split the RGB field into separate channels

    # Extract points and flatten them
    points = np.zeros((pc.shape[0]*pc.shape[1], 3))
    points[:, 0] = pc['x'].flatten()  # Flatten to ensure it's 1D
    points[:, 1] = pc['y'].flatten()  # Flatten to ensure it's 1D
    points[:, 2] = pc['z'].flatten()  # Flatten to ensure it's 1D

    # Extract RGB values and normalize them
    rgb = np.zeros((pc.shape[0]*pc.shape[1], 3))
    rgb[:, 0] = pc['r'].flatten() / 255.0  # Red channel normalized
    rgb[:, 1] = pc['g'].flatten() / 255.0  # Green channel normalized
    rgb[:, 2] = pc['b'].flatten() / 255.0  # Blue channel normalized

    # Print unique colors
    unique_colors = set(tuple(color) for color in rgb)  # Use tuples for uniqueness
    print("Unique Colors in the Point Cloud:")
    for color in unique_colors:
        print(color)

    # Create an Open3D point cloud with points and colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Save the point cloud with colors to a PCD file
    o3d.io.write_point_cloud("semantic_pointcloud_3.pcd", pcd)
    rospy.loginfo("Semantically segmented point cloud saved as semantic_pointcloud.pcd")
    rospy.signal_shutdown("Point cloud saved. Shutting down.")

def main():
    rospy.init_node("pointcloud_saver")
    rospy.Subscriber("/semantic_pointcloud", PointCloud2, pointcloud_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
