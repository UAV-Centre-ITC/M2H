import open3d as o3d
import numpy as np

# Load the dumped PCD file
pcd = o3d.io.read_point_cloud("semantic_pointcloud_1.pcd")

# Check if the point cloud has colors
if pcd.has_colors():
    print("Point cloud has colors.")
    
    # Get the colors from the point cloud
    colors = np.asarray(pcd.colors)
    
else:
    print("Point cloud does not have colors.")

# Visualize the point cloud with its semantic colors
o3d.visualization.draw_geometries([pcd], window_name="Dumped Semantically Segmented Point Cloud")
