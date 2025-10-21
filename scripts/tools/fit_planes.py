import open3d as o3d
import numpy as np

# Load the dumped PCD file
pcd = o3d.io.read_point_cloud("semantic_pointcloud_1.pcd")

# Check if the point cloud has colors
if pcd.has_colors():
    print("Point cloud has colors.")
    # Get the points and colors from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
else:
    print("Point cloud does not have colors.")

# Define height range (2m to 8m)
lower_bound = 2.0  # 2 meters
upper_bound = 8.0  # 8 meters

# Create a mask to filter points within the specified height range
mask = (points[:, 2] >= lower_bound) & (points[:, 2] <= upper_bound)

# Extract filtered points
filtered_points = points[mask]
filtered_colors = colors[mask] if pcd.has_colors() else None

# Define semantic class colors for filtering
floor_color = np.array([191, 191, 181])  # Floor color
ceiling_color = np.array([59, 163, 236])  # Ceiling color

# Create masks for the floor and ceiling based on their semantic labels
floor_mask = np.all(filtered_colors * 255 == floor_color, axis=1)
ceiling_mask = np.all(filtered_colors * 255 == ceiling_color, axis=1)

# Extract filtered points for floor and ceiling
floor_points = filtered_points[floor_mask]
ceiling_points = filtered_points[ceiling_mask]

# Create point clouds for the floor and ceiling
floor_pcd = o3d.geometry.PointCloud()
floor_pcd.points = o3d.utility.Vector3dVector(floor_points)

ceiling_pcd = o3d.geometry.PointCloud()
ceiling_pcd.points = o3d.utility.Vector3dVector(ceiling_points)

# Fit planes for floor and ceiling using RANSAC
if len(floor_points) > 0:
    floor_plane_model, floor_inliers = floor_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=100)
    print("Floor Plane Model:", floor_plane_model)
else:
    floor_plane_model = None

if len(ceiling_points) > 0:
    ceiling_plane_model, ceiling_inliers = ceiling_pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=100)
    print("Ceiling Plane Model:", ceiling_plane_model)
else:
    ceiling_plane_model = None

# Function to project points onto a plane defined by ax + by + cz + d = 0
def project_points_onto_plane(points, plane_model):
    if plane_model is not None:
        a, b, c, d = plane_model
        distance = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        projected_points = points - distance[:, np.newaxis] * np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
        return projected_points
    return points  # Return original points if no model is available

# Project ceiling and floor points onto their planes
projected_floor_points = project_points_onto_plane(floor_points, floor_plane_model)
projected_ceiling_points = project_points_onto_plane(ceiling_points, ceiling_plane_model)

# Create new point clouds for the projected points
projected_floor_pcd = o3d.geometry.PointCloud()
projected_floor_pcd.points = o3d.utility.Vector3dVector(projected_floor_points)
# Keep original color for the floor
projected_floor_pcd.colors = o3d.utility.Vector3dVector(np.tile(filtered_colors[floor_mask], (1, 1)))

projected_ceiling_pcd = o3d.geometry.PointCloud()
projected_ceiling_pcd.points = o3d.utility.Vector3dVector(projected_ceiling_points)
# Keep original color for the ceiling
projected_ceiling_pcd.colors = o3d.utility.Vector3dVector(np.tile(filtered_colors[ceiling_mask], (1, 1)))

# Combine the original filtered points with the projected floor and ceiling points,
# but only keep the points that are not part of the original ceiling and floor points
remaining_points = filtered_points[~(floor_mask | ceiling_mask)]  # Remaining points excluding floor and ceiling
remaining_colors = filtered_colors[~(floor_mask | ceiling_mask)]

# Create a final point cloud with the remaining points and projected floor/ceiling points
final_combined_points = np.vstack((remaining_points, projected_floor_points, projected_ceiling_points))
final_combined_colors = np.vstack((remaining_colors, filtered_colors[floor_mask], filtered_colors[ceiling_mask]))

# Create the final point cloud
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(final_combined_points)
final_pcd.colors = o3d.utility.Vector3dVector(final_combined_colors)

# Visualize the final combined point cloud
o3d.visualization.draw_geometries(
    [final_pcd],
    window_name="Final Combined Point Cloud with Projected Ceiling and Floor"
)
