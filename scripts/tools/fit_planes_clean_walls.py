import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

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

# ---------------------------
# Apply Statistical Outlier Removal
# ---------------------------
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = pcd.select_by_index(ind)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

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
    floor_plane_model, floor_inliers = floor_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    print("Floor Plane Model:", floor_plane_model)
else:
    floor_plane_model = None

if len(ceiling_points) > 0:
    ceiling_plane_model, ceiling_inliers = ceiling_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    print("Ceiling Plane Model:", ceiling_plane_model)
else:
    ceiling_plane_model = None

# Function to project points onto a plane defined by ax + by + cz + d = 0
def project_points_onto_plane(points, plane_model):
    if plane_model is not None:
        a, b, c, d = plane_model
        projected_points = []
        for point in points:
            # Calculate the projection of the point onto the plane
            distance = (a * point[0] + b * point[1] + c * point[2] + d) / np.sqrt(a**2 + b**2 + c**2)
            projected_point = point - distance * np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            projected_points.append(projected_point)
        return np.array(projected_points)
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

# Create the final point cloud with the remaining points and projected floor/ceiling points
final_combined_points = np.vstack((remaining_points, projected_floor_points, projected_ceiling_points))
final_combined_colors = np.vstack((remaining_colors, filtered_colors[floor_mask], filtered_colors[ceiling_mask]))

# Create the final point cloud
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(final_combined_points)
final_pcd.colors = o3d.utility.Vector3dVector(final_combined_colors)

# # ---------------------------
# # Wall Filtering Procedure
# # ---------------------------

# # Create a mask for wall points based on their semantic labels
# wall_color = np.array([0, 102, 51])  # Wall color
# wall_mask = np.all(colors * 255 == wall_color, axis=1)  # Adjust based on how the wall points are stored

# # Extract wall points
# wall_points = points[wall_mask]

# # Apply DBSCAN clustering for wall points
# dbscan = DBSCAN(eps=0.1, min_samples=100)  # Adjust parameters as needed
# labels = dbscan.fit_predict(wall_points)

# # Filter clusters based on size
# unique_labels, counts = np.unique(labels, return_counts=True)
# valid_clusters = unique_labels[counts > 50]  # Filter out small clusters

# # Collect points from valid clusters
# filtered_wall_points = wall_points[np.isin(labels, valid_clusters)]

# # Create a new point cloud for the filtered wall points
# filtered_wall_pcd = o3d.geometry.PointCloud()
# filtered_wall_pcd.points = o3d.utility.Vector3dVector(filtered_wall_points)

# # Keep the original color for the filtered walls
# filtered_wall_colors = colors[wall_mask][np.isin(labels, valid_clusters)]
# filtered_wall_pcd.colors = o3d.utility.Vector3dVector(filtered_wall_colors)

# # Add the filtered wall points to the final combined point cloud
# final_combined_points = np.vstack((final_combined_points, np.asarray(filtered_wall_pcd.points)))
# final_combined_colors = np.vstack((final_combined_colors, np.asarray(filtered_wall_pcd.colors)))

# # Create the final point cloud with walls included
# final_pcd = o3d.geometry.PointCloud()
# final_pcd.points = o3d.utility.Vector3dVector(final_combined_points)
# final_pcd.colors = o3d.utility.Vector3dVector(final_combined_colors)
o3d.io.write_point_cloud("semantic_pointcloud_cleaned.pcd", final_pcd)

# # Visualize the final combined point cloud
o3d.visualization.draw_geometries(
    [final_pcd],
    window_name="Final Combined Point Cloud with Projected Ceiling, Floor, and Filtered Walls"
)
