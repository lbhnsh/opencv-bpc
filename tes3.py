import open3d as o3d
import numpy as np

# Load the RGBD point cloud (from the RGBD image) - PCD
pcd_scene = o3d.io.read_point_cloud("cropped_point_cloud.ply")

# Load the 3D object model (PLY file)
object_model = o3d.io.read_triangle_mesh("C:/Users/ACER/Downloads/obj_000018.ply")
object_model = object_model.sample_points_poisson_disk(number_of_points=10000)  # Sample points for the model

# Visualize the scene point cloud and the object model before any transformation
print("Visualizing original scene and object model before transformation...")
o3d.visualization.draw_geometries([pcd_scene, object_model])

# Preprocess the object model and scene (downsampling for faster processing)
voxel_size = 0.005  # Adjust based on your data for better performance
pcd_scene_downsampled = pcd_scene
object_model_downsampled = object_model.voxel_down_sample(voxel_size)

# Step 1: Center both point clouds by subtracting their centroids
scene_centroid = np.mean(np.asarray(pcd_scene_downsampled.points), axis=0)
pcd_scene_downsampled.translate(-scene_centroid)

model_centroid = np.mean(np.asarray(object_model_downsampled.points), axis=0)
object_model_downsampled.translate(-model_centroid)

# Visualize the scene and object model after centering
print("Visualizing scene and object model after centering...")
o3d.visualization.draw_geometries([pcd_scene_downsampled, object_model_downsampled])

# Step 2: Compute the best rotation using SVD
# Use the points from the scene and the object model (after centering)
scene_points = np.asarray(pcd_scene_downsampled.points)
model_points = np.asarray(object_model_downsampled.points)

# Compute the covariance matrix
H = np.dot(model_points.T, scene_points)

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(H)

# Compute the optimal rotation matrix
rotation_matrix = np.dot(U, Vt)

# Ensure a proper rotation (determinant check for right-handed coordinate system)
if np.linalg.det(rotation_matrix) < 0:
    Vt[-1, :] *= -1
    rotation_matrix = np.dot(U, Vt)

# Apply the computed rotation matrix to the object model
object_model_downsampled.rotate(rotation_matrix, center=(0, 0, 0))

# Visualize after applying the rotation
print("Visualizing scene and rotated object model...")
o3d.visualization.draw_geometries([pcd_scene_downsampled, object_model_downsampled])

# Step 3: Perform translation alignment (if necessary)
# Since the rotation is already aligned, the only thing left to adjust is translation
translation_vector = scene_centroid - np.mean(np.asarray(object_model_downsampled.points), axis=0)
object_model_downsampled.translate(translation_vector)

# Visualize the final aligned object model with the scene
print("Visualizing scene and object model after alignment (rotation and translation)...")
o3d.visualization.draw_geometries([pcd_scene_downsampled, object_model_downsampled])

# Optionally, you can save the aligned object model:
# o3d.io.write_point_cloud("aligned_object_model.ply", object_model_downsampled)
