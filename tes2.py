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

# Preprocess the object model and scene (downsampling for faster registration)
voxel_size = 0.005  # Adjust based on your data for better performance
# pcd_scene_downsampled = pcd_scene.voxel_down_sample(voxel_size)
pcd_scene_downsampled = pcd_scene
object_model_downsampled = object_model.voxel_down_sample(voxel_size)

# Step 1: Rescale and Center the Point Clouds to a common reference frame

# Center the scene point cloud by subtracting the centroid
scene_centroid = np.mean(np.asarray(pcd_scene_downsampled.points), axis=0)
pcd_scene_downsampled.translate(-scene_centroid)

# Center the object model by subtracting the centroid
model_centroid = np.mean(np.asarray(object_model_downsampled.points), axis=0)
object_model_downsampled.translate(-model_centroid)

# Visualize the scene and object model after centering
print("Visualizing scene and object model after centering...")
o3d.visualization.draw_geometries([pcd_scene_downsampled, object_model_downsampled])

# Rescale the object model if needed (optional)
# Example: Rescale the object model to the scale of the scene if the model is too large or small
# For instance, if the object model appears too large, you can scale it down.
# You can estimate this scaling factor based on the size of the object and the scene
model_bounding_box = object_model_downsampled.get_axis_aligned_bounding_box()
scene_bounding_box = pcd_scene_downsampled.get_axis_aligned_bounding_box()

# For example, use the diagonal of bounding boxes as the scaling factor (simple heuristic)
model_size = np.linalg.norm(model_bounding_box.get_max_bound() - model_bounding_box.get_min_bound())
scene_size = np.linalg.norm(scene_bounding_box.get_max_bound() - scene_bounding_box.get_min_bound())

scaling_factor = scene_size / model_size
object_model_downsampled.scale(scaling_factor, center=(0, 0, 0))

# Visualize the rescaled object model and the scene again
print("Visualizing scene and object model after rescaling...")
o3d.visualization.draw_geometries([pcd_scene_downsampled, object_model_downsampled])

# Step 2: Perform ICP (Iterative Closest Point) for fine alignment
threshold = 99.9  # Maximum distance for a point to be considered part of the same object
icp_result = o3d.pipelines.registration.registration_icp(
    object_model_downsampled, pcd_scene_downsampled, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Step 3: Visualize the result (aligned model with the scene)
print("ICP RMSE (Inlier RMS error):", icp_result.inlier_rmse)
print("Transformation matrix:\n", icp_result.transformation)

# Apply the final transformation to the object model
object_model_aligned_final = object_model.transform(icp_result.transformation)

# Visualize the final result (aligned model with the scene)
print("Visualizing scene and object model after ICP transformation...")
o3d.visualization.draw_geometries([pcd_scene_downsampled ])
o3d.visualization.draw_geometries([pcd_scene_downsampled , object_model_aligned_final])


# Step 4: Extract the 6DoF pose (translation and rotation)
translation = icp_result.transformation[:3, 3]  # Extract translation vector
rotation = icp_result.transformation[:3, :3]    # Extract rotation matrix

print("Estimated Translation (Position):", translation)
print("Estimated Rotation (Matrix):\n", rotation)
