import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Bounding box information from the JSON
bbox_obj = [1816.0, 1565.0, 286.0, 253.0]  # xmin, ymin, width, height

# Read color and depth images
color_raw = o3d.io.read_image("C:/Users/ACER/Downloads/000000rgb.png")
depth_raw = o3d.io.read_image("C:/Users/ACER/Downloads/000000d.png")

# Convert to numpy arrays for easier cropping
color_array = np.asarray(color_raw)
depth_array = np.asarray(depth_raw)


# Get the bounding box coordinates
xmin, ymin, width, height = bbox_obj
xmax, ymax = xmin + width, ymin + height

# Crop the images based on the bounding box
cropped_color = color_array[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
cropped_depth = depth_array[int(ymin):int(ymax), int(xmin):int(xmax)].copy()



# Convert the cropped depth array to float32 and apply depth scale (if necessary)
cropped_depth = cropped_depth.astype(np.float32) * 0.1  # Apply depth scale

# Convert the cropped images back to Open3D images (ensure they are contiguous)
cropped_color_raw = o3d.geometry.Image(cropped_color)
cropped_depth_raw = o3d.geometry.Image(cropped_depth)

# Create the RGBD image from the cropped images
rgbd_image_cropped = o3d.geometry.RGBDImage.create_from_color_and_depth(cropped_color_raw, cropped_depth_raw)

# Set camera intrinsics from the JSON data
intrinsic_matrix = np.array([
    [3981.985991142684, 0.0, 1954.1872863769531],
    [0.0, 3981.985991142684, 1103.6978149414062],
    [0.0, 0.0, 1.0]
]) 


intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.intrinsic_matrix = intrinsic_matrix

# Create the point cloud from the cropped RGBD image
pcd_cropped = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_cropped, intrinsics)

# Rotation matrix and translation vector (same as before)
R = np.array([
    [0.9887484396846155, 0.1492973613521459, 0.00931779267969649],
    [-0.06795137741673805, 0.3927819263944453, 0.9171177509492592],
    [0.13326339926949368, -0.9074319022480067, 0.3985074785433647]
])
t = np.array([-337.4470492221228, 587.8991738764912, -10.381216800868486])

# Apply rotation and translation to the cropped point cloud
pcd_cropped.transform(np.vstack([np.hstack([R, t.reshape(-1, 1)]), [0, 0, 0, 1]]))

# Visualize the cropped point cloud
o3d.visualization.draw_geometries([pcd_cropped])
o3d.io.write_point_cloud("cropped_point_cloud.ply", pcd_cropped)


# import open3d as o3d
# import numpy as np

# # Load the point cloud (cropped point cloud from previous steps)
# # pcd_cropped = o3d.io.read_point_cloud("path_to_cropped_point_cloud.ply")  # replace with your point cloud file

# # Load the reference model (PLY model)
# model = o3d.io.read_triangle_mesh("C:/Users/ACER/Downloads/obj_000018.ply")  # replace with your model file

# # Convert the model to point cloud for ICP (if it's a mesh)
# model_pcd = model.sample_points_poisson_disk(number_of_points=5000)  # downsample the mesh to create a point cloud

# # **Step 1: Downsample the source point cloud (to a larger size for better feature matching)**
# # Increase the number of points sampled to match or be closer to the model's point count
# pcd_cropped = pcd_cropped.voxel_down_sample(voxel_size=0.02)  # Downsample the point cloud to match the model size
# pcd_cropped = pcd_cropped.uniform_down_sample(every_k_points=10)  # Increase the number of points if necessary

# # Preprocess point clouds (estimate normals and downsample)
# pcd_cropped.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# model_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# # Compute FPFH features for the point cloud and the model
# radius_normal = 0.1  # adjust depending on your point cloud density
# radius_feature = 0.25  # adjust based on the scale of your point cloud/model

# # Compute features for the cropped point cloud
# pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_cropped, 
#                                                            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# # Compute features for the model point cloud
# model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(model_pcd, 
#                                                              o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# # Perform global feature matching to find correspondences (RANSAC)
# result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#     pcd_cropped, model_pcd, pcd_fpfh, model_fpfh, 0.05,  # 0.05 is the maximum correspondence distance threshold
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),  # Point-to-Point error metric
#     4,  # number of iterations (you can increase it for better results)
#     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],  # max distance for correspondences
#     o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)  # max iterations and max correspondence distance
# )

# # Extract the initial transformation matrix
# initial_transformation = result.transformation
# print("Initial Transformation Matrix (Translation + Rotation):")
# print(initial_transformation)

# # **Step 2: Refine the alignment with ICP**
# threshold = 0.02  # ICP threshold (tune based on your model scale)
# icp_result = o3d.pipelines.registration.registration_icp(
#     pcd_cropped, model_pcd, threshold, initial_transformation,  # Using the initial transformation from RANSAC
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
# )

# # Print the final transformation matrix (6DOF pose)
# final_transformation = icp_result.transformation
# print("Final Transformation Matrix (6DOF Pose):")
# print(final_transformation)

# # Extract translation (X, Y, Z)
# translation_vector = final_transformation[:3, 3]
# x, y, z = translation_vector
# print(f"Translation: X={x}, Y={y}, Z={z}")

# # Extract rotation matrix and convert to Euler angles (Roll, Pitch, Yaw)
# rotation_matrix = final_transformation[:3, :3]

# def rotation_matrix_to_euler_angles(R):
#     assert(R.shape == (3, 3))
#     sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
#     singular = sy < 1e-6

#     if not singular:
#         x = np.atan2(R[2, 1], R[2, 2])
#         y = np.atan2(-R[2, 0], sy)
#         z = np.atan2(R[1, 0], R[0, 0])
#     else:
#         x = np.atan2(-R[1, 2], R[1, 1])
#         y = np.atan2(-R[2, 0], sy)
#         z = 0

#     return x, y, z

# roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
# print(f"Rotation: Roll={np.degrees(roll)}°, Pitch={np.degrees(pitch)}°, Yaw={np.degrees(yaw)}°")

# # **Step 3: Visualize the result**
# # Apply the final transformation to the cropped point cloud
# pcd_cropped.transform(final_transformation)

# # Visualize the original model and aligned point cloud
# o3d.visualization.draw_geometries([model_pcd, pcd_cropped])
