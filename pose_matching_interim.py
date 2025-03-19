import open3d as o3d
import numpy as np
import cv2 

# Reading images: RGB, Depth, Mask
color = cv2.imread("000000.jpg")[:,:,::-1]
# color = np.rot90(color)
depth = cv2.imread("000000.png", cv2.IMREAD_UNCHANGED) 
# depth = np.rot90(depth)
depth = depth.astype(np.float32) * 0.1 # depth scale : 0.1
mask = cv2.imread("000000_000004.png", 0)
# mask = np.rot90(mask)

color = cv2.bitwise_and(color, color, mask=mask)
depth = depth.copy()
depth[mask == 0] = 0  # set background depth to 0

color = o3d.geometry.Image(np.ascontiguousarray(color))
depth = o3d.geometry.Image(np.ascontiguousarray(depth))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

# Create the PinholeCameraIntrinsic object
intrinsic_matrix = np.array([[4050.419714657658, 0.0, 1200.0], [0.0, 4050.419714657658, 1200.0], [0.0, 0.0, 1.0]]) # cam k : intrinsic matrix 
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.intrinsic_matrix = intrinsic_matrix

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

pcd_object = pcd.voxel_down_sample(voxel_size=0.005)
pcd_object.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

mesh = o3d.io.read_triangle_mesh("obj_000018.ply")
pcd_model = mesh.sample_points_uniformly(number_of_points=1000)
pcd_model = pcd_model.voxel_down_sample(voxel_size=0.005)
pcd_model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# o3d.io.write_point_cloud("point_cloud.pcd", pcd_object + pcd_model)
o3d.visualization.draw_geometries([pcd_object, pcd_model])

bbox_object = pcd_object.get_axis_aligned_bounding_box()
bbox_model = pcd_model.get_axis_aligned_bounding_box()
bbox_object.color = (1, 0, 0)
bbox_model.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd_object, bbox_object, pcd_model, bbox_model])

extent_object = np.array(bbox_object.get_extent())
extent_model = np.array(bbox_model.get_extent())

scale_factor = np.mean(extent_object) / np.mean(extent_model)
print("Scale Factor:", scale_factor)

pcd_model.scale(scale_factor, center=pcd_model.get_center())

centroid_object = pcd_object.get_center()
centroid_model = pcd_model.get_center()
print("Object centroid:", centroid_object)
print("Model centroid:", centroid_model)

translation_vector = centroid_object - centroid_model
print("Translation Vector:", translation_vector)

pcd_model.translate(translation_vector)

bbox_object = pcd_object.get_axis_aligned_bounding_box()
bbox_model = pcd_model.get_axis_aligned_bounding_box()
bbox_object.color = (1, 0, 0)
bbox_model.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd_object, bbox_object, pcd_model, bbox_model])

'''
BELOW CODE IS NOT OF USE FOR NOW
'''

# radius_feature = 0.0001
# fpfh_object = o3d.pipelines.registration.compute_fpfh_feature(
#     pcd_object,
#     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
# fpfh_model = o3d.pipelines.registration.compute_fpfh_feature(
#     pcd_model,
#     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# distance_threshold = 0.02  # adjust based on scale
# result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#     pcd_object, pcd_model, fpfh_object, fpfh_model,
#     True,                      # mutual_filter flag
#     distance_threshold,        # max correspondence distance
#     o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#     4,                         # ransac_n, number of points sampled per iteration
#     [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#     o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
# )


# print("Global Registration Transformation:")
# print(result_global.transformation)

result_icp = o3d.pipelines.registration.registration_icp(
    pcd_object, pcd_model, 0.1, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane())

print("Final 6DoF Pose (Transformation Matrix):")
print(result_icp.transformation)

pcd_model_transformed = pcd_model.transform(result_icp.transformation)
o3d.visualization.draw_geometries([pcd_object, pcd_model_transformed])
# o3d.visualization.draw_geometries([pcd])

import open3d as o3d
import numpy as np
import math

# Function to convert a rotation matrix to Euler angles (roll, pitch, yaw)
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0
    return np.degrees([roll, pitch, yaw])

# Assume 'result_icp.transformation' is the final transformation matrix from your ICP registration.
# For demonstration, here's a dummy transformation matrix.
# In practice, replace this with your 'result_icp.transformation'
transformation = np.array([[ 0.99, -0.01,  0.02, 0.1],
                           [ 0.01,  0.99, -0.03, 0.2],
                           [-0.02,  0.03,  0.99, 0.3],
                           [ 0.0,   0.0,   0.0,  1.0]])

# Extract translation and rotation from the transformation
translation = transformation[:3, 3]
rotation = transformation[:3, :3]
roll, pitch, yaw = rotationMatrixToEulerAngles(rotation)

print("Translation (x, y, z):", translation)
print("Rotation (roll, pitch, yaw in degrees):", roll, pitch, yaw)

# Create a coordinate frame representing the object's pose
object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
object_frame.transform(transformation)

# Create a world coordinate frame at the origin for reference
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)

# Optionally, load or create your point clouds (e.g., object point cloud and reference model)
# For demonstration, we'll just create an empty list and show the frames.
geometries = [world_frame, object_frame, pcd_object]
# If you have your point clouds (pcd_object, pcd_model_transformed), add them to the list:
# geometries.extend([pcd_object, pcd_model_transformed])

# Visualize the result with annotations for x, y, z axes.
o3d.visualization.draw_geometries(geometries)
