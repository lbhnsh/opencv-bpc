import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from scipy.spatial.transform import Rotation as R

def load_point_cloud(file_path):
    """Loads a point cloud from a file."""
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def normalize_scale(source, target):
    """Scales the source to match the target's bounding box dimensions."""
    source_bbox = source.get_axis_aligned_bounding_box()
    target_bbox = target.get_axis_aligned_bounding_box()
    
    source_extent = source_bbox.get_extent()
    target_extent = target_bbox.get_extent()
    
    scale_factors = target_extent / source_extent  # Scaling per axis
    uniform_scale = np.min(scale_factors)  # Use the minimum scale factor

    source.points = o3d.utility.Vector3dVector(np.asarray(source.points) * uniform_scale)
    return source, uniform_scale

def align_with_pca(pcd):
    """Aligns the point cloud with its principal axes using PCA."""
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # Compute PCA
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # Align to principal axes
    aligned_points = points_centered @ eigvecs
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(aligned_points)
    
    return pcd_aligned, eigvecs, centroid

def icp_registration(source, target):
    """Performs ICP registration."""
    icp = o3d.pipelines.registration.registration_icp(
        source, target, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return icp.transformation

def extract_rpy(transformation):
    """Extracts roll, pitch, yaw from a transformation matrix."""
    rotation_matrix = transformation[:3, :3]
    rpy = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
    return rpy

def plot_point_clouds(model, object_transformed, model_bbox, object_bbox):
    """Visualizes the model, object, and their bounding boxes."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy arrays
    model_pts = np.asarray(model.points)
    object_pts = np.asarray(object_transformed.points)

    # Plot point clouds
    ax.scatter(model_pts[:, 0], model_pts[:, 1], model_pts[:, 2], c='b', marker='o', s=1, label='Model')
    ax.scatter(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2], c='r', marker='o', s=1, label='Object')

    # Plot bounding boxes
    def plot_bbox(bbox, color):
        lines = [[0,1],[1,3],[3,2],[2,0], [4,5],[5,7],[7,6],[6,4], [0,4],[1,5],[3,7],[2,6]]
        corners = np.asarray(bbox.get_box_points())
        for line in lines:
            ax.plot3D(*zip(*corners[line]), c=color)

    plot_bbox(model_bbox, 'blue')
    plot_bbox(object_bbox, 'red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

### MAIN EXECUTION ###
# Load point clouds
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

# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
object_pcd = o3d.io.read_point_cloud('pcd_object.ply')
model_pcd = o3d.io.read_point_cloud('pcd_model.ply')
# object_pcd = pcd.voxel_down_sample(voxel_size=0.005)
# object_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

# mesh = o3d.io.read_triangle_mesh("obj_000018.ply")
# model_pcd = mesh.sample_points_uniformly(number_of_points=1000)
# model_pcd = model_pcd.voxel_down_sample(voxel_size=0.005)
# model_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))


# Normalize scale
# object_pcd, scale_factor = normalize_scale(object_pcd, model_pcd)

# PCA alignment
model_pcd_aligned, model_eigvecs, model_centroid = align_with_pca(model_pcd)
object_pcd_aligned, object_eigvecs, object_centroid = align_with_pca(object_pcd)

# Transform object to match model's PCA orientation
object_pcd_transformed = o3d.geometry.PointCloud()
object_pcd_transformed.points = o3d.utility.Vector3dVector(
    (np.asarray(object_pcd_aligned.points) @ model_eigvecs.T) + model_centroid
)

# ICP fine-tuning
transformation = icp_registration(object_pcd_transformed, model_pcd_aligned)

# Apply transformation
object_pcd_final = object_pcd_transformed.transform(transformation)

# Extract RPY angles
rpy_angles = extract_rpy(transformation)
print(f"Estimated Roll, Pitch, Yaw (degrees): {rpy_angles}")

# Compute bounding boxes
model_bbox = model_pcd.get_axis_aligned_bounding_box()
object_bbox = object_pcd_final.get_axis_aligned_bounding_box()

# Visualization
plot_point_clouds(model_pcd, object_pcd_final, model_bbox, object_bbox)
