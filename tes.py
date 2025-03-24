import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Read color image
color_raw = o3d.io.read_image("C:/Users/ACER/Downloads/000000rgb.png")

# Read depth image and apply depth scale (convert depth to meters)
depth_raw = o3d.io.read_image("C:/Users/ACER/Downloads/000000d.png")
depth_raw = np.array(depth_raw) * 0.1  # Apply depth scale to convert to meters

# Ensure depth image is of type float32 (or uint16, depending on your depth range)
depth_raw = depth_raw.astype(np.float32)  # Convert to float32

# Create depth image for Open3D
depth_raw = o3d.geometry.Image(depth_raw)

# Create RGBD image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

# Set camera intrinsics from the JSON data
intrinsic_matrix = np.array([
    [3981.985991142684, 0.0, 1954.1872863769531],
    [0.0, 3981.985991142684, 1103.6978149414062],
    [0.0, 0.0, 1.0]
])
intrinsics = o3d.camera.PinholeCameraIntrinsic()
intrinsics.intrinsic_matrix = intrinsic_matrix

# Create point cloud from RGBD image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

# Rotation matrix and translation vector from the JSON
R = np.array([
    [0.9887484396846155, 0.1492973613521459, 0.00931779267969649],
    [-0.06795137741673805, 0.3927819263944453, 0.9171177509492592],
    [0.13326339926949368, -0.9074319022480067, 0.3985074785433647]
])
t = np.array([-337.4470492221228, 587.8991738764912, -10.381216800868486])

# Apply rotation and translation to the point cloud
pcd.transform(np.vstack([np.hstack([R, t.reshape(-1, 1)]), [0, 0, 0, 1]]))

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])
