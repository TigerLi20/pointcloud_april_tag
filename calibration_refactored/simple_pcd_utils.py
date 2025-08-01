# utils for processing point cloud data
import numpy as np

def load_pcd(file_path):
    """
    Load a PCD file and return the point cloud data as a numpy array.
    
    Args:
        file_path (str): Path to the PCD file.
        
    Returns:
        np.ndarray: Point cloud data as a Nx3 numpy array.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the start of the data section
    data_start = lines.index('DATA ascii\n') + 1
    
    # Read the point cloud data
    points = []
    for line in lines[data_start:]:
        if line.strip():  # Skip empty lines
            points.append(list(map(float, line.strip().split())))
    
    return np.array(points, dtype=np.float32)

def load_rgb_depth_image(rgb_path, depth_path):
    """
    Load RGB and depth images from given file paths.
    
    Args:
        rgb_path (str): Path to the RGB image file.
        depth_path (str): Path to the depth image file.
        
    Returns:
        tuple: (rgb_image, depth_image) where both are numpy arrays.
    """
    rgb_image = np.array(Image.open(rgb_path))
    depth_image = np.array(Image.open(depth_path))
    
    return rgb_image, depth_image

def compute_transformation(R, t):
    """
    Construct a 4x4 homogeneous transformation matrix from rotation and translation.
    
    Args:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
        
    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T

def transform_from_a_to_b(a, b):
    """
    Compute the transformation matrix from frame A to frame B.
    
    Args:
        a (np.ndarray): Transformation matrix from world to frame A.
        b (np.ndarray): Transformation matrix from world to frame B.
        
    Returns:
        np.ndarray: Transformation matrix from frame A to frame B.
    """
    return np.linalg.inv(a) @ b

import cv2

def tag_to_frame_transformation(obj_pts_world, img_pts, camera_matrix, dist_coeffs):
    """
    Compute camera-to-world transformation using cv2.solvePnP.

    Args:
        obj_pts_world (np.ndarray): Nx3 array of object points in world coordinates.
        img_pts (np.ndarray): Nx2 array of corresponding image points.
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        dist_coeffs (np.ndarray): distortion coefficients.

    Returns:
        success (bool): Whether solvePnP succeeded.
        T_wc (np.ndarray): 4x4 transformation matrix from camera to world.
        rvec (np.ndarray): Rotation vector from solvePnP.
        tvec (np.ndarray): Translation vector from solvePnP.
    """
    success, rvec, tvec = cv2.solvePnP(obj_pts_world, img_pts, camera_matrix, dist_coeffs)
    if not success:
        return False, None, None, None
    R_cw, _ = cv2.Rodrigues(rvec)
    t_cw = tvec.reshape(3, 1)
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw
    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc.ravel()
    return True, T_wc, rvec, tvec


def intrinsics_to_extrinsics_tranformation(obj_pts_world=None, img_pts=None, camera_matrix=None, dist_coeffs=None):
    """
    Wrapper to compute extrinsics from intrinsics and correspondences, or return identity if not provided.
    """
    if obj_pts_world is None or img_pts is None or camera_matrix is None or dist_coeffs is None:
        return np.eye(4)
    success, T_wc, rvec, tvec = tag_to_frame_transformation(obj_pts_world, img_pts, camera_matrix, dist_coeffs)
    if not success:
        return np.eye(4)
    return T_wc