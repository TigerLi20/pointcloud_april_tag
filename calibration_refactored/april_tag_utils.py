# april_tag_utils.py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector

TAG_SIZE = 0.173
def detect_apriltags(image, camera_matrix, tag_size=TAG_SIZE, family='tag36h11'):
    """
    Detect AprilTags in an image and return their detections.
    
    Args:
        image: Input image (numpy array)
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients
        tag_size: physical size of the tag (meters)
        family: AprilTag family to use for detection
    Returns:

        detections: List of AprilTag detection objects
        
    """
    # Convert to grayscale if image is BGR
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detector = Detector(families=family, nthreads=1, quad_decimate=1.0, quad_sigma=0.0,
                        refine_edges=True, decode_sharpening=0.25, debug=False)

    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    camera_params = [fx, fy, cx, cy]
    # Detect tags
    detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params,
                                 tag_size=tag_size)

    return detections


def compute_extrinsics_from_apriltag(det, camera_matrix, dist_coeffs, tag_world_positions, tag_size):
    """
    Compute camera-to-world extrinsic matrix from an AprilTag detection.
    
    Args:
        det: AprilTag detection object (with .tag_id and .corners)
        camera_matrix: 3x3 intrinsic matrix
        dist_coeffs: distortion coefficients
        tag_world_positions: dict of tag_id -> world center position
        tag_size: physical size of the tag (meters)

    Returns:
        success (bool), T_wc (4x4 np.ndarray), rvec, tvec, tag_id
    """
    tag_id = det.tag_id
    if tag_id not in tag_world_positions:
        return False, None, None, None, tag_id

    # Tag corner points in local tag coordinate
    obj_pts = np.array([
        [-tag_size/2, -tag_size/2, 0],
        [ tag_size/2, -tag_size/2, 0],
        [ tag_size/2,  tag_size/2, 0],
        [-tag_size/2,  tag_size/2, 0]
    ], dtype=np.float32)

    # Shift to world position
    tag_center = tag_world_positions[tag_id]
    obj_pts_world = obj_pts + tag_center

    # If tag is flipped, apply 180° rotation about Y-axis
    if tag_id == 1:
        R_flip = R.from_euler('y', 180, degrees=True).as_matrix()
        obj_pts_world = (R_flip @ obj_pts_world.T).T

    # Image points from detection
    img_pts = np.array(det.corners, dtype=np.float32)

    # SolvePnP: world points -> image points
    success, rvec, tvec = cv2.solvePnP(obj_pts_world, img_pts, camera_matrix, dist_coeffs)

    if not success:
        return False, None, None, None, tag_id

    # Convert to camera-to-world transform
    R_cw, _ = cv2.Rodrigues(rvec)
    t_cw = tvec.reshape(3, 1)
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw

    T_wc = np.eye(4)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = t_wc.ravel()

    return True, T_wc, rvec, tvec, tag_id


# ==== Additional Utilities ====

def load_camera_intrinsics(yaml_path, cam_name):
    """
    Load camera intrinsics from a YAML file with structure:
    cam_name:
        fx, fy, cx, cy
        distortion_coeffs: [...]
    """
    import yaml
    with open(yaml_path, "r") as f:
        intrinsics_all = yaml.safe_load(f)
    cam_params = intrinsics_all[cam_name]
    fx, fy, cx, cy = cam_params["fx"], cam_params["fy"], cam_params["cx"], cam_params["cy"]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array(cam_params["distortion_coeffs"], dtype=np.float64)
    return K, dist_coeffs


def draw_tag_annotations(image, det, rvec, tvec, camera_matrix, dist_coeffs, tag_size):
    """
    Draw bounding box, ID, and axes for an AprilTag detection.
    """
    pts = det.corners.reshape((-1,1,2)).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=(0,255,255), thickness=2)
    corner_pos = tuple(pts[0][0])
    cv2.putText(image, f"ID:{det.tag_id}", (corner_pos[0]+5, corner_pos[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Draw axes
    axis = np.float32([[0,0,0],[tag_size,0,0],[0,tag_size,0],[0,0,-tag_size]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    o = tuple(imgpts[0].ravel().astype(int))
    cv2.line(image, o, tuple(imgpts[1].ravel().astype(int)), (0,0,255), 2)
    cv2.line(image, o, tuple(imgpts[2].ravel().astype(int)), (0,255,0), 2)
    cv2.line(image, o, tuple(imgpts[3].ravel().astype(int)), (255,0,0), 2)
    return image


# ==== Main test case ====
if __name__ == "__main__":
    import os

    # === Configuration ===
    test_image_path = "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam1/color/frame_000001.png"
    intrinsics_path = "/Users/ivy/Downloads/calibrate_dataset/3_cam_intrinsics.yaml"
    cam_name = "cam1"
    tag_world_positions = {0: np.array([0,0,0]), 1: np.array([0.5,0.15,-0.46])}

    # === Load image ===
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        exit(1)
    image = cv2.imread(test_image_path)

    # === Load camera intrinsics ===
    K, dist = load_camera_intrinsics(intrinsics_path, cam_name)

    # === Detect AprilTags ===
    detections = detect_apriltags(image, K, TAG_SIZE)
    if not detections:
        print("No AprilTags detected.")
        exit(0)

    # === Compute extrinsics for first detected tag ===
    det = detections[0]
    success, T_wc, rvec, tvec, tag_id = compute_extrinsics_from_apriltag(det, K, dist, tag_world_positions, tag_size)
    if success:
        print(f"Tag ID: {tag_id}")
        print("Camera-to-World Extrinsics:\n", T_wc)

        # Draw annotations
        annotated = draw_tag_annotations(image.copy(), det, rvec, tvec, K, dist, tag_size)
        cv2.imshow("AprilTag Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to compute extrinsics.")