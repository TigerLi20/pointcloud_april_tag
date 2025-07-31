import cv2
import numpy as np
from pupil_apriltags import Detector
import glob
import os
import yaml
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
import sys

# === 🟨 CONFIGURATION ===

def prompt_user_choice(prompt, options):
    print(prompt)
    for i, opt in enumerate(options):
        print(f"  {i+1}. {opt}")
    while True:
        try:
            idx = int(input("Enter number: ")) - 1
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print("Invalid selection. Try again.")
        except Exception:
            print("Invalid input. Enter a number.")

# Prompt for trial and camera
trial_options = ["trial_1", "trial_2", "trial_3", "trial_4"]
cam_options = ["cam1", "cam2", "cam3"]

selected_trial = prompt_user_choice("Select trial:", trial_options)
selected_cam = prompt_user_choice("Select camera:", cam_options)

color_dir = f"/Users/tigerli/Downloads/CALIBRATE/{selected_trial}/{selected_cam}/color"
depth_dir = f"/Users/tigerli/Downloads/CALIBRATE/{selected_trial}/{selected_cam}/depth"

# Set intrinsics and distortion for each camera
if selected_cam == "cam1":
    fx = 385.716552734375
    fy = 385.2541198730469
    cx = 320.8727722167969
    cy = 239.6362762451172
    dist_coeffs = np.array([-0.05676564201712608, 0.06551581621170044, 0.00036026633461005986, 0.0010946334805339575, -0.020970700308680534], dtype=np.float64)
elif selected_cam == "cam2":
    fx = 614.52099609375
    fy = 614.460693359375
    cx = 322.1363525390625
    cy = 248.58843994140625
    dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
elif selected_cam == "cam3":
    fx = 388.5696716308594
    fy = 388.11639404296875
    cx = 319.456787109375
    cy = 248.93724060058594
    dist_coeffs = np.array([-0.05426320433616638, 0.06088121235370636, 0.0007697998662479222, 0.0004985771374776959, -0.019820772111415863], dtype=np.float64)
else:
    print("Unknown camera selection.")
    sys.exit(1)

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

color_images = sorted(glob.glob(os.path.join(color_dir, "frame_*.png")))  # process all frames

tag_size = 0.173  # in meters

# Known tag world positions (cam1 = origin)
tag_world_positions = {
    0: np.array([0.0, 0.0, 0.0]),
    1: np.array([0.5, 0.15, -0.46])  # relative offset from tag 0 (based on your tvecs)
}

def compute_transform(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Bm.T @ Am
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_A - R @ centroid_B
    return R, t

def draw_axes(img, rvec, tvec, cam_mtx, dist, length):
    axis = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, -length]
    ])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, cam_mtx, dist)
    origin = tuple(imgpts[0].ravel().astype(int))
    cv2.line(img, origin, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 2)
    cv2.line(img, origin, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 2)
    cv2.line(img, origin, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 2)
    return img

def draw_world_axes(img, origin=(40, 40), length=30):
    ox, oy = origin
    # X axis: out of screen (draw as a blue circle with label 'X')
    cv2.circle(img, (ox, oy), 8, (255, 0, 0), -1)
    cv2.putText(img, 'X', (ox - 25, oy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    # Y axis: right (green)
    cv2.arrowedLine(img, (ox, oy), (ox - length, oy), (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(img, 'Z', (ox - length - 20, oy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Z axis: up (red)
    cv2.arrowedLine(img, (ox, oy), (ox, oy - length), (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(img, 'Y', (ox - 10, oy - length - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img

def depth2pcd(depth, camera_matrix):
    """Convert depth image to point cloud in camera frame."""
    h, w = depth.shape
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter valid depth values
    valid_mask = (depth > 0.0) & (depth < 5.0)
    
    # Convert to 3D points in camera frame
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    
    # Stack points
    points = np.stack([x, y, z], axis=1)
    return points, valid_mask

def create_point_cloud(rgb_image, depth_image, camera_matrix, extrinsics):
    """Step 3: Create point cloud using Open3D library."""
    # Ensure RGB and depth are the same size
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        print(f"Resizing RGB from {rgb_image.shape[:2]} to {depth_image.shape[:2]}")
        rgb_image = cv2.resize(rgb_image, (depth_image.shape[1], depth_image.shape[0]))

    # Clamp depth values to a reasonable range (10cm to 5m)
    valid_mask = (depth_image > 0.1) & (depth_image < 5.0)
    h, w = depth_image.shape
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_image[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)

    # Optionally flip axes if needed (uncomment one if your cloud is upside down)
    #points_cam[:, 1] *= -1  # Flip Y
    # points_cam[:, 2] *= -1  # Flip Z

    # Transform to world coordinates
    points_homogeneous = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
    points_world = (extrinsics @ points_homogeneous.T).T[:, :3]

    # Get colors
    colors = rgb_image[valid_mask] / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

print(f"Found {len(color_images)} image(s) to process.")

for image_path in color_images:
    print(f"\nProcessing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found at: {image_path}")
        continue

    # === STEP 1: Identify 3D pose of AprilTags in camera frame ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True)
    detections = detector.detect(gray)
    print(f"Found {len(detections)} AprilTags.")
    tag_ids = [det.tag_id for det in detections]
    print(f"Detected tag IDs: {tag_ids}")

    camera_observed_points = []
    world_known_points = []

    for det in detections:
        tag_id = det.tag_id
        if tag_id not in tag_world_positions:
            continue

        obj_pts = np.array([
            [-tag_size/2, -tag_size/2, 0],
            [ tag_size/2, -tag_size/2, 0],
            [ tag_size/2,  tag_size/2, 0],
            [-tag_size/2,  tag_size/2, 0]
        ], dtype=np.float32)

        img_pts = np.array(det.corners, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)

        if success:
            print(f"✅ Tag ID: {tag_id}")
            print("Rotation vector (rvec):\n", rvec)
            print("Translation vector (tvec):\n", tvec)
            distance = np.linalg.norm(tvec)
            print(f"Distance to tag center: {distance:.2f} meters")
            R_tag, _ = cv2.Rodrigues(rvec)
            print("Rotation matrix (tag in camera frame):\n", R_tag)

            # Tag center in camera frame is just tvec
            camera_observed_points.append(tvec.flatten())
            world_known_points.append(tag_world_positions[tag_id])

            # Draw yellow bounding box and label tag ID
            img_pts_int = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [img_pts_int], isClosed=True, color=(0, 255, 255), thickness=2)
            tag_id_pos = tuple(det.corners[0].astype(int))
            cv2.putText(image, f"ID:{tag_id}", tag_id_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw axes
            image = draw_axes(image, rvec, tvec, camera_matrix, dist_coeffs, length=tag_size/2)
        else:
            print(f"❌ Failed to estimate pose for tag {tag_id}")

    # Draw world axes image
    image = draw_world_axes(image, origin=(40, 40), length=30)

    # Save debug image to debug_detect folder
    debug_dir = os.path.join(os.path.dirname(image_path), "debug_detect")
    os.makedirs(debug_dir, exist_ok=True)
    debug_img_path = os.path.join(debug_dir, os.path.basename(image_path))
    cv2.imwrite(debug_img_path, image)
    print(f"Saved debug image to: {debug_img_path}")

    # === Display the result for 0.1 seconds ===
    cv2.imshow("AprilTag Pose Estimation", image)
    # Do not call waitKey or destroyAllWindows here, keep window open

    # Remove old extrinsics YAML files from the color folder (only once, before processing images)
    if 'extrinsics_cleanup_done' not in globals():
        for f in os.listdir(os.path.dirname(image_path)):
            if f.endswith('_extrinsics.yaml'):
                try:
                    os.remove(os.path.join(os.path.dirname(image_path), f))
                except Exception as e:
                    print(f"Warning: could not remove {f}: {e}")
        global extrinsics_cleanup_done
        extrinsics_cleanup_done = True

    # === STEP 2: Compute camera extrinsics using world locations ===
    if len(camera_observed_points) >= 2:
        R, t = compute_transform(world_known_points, camera_observed_points)
        print("\n📌 Estimated Camera Pose in World Frame:")
        print("Rotation matrix:\n", R)
        print("Translation vector:\n", t)

        # Create 4x4 extrinsics matrix
        extrinsics_matrix = np.eye(4)
        extrinsics_matrix[:3, :3] = R
        extrinsics_matrix[:3, 3] = t

        extrinsics = {
            "rotation_matrix": R.tolist(),
            "translation_vector": t.tolist(),
            "extrinsics_4x4": extrinsics_matrix.tolist()
        }

        # Save a unique YAML file for each image in a subfolder 'extrinsics'
        extrinsics_dir = os.path.join(os.path.dirname(image_path), "extrinsics")
        os.makedirs(extrinsics_dir, exist_ok=True)
        yaml_name = os.path.splitext(os.path.basename(image_path))[0] + "_extrinsics.yaml"
        output_yaml = os.path.join(extrinsics_dir, yaml_name)
        with open(output_yaml, "w") as f:
            yaml.dump(extrinsics, f)
        print(f"Saved extrinsics to: {output_yaml}")

        # === STEP 3: Create point cloud using Open3D library ===
        # Look for corresponding depth image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        depth_candidates = [
            os.path.join(depth_dir, f"{base_name}.npy"),
            os.path.join(depth_dir, f"{base_name}.png"),
            os.path.join(depth_dir, f"{base_name}_depth.png"),
        ]
        
        depth_path = None
        for candidate in depth_candidates:
            if os.path.exists(candidate):
                depth_path = candidate
                break
        
        if depth_path is not None:
            print(f"Found depth image: {depth_path}")
            
            # Load depth image
            if depth_path.endswith('.npy'):
                depth_image = np.load(depth_path)
                if depth_image.max() > 10:  # Convert from mm to m if needed
                    depth_image = depth_image / 1000.0
            else:
                depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                if depth_image.max() > 10:  # Convert from mm to m if needed
                    depth_image = depth_image / 1000.0
            
            # Convert BGR to RGB for point cloud colors
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create point cloud
            try:
                pcd = create_point_cloud(rgb_image, depth_image, camera_matrix, extrinsics_matrix)
                
                # Save point cloud
                pcd_dir = os.path.join(os.path.dirname(image_path), "pointclouds")
                os.makedirs(pcd_dir, exist_ok=True)
                pcd_path = os.path.join(pcd_dir, f"{base_name}.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f"Saved point cloud to: {pcd_path}")
                
                # Optionally visualize point cloud (comment out if not needed)
                coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
                print("Displaying point cloud... Close window to continue.")
                o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                                window_name=f"Point Cloud - {base_name}")
                # After closing point cloud window, close the color image window
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f"❌ Error creating point cloud: {e}")
        else:
            print(f"❌ No depth image found for {base_name}")
            print("Point cloud creation skipped.")
    else:
        print("❌ Not enough valid tags for computing extrinsics.")

print("\n🎉 Processing complete!")
print("Results saved in:")
print("  - debug_detect/: Detection visualization images")
print("  - extrinsics/: Camera extrinsics YAML files") 
print("  - pointclouds/: Point cloud PLY files")