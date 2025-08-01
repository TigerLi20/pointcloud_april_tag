import os
import cv2
import numpy as np
import yaml
import open3d as o3d
from april_tag_utils import detect_apriltags, compute_extrinsics_from_apriltag, load_camera_intrinsics, draw_tag_annotations

def run_pointcloud_concatenation_pipeline(
    image_paths,
    depth_paths,
    cam_names,
    intrinsics_path,
    tag_world_positions,
    tag_size,
    visualize=True,
    manual_origin=True
):
    """
    Pipeline to load images, detect AprilTags, compute extrinsics,
    generate and concatenate point clouds, and visualize.
    Args:
        image_paths: list of color image paths (one per camera)
        depth_paths: list of depth image paths (one per camera)
        cam_names: list of camera names matching the YAML intrinsics
        intrinsics_path: path to YAML intrinsics file
        tag_world_positions: dict mapping tag_id to world center np.array([x, y, z])
        tag_size: physical size of AprilTag (meters)
        visualize: if True, visualize concatenated point clouds
        manual_origin: if True, manually select origin by clicking on first camera image
    Returns:
        list of open3d.geometry.PointCloud objects (transformed to world)
    """
    assert len(image_paths) == len(depth_paths) == len(cam_names), "Input lists must have same length"
    pointclouds = []
    extrinsics_dict = {}

    # Define path for world_origin.yaml in same directory as intrinsics_path
    intrinsics_dir = os.path.dirname(intrinsics_path)
    world_origin_path = os.path.join(intrinsics_dir, "world_origin.yaml")

    # Check if world_origin.yaml exists and load T_world_origin, else None
    if os.path.exists(world_origin_path):
        with open(world_origin_path, "r") as f:
            data = yaml.safe_load(f)
            T_world_origin = np.array(data["T_world_origin"])
        print(f"Loaded existing world origin from {world_origin_path}")
    else:
        T_world_origin = None

    T_wc_first = None

    for idx, (img_path, depth_path, cam_name) in enumerate(zip(image_paths, depth_paths, cam_names)):
        print(f"\nProcessing camera: {cam_name}")
        # --- Step 1: Load color, depth, and intrinsics ---
        color_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if color_img is None:
            raise FileNotFoundError(f"Color image not found: {img_path}")
        # Handle .npy depth files
        if depth_path.lower().endswith(".npy"):
            depth_img = np.load(depth_path)
            # Convert to float32 and meters if likely in millimeters
            if depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32)
            if np.nanmax(depth_img) > 10:  # likely millimeters
                depth_img = depth_img / 1000.0
        else:
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                raise FileNotFoundError(f"Depth image not found: {depth_path}")
        K, dist = load_camera_intrinsics(intrinsics_path, cam_name)

        # If manual_origin is True and this is first camera and no world_origin.yaml exists, get origin from click
        if manual_origin and idx == 0 and T_world_origin is None:
            clicked_point = []
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    clicked_point.append((x, y))
                    cv2.destroyAllWindows()
            cv2.namedWindow("Select Origin Point")
            cv2.setMouseCallback("Select Origin Point", mouse_callback)
            cv2.imshow("Select Origin Point", color_img)
            print("Please click on the desired world origin point in the image window.")
            while True:
                key = cv2.waitKey(1)
                if len(clicked_point) > 0 or key == 27:  # ESC to exit if no click
                    break
            cv2.destroyAllWindows()
            if len(clicked_point) == 0:
                raise RuntimeError("No point selected for manual origin.")
            u, v = clicked_point[0]
            # Ensure pixel indices are integers and within image bounds
            u = int(np.clip(u, 0, depth_img.shape[1] - 1))
            v = int(np.clip(v, 0, depth_img.shape[0] - 1))
            depth_val = depth_img[v, u]
            if depth_val == 0 or np.isnan(depth_val):
                raise RuntimeError("Selected pixel has invalid depth value.")
            # Back-project to camera coordinates
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            X_c = np.array([
                (u - cx) * depth_val / fx,
                (v - cy) * depth_val / fy,
                depth_val
            ])
            # Construct T_world_origin as identity with translation = -X_c
            T_world_origin = np.eye(4)
            T_world_origin[0:3, 3] = -X_c
            # Save to world_origin.yaml
            with open(world_origin_path, "w") as f:
                yaml.dump({"T_world_origin": T_world_origin.tolist()}, f)
            print(f"Saved manual world origin to {world_origin_path}")

        # --- Step 2: Detect AprilTags and draw bounding boxes for debugging ---
        detections = detect_apriltags(color_img, K, tag_size)
        if len(detections) == 0:
            print(f"No AprilTags detected in {cam_name}")
            continue
        # Use the first detected tag for extrinsics
        det = detections[0]
        # Compute extrinsics
        success, T_wc, rvec, tvec, tag_id = compute_extrinsics_from_apriltag(
            det, K, dist, tag_world_positions, tag_size
        )
        if not success:
            print(f"Failed to compute extrinsics for {cam_name}")
            continue

        # If T_world_origin is None and this is first successful camera, assign and save
        if T_world_origin is None:
            T_world_origin = T_wc
            T_wc_first = T_wc
            with open(world_origin_path, "w") as f:
                yaml.dump({"T_world_origin": T_world_origin.tolist()}, f)
            print(f"Saved new world origin to {world_origin_path}")
        elif T_wc_first is None:
            T_wc_first = T_wc

        # Align extrinsics
        T_wc_aligned = T_world_origin @ np.linalg.inv(T_wc_first) @ T_wc

        # Draw annotation for debug
        annotated = draw_tag_annotations(color_img.copy(), det, rvec, tvec, K, dist, tag_size)
        debug_vis_path = os.path.splitext(img_path)[0] + "_apriltag_debug.png"
        cv2.imwrite(debug_vis_path, annotated)
        print(f"AprilTag annotation saved to {debug_vis_path}")

        # --- Step 3: Save extrinsics as YAML (aligned) ---
        extrinsics_dict[cam_name] = {"T_wc": T_wc_aligned.tolist(), "tag_id": int(tag_id)}
        extrinsics_yaml_path = os.path.splitext(img_path)[0] + "_extrinsics.yaml"
        with open(extrinsics_yaml_path, "w") as f:
            yaml.dump({"T_wc": T_wc_aligned.tolist(), "tag_id": int(tag_id)}, f)
        print(f"Extrinsics saved to {extrinsics_yaml_path}")

        # --- Step 4: Create point cloud and transform to world frame ---
        # Convert color to RGB for Open3D
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        # Open3D expects float depth in meters
        if depth_img.dtype == np.uint16:
            # Assume depth is in millimeters
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)
        color_o3d = o3d.geometry.Image(color_rgb)
        depth_o3d = o3d.geometry.Image(depth_m)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1.0, depth_trunc=5.0
        )
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width=color_rgb.shape[1],
            height=color_rgb.shape[0],
            fx=K[0,0], fy=K[1,1],
            cx=K[0,2], cy=K[1,2]
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic_o3d
        )
        # Transform to world frame using aligned extrinsics
        pcd.transform(T_wc_aligned)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pointclouds.append(pcd)
    # --- Step 5: Visualization ---
    if visualize and len(pointclouds) > 0:
        print("Visualizing concatenated point clouds...")
        vis_objs = []
        for idx, (pcd, cam_name) in enumerate(zip(pointclouds, cam_names)):
            # Use original colors without painting uniform color
            pcd_c = o3d.geometry.PointCloud(pcd)
            vis_objs.append(pcd_c)
            # Add camera coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis_objs.append(frame.transform(extrinsics_dict[cam_name]["T_wc"]))
        # Add world frame
        vis_objs.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
        o3d.visualization.draw_geometries(vis_objs)
    return pointclouds


if __name__ == "__main__":
    # === Example usage ===
    # Please update the paths and tag_world_positions as per your dataset.
    example_image_paths = [
        "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam1/color/frame_000001.png",
        "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam2/color/frame_000001.png",
        "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam3/color/frame_000001.png",
    ]
    example_depth_paths = [
        "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam1/depth/frame_000001.npy",
        "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam2/depth/frame_000001.npy",
        "/Users/ivy/Downloads/calibrate_dataset/trial_1/cam3/depth/frame_000001.npy",
    ]
    example_cam_names = ["cam1", "cam2", "cam3"]
    example_intrinsics_path = "/Users/ivy/Downloads/calibrate_dataset/3_cam_intrinsics.yaml"
    example_tag_world_positions = {0: np.array([0,0,0]), 1: np.array([0.5,0.15,-0.46])}
    example_tag_size = 0.173
    run_pointcloud_concatenation_pipeline(
        example_image_paths,
        example_depth_paths,
        example_cam_names,
        example_intrinsics_path,
        example_tag_world_positions,
        example_tag_size,
        visualize=True
    )