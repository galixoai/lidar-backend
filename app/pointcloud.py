import os
import open3d as o3d
import numpy as np
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_rgbd_image(color_path, depth_path, depth_scale=1000.0, depth_trunc=50.0):
    # Load images using Open3D
    color = o3d.io.read_image(color_path)
    
    data = np.fromfile(depth_path, dtype=np.uint16)
    color_np = np.asarray(color)
    width = color_np.shape[1]
    height = color_np.shape[0]
    if data.size != width * height:
        raise ValueError(f"Unexpected data size {data.size} for resolution {width}x{height}")
    
    data = data.reshape((height, width))
    logger.info("Depth data array shape: %s", data.shape)
    
    depth = o3d.geometry.Image(data)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    return rgbd_image

def create_point_cloud(rgbd_image, intrinsic, extrinsic):
    """
    Generate a point cloud from an RGB-D image and transform it using extrinsics.
    """
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform(extrinsic)  # Apply transformation
    return pcd

def estimate_normals(pcd, search_radius=0.2, max_nn=30):
    """
    Estimate normals for the point cloud without parallel processing.
    """
    logger.info("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn)
    )
    logger.info("Orienting normals...")
    pcd.orient_normals_towards_camera_location()
    
    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.1):
    """
    Downsample and remove statistical outliers from the point cloud.
    """
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    # Remove statistical outliers
    pcd_clean, ind = pcd_down.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
    return pcd_clean

def load_intrinsic(intrinsic_path=None):
    """
    Load camera intrinsic parameters from a text file or use default PrimeSense parameters.
    """
    if intrinsic_path and os.path.exists(intrinsic_path):
        with open(intrinsic_path, "r") as f:
            lines = f.readlines()
        
        params = {}
        for line in lines:
            key, value = line.strip().split(":")
            params[key.strip()] = float(value.strip())
        
        fx, fy = params["fx"], params["fy"]
        cx, cy = params["cx"], params["cy"]
        width, height = int(2 * cx), int(2 * cy)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        logger.info("Loaded camera intrinsics from %s", intrinsic_path)
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
        logger.info("Using default PrimeSense intrinsic parameters.")
    return intrinsic

def load_extrinsic(metadata_path):
    """
    Load extrinsic transformation matrix from metadata JSON.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return np.array(metadata["transformed_pose"])  # Load transformed pose as a NumPy array

def register_point_clouds(source, target, max_correspondence_distance_coarse=0.02, max_correspondence_distance_fine=0.01):
    """
    Perform ICP registration between source and target point clouds.
    Returns the transformation matrix.
    """
    logger.info("Starting ICP registration...")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    if icp_coarse.fitness < 0.1:
        logger.warning("Coarse ICP fitness too low, skipping registration.")
        return None

    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    if icp_fine.fitness < 0.1:
        logger.warning("Fine ICP fitness too low, skipping registration.")
        return None

    return icp_fine.transformation

def process_point_cloud(self, project_directory, intrinsic_path=None, voxel_size=0.005, 
                        max_correspondence_distance_coarse=0.02, max_correspondence_distance_fine=0.01):
    """
    Celery task to process a project's image frames into a combined point cloud.
    Updates its state along the way.
    """
    try:
        logger.info("Starting process_point_cloud task.")

        intrinsic = load_intrinsic(intrinsic_path)
        combined_pcd = None

        # Define directories
        color_dir = os.path.join(project_directory, 'rgb')
        depth_dir = os.path.join(project_directory, 'depth')
        metadata_dir = os.path.join(project_directory, 'metadata')

        if not all(os.path.exists(d) for d in [color_dir, depth_dir, metadata_dir]):
            error_msg = f"Missing one or more directories in {project_directory}"
            logger.error(error_msg)
            raise Exception(error_msg)

        try:
            color_files = sorted(os.listdir(color_dir))
            depth_files = sorted(os.listdir(depth_dir))
            metadata_files = sorted(os.listdir(metadata_dir))
        except Exception as e:
            logger.error("Error listing directory contents: %s", str(e))
            raise Exception(str(e))

        if len(color_files) != len(depth_files) or len(color_files) != len(metadata_files):
            error_msg = "Mismatch between file counts in project directory."
            logger.error(error_msg)
            raise Exception(error_msg)

        total_frames = len(color_files)
        logger.info("Processing %d image pairs.", total_frames)

        # Process each frame and update progress
        for idx, (color_file, depth_file, metadata_file) in enumerate(zip(color_files, depth_files, metadata_files)):
            # Build full paths
            color_path = os.path.join(color_dir, color_file)
            depth_path = os.path.join(depth_dir, depth_file)
            metadata_path = os.path.join(metadata_dir, metadata_file)

            if not all(os.path.isfile(p) for p in [color_path, depth_path, metadata_path]):
                logger.warning("Missing files for frame %d, skipping.", idx + 1)
                continue

            try:
                rgbd_image = load_rgbd_image(color_path, depth_path)
                if rgbd_image is None:
                    logger.warning("Failed to load RGBD image for frame %d, skipping.", idx + 1)
                    continue

                extrinsic = load_extrinsic(metadata_path)
                pcd = create_point_cloud(rgbd_image, intrinsic, extrinsic)
                pcd = estimate_normals(pcd)
                # Optionally, you can preprocess the point cloud:
                # pcd = preprocess_point_cloud(pcd, voxel_size=voxel_size)
                
                if combined_pcd is None:
                    combined_pcd = pcd
                    logger.info("Initialized combined point cloud with frame %d.", idx + 1)
                else:
                    transformation = register_point_clouds(pcd, combined_pcd, 
                                                           max_correspondence_distance_coarse, 
                                                           max_correspondence_distance_fine)
                    if transformation is not None:
                        pcd.transform(transformation)
                        combined_pcd += pcd
                        logger.info("Registered and merged frame %d.", idx + 1)
                    else:
                        logger.warning("Skipping frame %d due to poor registration.", idx + 1)
            except Exception as e:
                logger.error("Error processing frame %d: %s", idx + 1, str(e))
                continue

        # Final success state
        if combined_pcd is not None and combined_pcd.has_points():
            output_path = os.path.join(project_directory, "point_cloud.ply")
            try:
                o3d.io.write_point_cloud(output_path, combined_pcd)
                logger.info("Saved combined point cloud to %s", output_path)
                result = {
                    'output_path': output_path
                }
                return result
            except Exception as e:
                logger.error("Error saving point cloud: %s", str(e))
                raise Exception(str(e))
        else:
            error_msg = "No valid point cloud data generated."
            logger.error(error_msg)
            raise Exception(error_msg)
    except Exception as e:
        logger.exception("Unexpected error in process_point_cloud:")
        raise Exception(str(e))
