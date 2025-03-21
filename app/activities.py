from temporalio import activity
import logging
from app.pointcloud import process_point_cloud

logger = logging.getLogger(__name__)

@activity.defn
async def process_point_cloud_activity(project_directory, intrinsic_path):
    logger.info(f"Processing Point Cloud for {project_directory}")
    result = process_point_cloud(project_directory, intrinsic_path)
    return result
