from temporalio import activity
from app.pointcloud import process_point_cloud


@activity.defn
async def process_point_cloud_activity(project_directory: str, intrinsic_path: str):
    print(
        f"[Activity] Running process_point_cloud with {project_directory}, {intrinsic_path}")
    result = process_point_cloud(project_directory, intrinsic_path)
    print(f"[Activity] Finished with result: {result}")
    return result


# @activity.defn
# async def process_point_cloud_activity(project_directory: str, intrinsic_path: str):
#     print("[Activity] Simulated processing...")
#     return {"output_path": "/simulated/test.ply"}
