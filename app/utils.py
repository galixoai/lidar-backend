from temporalio import workflow
from .activities import process_point_cloud_activity

@workflow.defn
class PointCloudWorkflow:
    @workflow.run
    async def run(self, project_directory: str, intrinsic_path: str):
        return await workflow.execute_activity(
            process_point_cloud_activity, project_directory, intrinsic_path,
            start_to_close_timeout=600  # Timeout in seconds
        )
