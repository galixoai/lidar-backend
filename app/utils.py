from datetime import timedelta
from temporalio import workflow


@workflow.defn
class PointCloudWorkflow:
    @workflow.run
    async def run(self, project_directory: str, intrinsic_path: str):
        return await workflow.execute_activity(
            "process_point_cloud_activity",
            args=[project_directory, intrinsic_path],
            start_to_close_timeout=timedelta(seconds=600)
        )
