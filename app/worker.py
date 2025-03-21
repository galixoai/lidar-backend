from temporalio.client import Client
from temporalio.worker import Worker
from .utils import PointCloudWorkflow
from .activities import process_point_cloud_activity

async def main():
    client = await Client.connect("temporal-worker:7233")  # Change for Temporal Cloud if needed

    worker = Worker(
        client,
        task_queue="pointcloud-queue",
        workflows=[PointCloudWorkflow],
        activities=[process_point_cloud_activity],
    )

    await worker.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
