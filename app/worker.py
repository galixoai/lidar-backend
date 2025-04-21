from temporalio.worker import Worker
from temporalio.client import Client
from app.activities import process_point_cloud_activity
from app.utils import PointCloudWorkflow
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
print("[Worker] Starting worker...")


async def main():
    client = await Client.connect("temporal-worker:7233")

    worker = Worker(
        client,
        task_queue="pointcloud-queue",
        workflows=[PointCloudWorkflow],
        activities=[process_point_cloud_activity],
    )

    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
