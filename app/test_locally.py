# run_local.py
import os
from pointcloud import process_point_cloud

if __name__ == "__main__":
    # Path to the project directory with `rgb`, `depth`, `metadata` folders and `intrinsics.txt`
    project_dir = "./ProjectData/91488e4d-77db-4ea2-9263-e84e5f191a96"
    intrinsic_path = os.path.join(project_dir, "intrinsics.txt")

    result = process_point_cloud(project_dir, intrinsic_path)
    print("✅ Point cloud saved at:", result["output_path"])
