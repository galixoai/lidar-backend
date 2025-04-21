from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse
import uuid
import os
import json
import shutil
import numpy as np
import logging
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from temporalio.client import Client
from app.utils import PointCloudWorkflow

from temporalio.service import RPCError, RPCStatusCode
import open3d as o3d


# Set up loggingsq
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "./ProjectData"
os.makedirs(BASE_DIR, exist_ok=True)


def transform_iphone_pose_to_open3d(pose_matrix):
    """
    Converts iPhone (ARKit) rotation matrices to Open3D format.
    - Negates the Z-axis to match Open3D's convention.
    """
    pose_matrix = np.array(pose_matrix)
    pose_matrix[:, 2] *= -1
    return pose_matrix.tolist()


@app.post("/start-project")
async def create_project(
    name: str = Form(...),
    intrisics_params: UploadFile = File(...)
):
    guid = str(uuid.uuid4())
    project_dir = os.path.join(BASE_DIR, guid)
    os.makedirs(project_dir, exist_ok=True)

    intrisics_path = os.path.join(project_dir, "intrinsics.txt")
    with open(intrisics_path, "wb") as f:
        shutil.copyfileobj(intrisics_params.file, f)

    metadata_path = os.path.join(project_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        print("Project Creation")
        f.write(f"Project Name: {name}\n")
        f.write(f"Intrinsics Params: {intrisics_path}\n")

    logger.info("Created project %s", guid)
    return {"guid": guid}


@app.post("/upload-frame")
async def upload_frames(
    rgb_frames: List[UploadFile] = File(...),
    depth_frames: List[UploadFile] = File(...),
    metadata: List[UploadFile] = File(...),
    coordinates: Optional[List[UploadFile]] = File(None),
    guid: str = Form(...),
):
    project_dir = os.path.join(BASE_DIR, guid)

    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail="Project GUID not found.")

    rgb_dir = os.path.join(project_dir, "rgb")
    depth_dir = os.path.join(project_dir, "depth")
    metadata_dir = os.path.join(project_dir, "metadata")
    coordinates_dir = os.path.join(project_dir, "coordinates")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(coordinates_dir, exist_ok=True)

    if len(rgb_frames) != len(depth_frames) or len(rgb_frames) != len(metadata):
        raise HTTPException(
            status_code=400, detail="RGB, depth frames, and metadata must have the same length.")

    existing_files = [f for f in os.listdir(rgb_dir)]
    max_index = max([int(f.split("_")[0]) for f in existing_files], default=-1)

    for i in range(len(rgb_frames)):
        frame_index = max_index + i + 1
        frame_index_str = f"{frame_index:05d}"

        # Save RGB image
        rgb_path = os.path.join(rgb_dir, f"{frame_index_str}_rgb.png")
        with open(rgb_path, "wb") as f:
            shutil.copyfileobj(rgb_frames[i].file, f)

        # Save Depth image
        depth_path = os.path.join(depth_dir, f"{frame_index_str}_depth.png")
        with open(depth_path, "wb") as f:
            shutil.copyfileobj(depth_frames[i].file, f)

        # Process and save metadata
        metadata_path = os.path.join(
            metadata_dir, f"{frame_index_str}_metadata.json")
        try:
            metadata_content = metadata[i].file.read().decode("utf-8").strip()
            # if metadata_content.startswith("simd_float4x4"):
            #     metadata_content = metadata_content.replace(
            #         "simd_float4x4(", "").replace(")", "")
            # original_pose_matrix = eval(metadata_content)

            # if metadata_content.startswith("simd_float4x4"):
            #     metadata_content = metadata_content.replace(
            #         "simd_float4x4(", "").replace(")", "")

            # try:
            #     original_pose_matrix = ast.literal_eval(metadata_content)
            #     pose_matrix = np.array(original_pose_matrix)
            #     if pose_matrix.ndim != 2 or pose_matrix.shape != (4, 4):
            #         raise ValueError("Pose matrix must be a 4x4 array.")
            #     transformed_pose = transform_iphone_pose_to_open3d(pose_matrix)
            # except Exception as e:
            #     raise HTTPException(
            #         status_code=500, detail=f"Invalid metadata format: {str(e)}")
            # transformed_pose = transform_iphone_pose_to_open3d(
            #     original_pose_matrix)
            # metadata_json = {
            #     "original_pose": original_pose_matrix,
            #     "transformed_pose": transformed_pose
            # }

            try:
                metadata_dict = json.loads(metadata_content)
                original_pose_matrix = metadata_dict["original_pose"]
                pose_matrix = np.array(original_pose_matrix)

                if pose_matrix.ndim != 2 or pose_matrix.shape != (4, 4):
                    raise ValueError("Pose matrix must be a 4x4 array.")

                transformed_pose = transform_iphone_pose_to_open3d(pose_matrix)
                metadata_json = {
                    "original_pose": original_pose_matrix,
                    "transformed_pose": transformed_pose
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata_json, f, indent=4)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error processing metadata: {str(e)}")

            with open(metadata_path, "w") as f:
                json.dump(metadata_json, f, indent=4)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing metadata: {str(e)}")

        if coordinates and len(coordinates) > i:
            coordinates_path = os.path.join(
                coordinates_dir, coordinates[i].filename)
            with open(coordinates_path, "wb") as f:
                shutil.copyfileobj(coordinates[i].file, f)

    logger.info("Uploaded %d frames for project %s", len(rgb_frames), guid)
    return {"message": "Frames, metadata, and transformed poses uploaded successfully."}


@app.post("/finalize")
async def finalize_project(

    guid: str = Form(...),
    project_name: str = Form(...)
):
    project_dir = os.path.join(BASE_DIR, guid)
    if not os.path.exists(project_dir):
        raise HTTPException(status_code=404, detail="Project GUID not found.")

    intrinsic_path = os.path.join(project_dir, "intrinsics.txt")

    try:
        client = await Client.connect("temporal-worker:7233")

        # ✅ Create a unique workflow ID using UUID
        workflow_id = f"workflow-{guid}-{uuid.uuid4()}"

        handle = await client.start_workflow(
            PointCloudWorkflow.run,
            args=[project_dir, intrinsic_path],
            id=workflow_id,
            task_queue="pointcloud-queue"
        )

        return {
            "workflow_id": handle.id,
            "run_id": handle.first_execution_run_id,
            "status": "Processing started"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start workflow: {str(e)}")


@app.get("/workflow-status")
async def get_workflow_status(
    workflow_id: str = Query(...),
    run_id: str = Query(...)
):
    try:
        client = await Client.connect("temporal-worker:7233")
        handle = client.get_workflow_handle(workflow_id, run_id=run_id)

        info = await handle.describe()

        return {
            "workflow_id": workflow_id,
            "run_id": run_id,
            "status": info.status.name  # e.g., RUNNING, COMPLETED, FAILED
        }

    except RPCError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=404, detail=f"Workflow '{workflow_id}' with run '{run_id}' not found.")
        raise HTTPException(
            status_code=500, detail=f"Temporal RPC error: {str(e)}")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/upload-frame-count")
async def get_project_data(guid: str):
    project_dir = os.path.join(BASE_DIR, guid)
    rgb_dir = os.path.join(project_dir, "rgb")
    if not os.path.exists(rgb_dir):
        return {"upload_frame_count": 0}
    return {"upload_frame_count": len(os.listdir(rgb_dir))}


@app.delete("/delete-project")
async def delete_project(guid: str):
    project_dir = os.path.join(BASE_DIR, guid)
    shutil.rmtree(project_dir)
    logger.info("Deleted project %s", guid)
    return {"message": "Project deleted successfully."}


@app.get("/hello-world")
async def hello_world():
    return {"message": "Hello, World!"}


@app.get("/get_point_cloud")
async def get_point_cloud(guid: str):
    ply_path = os.path.join(BASE_DIR, guid, "point_cloud.ply")
    if os.path.exists(ply_path):
        return FileResponse(ply_path, media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Point cloud file not found.")


@app.get("/visualize", response_class=HTMLResponse)
async def visualize_point_cloud(guid: str):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>3D Point Cloud Viewer</title>
        <script src="https://cdn.jsdelivr.net/npm/three@0.125.2/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.125.2/examples/js/loaders/PLYLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.125.2/examples/js/controls/OrbitControls.js"></script>
        <style>
            body {{ margin: 0; overflow: hidden; }}
            canvas {{ display: block; }}
        </style>
    </head>
    <body>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                camera.position.set(0, 0, 5);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.body.appendChild(renderer.domElement);
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.screenSpacePanning = false;
                controls.maxDistance = 100;
                controls.minDistance = 0.1;
                const light = new THREE.PointLight(0xffffff, 1);
                light.position.set(10, 10, 10);
                scene.add(light);
                scene.add(new THREE.AmbientLight(0x404040));
                const loader = new THREE.PLYLoader();
                loader.load('/get_point_cloud?guid={guid}', function (geometry) {{
                    geometry.computeVertexNormals();
                    const material = new THREE.PointsMaterial({{ size: 0.05, vertexColors: true }});
                    const points = new THREE.Points(geometry, material);
                    scene.add(points);
                    const center = new THREE.Vector3();
                    geometry.computeBoundingBox();
                    geometry.boundingBox.getCenter(center);
                    points.position.sub(center);
                }});
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                window.addEventListener("resize", () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/get_mesh")
async def get_mesh(guid: str, format: str = "obj"):
    import traceback

    ply_path = os.path.join(BASE_DIR, guid, "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise HTTPException(
            status_code=404, detail="Point cloud file not found.")

    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.estimate_normals()

        try:
            # Fast method
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha=0.03)
        except Exception as alpha_err:
            print("[⚠️ Fallback] Alpha shape failed. Using Poisson...")
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=7)

        mesh.compute_vertex_normals()

        if format == "obj":
            output_path = os.path.join(BASE_DIR, guid, "mesh.obj")
            o3d.io.write_triangle_mesh(output_path, mesh)
            return FileResponse(output_path, media_type="text/plain")
        elif format == "glb":
            output_path = os.path.join(BASE_DIR, guid, "mesh.glb")
            o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
            return FileResponse(output_path, media_type="model/gltf-binary")
        else:
            raise HTTPException(
                status_code=400, detail="Unsupported format. Use 'obj' or 'glb'.")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Mesh generation failed: {str(e)}")
