# RGBD Point Cloud API

This is a FastAPI application for handling RGBD data and generating point clouds.

## Endpoints

1. **Create Project**
   - **POST** `/projects`
   - Request Body: `{ "name": "Project Name", "coordinates": [x, y, z] }`
   - Response: Unique project GUID.

2. **Upload RGBD Data**
   - **POST** `/projects/{guid}/upload`
   - Form Data: RGB files, Depth files, and a text file.
   - Response