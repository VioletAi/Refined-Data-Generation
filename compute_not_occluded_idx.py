# import numpy as np
# import torch
# from pytorch3d.renderer import (
#     PerspectiveCameras)

# def construct_bbox_corners(center, box_size):
#     """ Calculate the 8 corners of the 3D bounding box """
#     sx, sy, sz = box_size
#     x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
#     y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
#     z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
#     corners_3d = np.vstack([x_corners, y_corners, z_corners])
#     corners_3d[0, :] = corners_3d[0, :] + center[0]
#     corners_3d[1, :] = corners_3d[1, :] + center[1]
#     corners_3d[2, :] = corners_3d[2, :] + center[2]
#     corners_3d = np.transpose(corners_3d)

#     return corners_3d

# def calculate_occlusion_rate(box1, box2):
#     """ Calculate Intersection over Union (IoU) between two bounding boxes """
#     xA = max(box1[0], box2[0])
#     yA = max(box1[1], box2[1])
#     xB = min(box1[2], box2[2])
#     yB = min(box1[3], box2[3])

#     inter_width = max(0, xB - xA)
#     inter_height = max(0, yB - yA)
#     inter_area = inter_width * inter_height

#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     visible_portion = inter_area / float(box1_area)
#     return visible_portion

# def get_2d_bounding_box(corners_2d):
#     """ Get 2D bounding box from 2D corners """
#     x_min, y_min = torch.min(corners_2d, dim=0)[0]
#     x_max, y_max = torch.max(corners_2d, dim=0)[0]
#     return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]

# def check_visibility(objects, camera, image_width, image_height):
#     points_3d = objects[:, :3]
#     center_in_camera_coordinate = camera.get_world_to_view_transform().transform_points(points_3d)
#     bbox_sizes = objects[:, 3:]

#     # Calculate depths of objects from the camera
#     depths_in_camera_view = center_in_camera_coordinate[:, 2]
#     # print("Depths:", depths_in_camera_view)

#     # Sort objects based on their depth (closest first)
#     sorted_indices = torch.argsort(depths_in_camera_view)
#     # print("Sorted indices based on depth:", sorted_indices)
#     visible_indices = []

#     # Loop through each object and check for occlusion and image bounds
#     for i in sorted_indices:
#         # Skip objects behind the camera
#         if depths_in_camera_view[i] < 1 :
#             continue
#         center = points_3d[i].cpu().numpy()
#         size = bbox_sizes[i].cpu().numpy()

#         # Calculate 3D corners and transform to 2D image coordinates
#         corners_3d = construct_bbox_corners(center, size)
#         corners_3d = torch.tensor(corners_3d, dtype=torch.float32, device=points_3d.device)

#         corners_2d = camera.transform_points_screen(corners_3d)[:,:2]
#         # Check if any corner is within image bounds

#         ################3

#         projected_centers = camera.transform_points_screen(points_3d[i].unsqueeze(0))[0]
        
#         x, y, z = int(projected_centers[0]), int(projected_centers[1]), projected_centers[2]

#         # Check if the object is in front of the camera
#         # if z < 0:
#         #     continue  # Skip objects behind the camera

#         # Adjust coordinates if necessary
#         if x < 0 or x >= image_width or y < 0 or y >= image_height:
#             continue  # Skip drawing labels outside the image bounds


#         # if depths_in_camera_view[i] < 0 and not((corners_2d[:, 0] >= 0) and (corners_2d[:, 0] < image_width) and (corners_2d[:, 1] >= 0) and (corners_2d[:, 1] < image_height)):
#         #     continue
#         print("inside here: i: ",i,"depths: ",depths_in_camera_view[i])
#         bbox_2d = get_2d_bounding_box(corners_2d)
#         print("i",i,bbox_2d)

#         # Check occlusion percentage with previously visible objects
#         occluded = False
#         for vi in visible_indices:
#             vi_center = points_3d[vi].cpu().numpy()
#             vi_size = bbox_sizes[vi].cpu().numpy()
#             vi_corners_3d = construct_bbox_corners(vi_center, vi_size)
#             vi_corners_3d = torch.tensor(vi_corners_3d, dtype=torch.float32, device=points_3d.device)
#             vi_corners_2d = camera.transform_points_screen(vi_corners_3d)[:,:2]
#             vi_bbox_2d = get_2d_bounding_box(vi_corners_2d)
#             occlusion_percentatge = calculate_occlusion_rate(bbox_2d, vi_bbox_2d)
#             if occlusion_percentatge >=0.7:
#                 print(f"Object {i} is occluded by object {vi} with IoU {occlusion_percentatge}.")
#                 occluded = True
#                 break

#         if not occluded:
#             visible_indices.append(i.item())
#     print("visible indices",visible_indices)
#     return visible_indices


import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.transforms import Rotate, Translate

def check_visibility(objects, camera, image_width, image_height):
    """
    Check the visibility of 3D bounding boxes in the camera view.
    
    Parameters:
        objects (Tensor): Tensor of shape (N, 6) containing center coordinates (x, y, z)
                          and dimensions (width, height, length) of N bounding boxes.
        camera (PerspectiveCameras): Pytorch3D camera object.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
    
    Returns:
        visible_indices (List[int]): List of indices of visible objects.
    """

    # Initialize lists for holding meshes and bounding box indices
    meshes = []
    object_indices = []
    
    for i, obj in enumerate(objects):
        # Unpack the object's parameters
        x, y, z, width, height, length = obj
        
        # Compute the 8 corners of the bounding box
        corners = torch.tensor([
            [x - width / 2, y - height / 2, z - length / 2],
            [x + width / 2, y - height / 2, z - length / 2],
            [x + width / 2, y + height / 2, z - length / 2],
            [x - width / 2, y + height / 2, z - length / 2],
            [x - width / 2, y - height / 2, z + length / 2],
            [x + width / 2, y - height / 2, z + length / 2],
            [x + width / 2, y + height / 2, z + length / 2],
            [x - width / 2, y + height / 2, z + length / 2]
        ])
        
        # Define faces (triangles) for the bounding box (12 triangles, 6 faces)
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # front face
            [4, 5, 6], [4, 6, 7],  # back face
            [0, 1, 5], [0, 5, 4],  # bottom face
            [2, 3, 7], [2, 7, 6],  # top face
            [0, 3, 7], [0, 7, 4],  # left face
            [1, 2, 6], [1, 6, 5],  # right face
        ])
        
        # Create a mesh from the bounding box corners and faces
        mesh = Meshes(verts=[corners], faces=[faces])
        meshes.append(mesh)
        object_indices.append(i)

    # Combine all meshes into a single Meshes object
    combined_mesh = Meshes(verts=[m.verts_packed() for m in meshes],
                           faces=[m.faces_packed() for m in meshes])
    
    # Define the rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(image_width, image_height),
        blur_radius=0.0,
        faces_per_pixel=1
    )
    
    # Create a rasterizer
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=raster_settings
    )
    
    # Rasterize the meshes to obtain fragments
    fragments = rasterizer(combined_mesh)
    
    # Get the depth of each pixel
    depth_map = fragments.zbuf.squeeze(-1)
    
    # Check visibility: an object is visible if any of its pixels is the closest one to the camera
    visible_indices = []
    for i, depth in enumerate(depth_map):
        if torch.any(depth < raster_settings.faces_per_pixel):
            visible_indices.append(object_indices[i])
    
    return visible_indices
