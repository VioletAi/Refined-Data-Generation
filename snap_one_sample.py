import sys
import json
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams)

from PIL import Image, ImageDraw, ImageFont

import numpy as np

from compute_not_occluded_idx import *

import os
sys.path.append(os.path.join(os.getcwd()))

SCANNET_ROOT = "/home/wa285/rds/hpc-work/Thesis/dataset/scannet/raw/scans"
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply")  # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt")  # scene_id, scene_id

def lookat(center, target, up):
    """
    From: LAR-Look-Around-and-Refer
    https://github.com/eslambakr/LAR-Look-Around-and-Refer
    https://github.com/isl-org/Open3D/issues/2338
    https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    https://www.youtube.com/watch?v=G6skrOtJtbM
    f: forward
    s: right
    u: up
    """
    f = target - center
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = -s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    t = np.matmul(-m[:3, :3], center)
    m[:3, 3] = t

    return m


def get_extrinsic(camera_location,target_location):
    camera_location=np.array(camera_location)
    target_location=np.array(target_location)

    up_vector = np.array([0, 0, -1])
    pose_matrix = lookat(camera_location, target_location, up_vector)
    pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
    return pose_matrix_calibrated


def render_mesh(pose, intrin_path, image_width, image_height, mesh, name, device):
    """
    Given the mesh in PyTorch3D format, render images with a defined pose, intrinsic matrix, and image dimensions
    """

    background_color = (1.0, 1.0, 1.0)
    intrinsic_matrix = torch.zeros([4, 4])
    intrinsic_matrix[3, 3] = 1

    intrinsic_matrix_load = intrin_path
    intrinsic_matrix_load_torch = torch.from_numpy(intrinsic_matrix_load)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch
    extrinsic_load = pose
    camera_to_world = torch.from_numpy(extrinsic_load)
    world_to_camera = torch.inverse(camera_to_world)
    fx, fy, cx, cy = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    width, height = image_width, image_height
    rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
  # print("translation vector",translation_vector)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    camera = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=rotation_matrix,
        T=translation_vector,
        image_size=torch.tensor([[height, width]]),
        in_ndc=False,
        device=device,
    )
    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(
            blend_params=BlendParams(background_color=background_color),
            device=lights.device,
            cameras=camera,
            lights=lights,
        ),
    )
    rendered_image = renderer(mesh)
    # print("rendered image shape",rendered_image.shape)
    rendered_image = rendered_image[0].cpu().numpy()

    color = rendered_image[..., :3]
    # print("color: ",color[0])
    color_image = Image.fromarray((color * 255).astype(np.uint8))
    # display(color_image)
    # color_image.save(name)
    return color_image,camera


def get_description_from_json(file_path, scene_id, object_id, ann_id):
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Assuming the JSON structure contains a list of entries
    for entry in data:
        # Check for matching scene_id, object_id, and ann_id
        if (entry['scene_id'] == scene_id and
            entry['object_id'] == object_id and
            entry['ann_id'] == ann_id):
            # Return the description if a match is found
            return entry['description']

    # Return None if no matching entry is found
    return None




def revert_alignment(scene_id, original_coords):

    # Calculate the inverse of the alignment matrix

    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array(
                [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    # Convert the numpy array to a PyTorch tensor
    axis_align_matrix = torch.tensor(axis_align_matrix, dtype=torch.float32, device=original_coords.device)


    # Calculate the inverse of the alignment matrix
    axis_align_matrix = torch.linalg.inv(axis_align_matrix)


    # Prepare aligned coordinates by adding a column of ones for homogeneous coordinates
    ones = torch.ones((original_coords.shape[0], 1), device=original_coords.device)
    homogeneous_coords = torch.cat((original_coords, ones), dim=1)

    # Apply the inverse transformation
    original_coords_homogeneous = torch.matmul(homogeneous_coords, axis_align_matrix.t())

    # Convert back from homogeneous to 3D coordinates
    original_coords = original_coords_homogeneous[:, :3]

    return original_coords


def capture_one_image(scene_id, annotated_camera, instance_attribute_file, image_width, image_height, save_path):
    device = torch.device("cuda")
    instance_attrs = torch.load(instance_attribute_file)

    mesh_file = f"/home/wa285/rds/hpc-work/Thesis/dataset/scannet/raw/scans/{scene_id}/{scene_id}_vh_clean_2.ply"

    file_path = f'/home/wa285/rds/hpc-work/Thesis/download_intinsic_matrix/output/{scene_id}/intrinsic/intrinsic_depth.txt'

    camera_position = annotated_camera["position"]
    lookat_point = annotated_camera["lookat"]

    # get extrinsic matrix
    view_matrix = get_extrinsic(camera_position, lookat_point)

    # Load the file as a numpy array
    depth_intrinsic = np.loadtxt(file_path)

    # device = "cpu"

    pt3d_io = IO()
    mesh = pt3d_io.load_mesh(mesh_file, device=device)

    image,camera =render_mesh(
            view_matrix,
            depth_intrinsic[:3, :3],
            image_width,
            image_height,
            mesh,
            f"./image_rendered_angle.png", device=device)

    instance_attributes=instance_attrs[scene_id]

    locs = instance_attributes['locs']
    objects = instance_attributes['objects']
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('OpenSans-Bold.ttf', 15)  
  

    depth_intrinsic=torch.tensor(depth_intrinsic, dtype=torch.float32, device=device)
    view_matrix=torch.tensor(view_matrix, dtype=torch.float32, device=device)
    locs = locs.clone().detach().to(device).float()
    locs[:,:3] = revert_alignment(scene_id, locs[:,:3])

    visible_idx=check_visibility(locs, camera, image_width, image_height)

    for idx, loc in enumerate(locs):

        if idx in visible_idx:
            if objects[idx] in ["ceiling", "floor", "object", "item"]:
                continue

            center = loc[:3].to(device).unsqueeze(0)

            projected_centers = camera.transform_points_screen(center)[0]
            
            x, y, z = int(projected_centers[0]), int(projected_centers[1]), projected_centers[2]

            # Check if the object is in front of the camera
            # if z < 0:
            #     continue  # Skip objects behind the camera

            # Adjust coordinates if necessary
            # if x < 0 or x >= image_width or y < 0 or y >= image_height:
            #     continue  # Skip drawing labels outside the image bounds

            # draw.text((x, y), f'OBJ{idx} {objects[idx]}', fill="red")
            # draw.text((x, y), f'OBJ{idx:03}', fill="red")

             # Calculate text size and adjust to center the text
            text = f'OBJ{idx:03}'
            #  = draw.textsize(text, font=font)
            _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)
            text_x = x - text_width // 2
            text_y = y - text_height // 2

            # Calculate rectangle position
            rect_x0 = text_x
            rect_y0 = text_y
            rect_x1 = text_x + text_width
            rect_y1 = text_y + text_height

            # Draw a white rectangle behind the text
            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="white")

            # Draw text in red, bold, and centered
            
            draw.text((text_x, text_y), text, font=font, fill="red")

    image.save(save_path)


if __name__ == "__main__":
    data_idx=20

    with open("/home/wa285/rds/hpc-work/Thesis/annotated_cameras/scene0000_00.anns.json", 'r') as file:
        parsed_data = json.load(file)

    instance_attribute_file = f"/home/wa285/rds/hpc-work/Thesis/Thesis-Chat-3D-v2/annotations/scannet_mask3d_train_attributes.pt"
    
    test_annotated_camera=parsed_data[data_idx]['camera']
    print(parsed_data[data_idx])
    file_path = '/home/wa285/rds/hpc-work/Thesis/download_intinsic_matrix/output/scene0000_00/intrinsic/intrinsic_depth.txt'
    save_path="hello.png"
    capture_one_image("scene0000_00", test_annotated_camera, instance_attribute_file, 800, 600, save_path)
 
