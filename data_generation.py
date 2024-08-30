import numpy as np
import json
import sys
sys.path.append('.')
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from prompts.prompts import scanrefer_prompt
import string
from snap_one_sample import *


def get_annotated_camera(file_path, scene_id, object_id, ann_id):
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
            return entry['camera']

    # Return None if no matching entry is found
    print("No match found!!!")
    return None


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

for split in ["train", "val"]:
    count = [0] * 100
    annos = json.load(open(f"/home/wa285/rds/hpc-work/Thesis/Thesis-Chat-3D-v2/annotations_with_position/scanrefer/ScanRefer_filtered_{split}.json", "r"))
    annos = sorted(annos, key=lambda p: f"{p['scene_id']}_{int(p['object_id']):03}")
    new_annos = []

    instance_attribute_file = f"/home/wa285/rds/hpc-work/Thesis/Thesis-Chat-3D-v2/annotations_with_position/scannet_{segmentor}_{split}_attributes{version}.pt"
    scannet_attribute_file = f"/home/wa285/rds/hpc-work/Thesis/Thesis-Chat-3D-v2/annotations_with_position/scannet_{split}_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file)
    scannet_attrs = torch.load(scannet_attribute_file)

    iou25_count = 0
    iou50_count = 0
    for i, anno in tqdm(enumerate(annos)):
        scene_id = anno['scene_id']
        obj_id = int(anno['object_id'])
        desc = anno['description']

        #read the camera annos file and extract camera extrinsic parameters
        camera_annotation_file_path=f"/home/wa285/rds/hpc-work/Thesis/annotated_cameras/{scene_id}.anns.json"
        annotated_camera=get_annotated_camera(camera_annotation_file_path, scene_id, anno['object_id'], anno['ann_id'])

        if desc[-1] in string.punctuation:
            desc = desc[:-1]
        prompt = random.choice(scanrefer_prompt).replace('<description>', desc)
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        max_iou, max_id = -1, -1
        for pred_id in range(instance_num):
            pred_locs = instance_locs[pred_id].tolist()
            gt_locs = scannet_locs[obj_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)
            if iou > max_iou:
                max_iou = iou
                max_id = pred_id
        if max_iou >= 0.25:
            iou25_count += 1
        if max_iou >= 0.5:
            iou50_count += 1
        count[max_id] += 1
        if split == "train":
            if max_iou >= args.train_iou_thres:
                save_path=f"/home/wa285/rds/hpc-work/Thesis/annotated_photos/{split}/{scene_id}_{anno['object_id']}_{anno['ann_id']}.png"
                capture_one_image(scene_id, annotated_camera, instance_attribute_file, 800, 600, save_path)
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": max_id,
                    "caption": f"<OBJ{max_id:03}>.",
                    "prompt": prompt,
                    "camera": annotated_camera,
                    "image_path": save_path,
                    "scanrefer_description": desc
                })
        else:
            save_path=f"/home/wa285/rds/hpc-work/Thesis/annotated_photos/{split}/{scene_id}_{anno['object_id']}_{anno['ann_id']}.png"
            capture_one_image(scene_id, annotated_camera, instance_attribute_file, 800, 600, save_path)
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "ref_captions": [f"<OBJ{max_id:03}>."],
                "prompt": prompt,
                "camera": annotated_camera,
                "image_path": save_path,
                "scanrefer_description": desc
            })

    print(len(new_annos))
    print(count)


    with open(f"scanrefer_new_anno_{split}.json", "w") as f:
        json.dump(new_annos, f, indent=4)