import sys
sys.path.append('/home/yifanliu/univlg')
from feature_map_tools.qwen_multicam import Qwen2_5_Projected3D
from qwen_vl_utils import process_vision_info
from univlg.modeling.backproject.backproject import backprojector, voxelization
from torch_scatter import scatter_mean, scatter_min
from torch.nn import functional as F

from PIL import Image
import ipdb
import numpy as np
import os
import wandb
import torch
import requests
from transformers import AutoProcessor, AutoTokenizer


st = ipdb.set_trace


def point_sampling(points, poses, intrinsics, H, W):
    """
    Input:
        points: B, num_query, 3
        poses: B, num_cam, 4, 4
        intrinsics: B, num_cam, 4, 4
    Output:
        points_cam: num_cam, B, num_query, 2
        valid_mask: num_cam, B, num_query, 2
    """
    # construct the camera projection matrix
    assert points.dtype == torch.float32
    assert poses.dtype == torch.float32
    depth2img = intrinsics @ poses.inverse()

    points = torch.cat(
        (points, torch.ones_like(points[..., :1])), -1)

    B, num_query = points.shape[:2]
    num_cam = depth2img.shape[1]
    
    points = points[:, None].repeat(1, num_cam, 1, 1)

    # points = points.view(
    #     D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    # depth2img = depth2img.view(
    #     1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    points_cam = torch.einsum('bnij,bnmj->bnmi', depth2img, points)
    
    # points_cam = torch.matmul(depth2img.to(torch.float32),
                                        # points.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    valid_mask = (points_cam[..., 2:3] > eps)
    points_cam = points_cam[..., 0:2] / torch.maximum(
        points_cam[..., 2:3], torch.ones_like(points_cam[..., 2:3]) * eps)

    points_cam[..., 0] /= W
    points_cam[..., 1] /= H

    valid_mask = (valid_mask & (points_cam[..., 1:2] > 0.0)
                & (points_cam[..., 1:2] < 1.0)
                & (points_cam[..., 0:1] < 1.0)
                & (points_cam[..., 0:1] > 0.0))
    valid_mask = torch.nan_to_num(valid_mask)
    return points_cam, valid_mask.squeeze(-1)

    # D, B, num_cam, num_query, 2 -> num_cam, B, num_query, D, 2
    # points_cam = points_cam.permute(2, 1, 3, 0, 4)
    # valid_mask = valid_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    # return points_cam, valid_mask
    

def run_qwen(
    model, processor, rgb_images_torch, best_camera_idx=0, points_cam=None, valid_mask=None, p2v=None
):
    """
    Run the Qwen model with the best camera frame.
    
    Args:
        model: The Qwen model
        processor: The processor for the model
        rgb_images_torch: Torch tensor of all RGB images [B, H, W, 3]
        best_camera_idx: Index of the best camera to use (default: 0)
        points_cam: Camera-space points [B, num_points, 2]
        valid_mask: Visibility mask for points
        p2v: Point-to-voxel mapping
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the room in detail."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # get all image features from qwen
    if points_cam is not None:
        image_inputs = processor.image_processor(
            images=rgb_images_torch,
            do_rescale=True, 
            return_tensors="pt"
        )
        
        image_inputs = image_inputs.to(model.device)
        image_embeds = model.visual(
            image_inputs['pixel_values'], 
            grid_thw=image_inputs['image_grid_thw']
        ).to(model.device)

        image_embeds = scatter_mean(image_embeds, p2v[0], 0).to(model.device)
        
    # Use the best camera's image instead of hardcoding the first one
    inputs = processor(
        text=[text],
        images=rgb_images_torch[best_camera_idx][None],  # Use best camera index
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(model.device)
    print(f"Using camera {best_camera_idx} for image input")
    
    if points_cam is not None:
        inputs['points_cam'] = points_cam.to(model.device)
        inputs['valid_mask'] = valid_mask.to(model.device)
        inputs['all_image_embeds'] = image_embeds.to(model.device)
    
    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text


def run_qwen_for_boundingbox(
    model, processor, rgb_images_torch, target_sentence, best_camera_idx=0, points_cam=None, valid_mask=None, p2v=None
):
    """
    Run the Qwen model with the best camera frame.
    
    Args:
        model: The Qwen model
        processor: The processor for the model
        rgb_images_torch: Torch tensor of all RGB images [B, H, W, 3]
        best_camera_idx: Index of the best camera to use (default: 0)
        points_cam: Camera-space points [B, num_points, 2]
        valid_mask: Visibility mask for points
        p2v: Point-to-voxel mapping
    """
    
    # SYSTEM_PROMPT = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n\nIMPORTANT:\n\n- The Assistant must always include a <think> section first, where it reasons step by step.\n- After the <think> section, the Assistant provides the final answer inside an <answer> section.\n- Both sections are required in every response. Do not skip the <think> section.\n- The <answer> must contain only a single string in json format bounding box.\n- Your task is to help the user identify the precise bounding box of a specific object in the room based on a description.\n- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.\n- The final output should be the single most precise bounding box for the requested element.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer. Outline the position of the object described as \'{target_sentence}\' and output the coordinates in JSON format."
    
    SYSTEM_PROMPT =f"""Outline a point on the object described as \'{target_sentence}\' and output the coordinates in (x, y), width first, with two integer numbers. """
    
    
    # SYSTEM_PROMPT =f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n\nIMPORTANT:\n\n- The Assistant must always include a <think> section first, where it reasons step by step.\n- After the <think> section, the Assistant provides the final answer inside an <answer> section.\n- Both sections are required in every response. Do not skip the <think> section.\n- The <answer> must contain only a single string in the format (x, y) with the coordinates.\n- Your task is to help the user identify the precise coordinates (x, y) of a specific area, element, or object on the screen based on a description.\n- Your response should aim to point to the center or a representative point within the described area/element/object as accurately as possible.\n- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.\n- The final output should be the single most precise coordinate for the requested element.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer. The question is: Outline a point on the object described as \'{target_sentence}\' and output the coordinates (x, y) with two integer numbers. """
    
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{SYSTEM_PROMPT} "},
            ],
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": f"Outline the position of the object described as \'{target_sentence}\' and output the coordinates in JSON format."},
    #         ],
    #     }
    # ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # text = text.replace("system\nYou are a helpful assistant.", SYSTEM_PROMPT)
    # import pdb;pdb.set_trace()
    # get all image features from qwen
    if points_cam is not None:
        image_inputs = processor.image_processor(
            images=rgb_images_torch,
            do_rescale=True, 
            return_tensors="pt"
        )
        
        image_inputs = image_inputs.to(model.device)
        image_embeds = model.visual(
            image_inputs['pixel_values'], 
            grid_thw=image_inputs['image_grid_thw']
        ).to(model.device)

        image_embeds = scatter_mean(image_embeds, p2v[0], 0).to(model.device)
        
    # Use the best camera's image instead of hardcoding the first one
    inputs = processor(
        text=[text],
        images=rgb_images_torch[best_camera_idx][None],  # Use best camera index
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(model.device)
    print(f"Using camera {best_camera_idx} for image input")
    
    if points_cam is not None:
        inputs['points_cam'] = points_cam.to(model.device)
        inputs['valid_mask'] = valid_mask.to(model.device)
        inputs['all_image_embeds'] = image_embeds.to(model.device)
    
    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text


def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def load_image(path):
    image = Image.open(path)
    return np.array(image)


def load_scannet(scene_name, model=None, processor=None):
    DATA_PATH = f'/data/user_data/ayushj2/SEMSEG_100k/frames_square_highres/{scene_name}'
    RGB_PATH = f'{DATA_PATH}/color'
    DEPTH_PATH = f'{DATA_PATH}/depth'  
    POSE_PATH = f'{DATA_PATH}/pose'
    MAX_INDEX = 1080 # take up to this index of images
    SKIP = 20 # take one image of every SKIP to speed up the processing
    MAX_NUM_IMAGES = 10
    
    rgb_images = [load_image(os.path.join(RGB_PATH, f'{i}.png')) for i in range(0, MAX_INDEX, SKIP)][:MAX_NUM_IMAGES]
    depth_images = [load_image(os.path.join(DEPTH_PATH, f'{i}.png'))for i in range(0, MAX_INDEX, SKIP)][:MAX_NUM_IMAGES]
    
    intrinsic_depth = load_matrix_from_txt(os.path.join(DATA_PATH, 'intrinsic', 'intrinsic_depth.txt')) # 480 X 640
    poses = [load_matrix_from_txt(os.path.join(POSE_PATH, f'{i}.txt')) for i in range(0, MAX_INDEX, SKIP)][:MAX_NUM_IMAGES]
    

    # build 3D point cloud
    rgb_images_torch = torch.from_numpy(np.array(rgb_images)).float().cuda()
    depth_images_torch = torch.from_numpy(np.array(depth_images)).float().cuda() / 1000.0
    intrinsics_torch = torch.from_numpy(intrinsic_depth).float().cuda()[None].repeat(len(rgb_images_torch), 1, 1)
    poses_torch = torch.from_numpy(np.array(poses)).float().cuda()
    
    B, H, W = depth_images_torch.shape
    world_coords = backprojector(
            [[B, 3, H, W]], depth_images_torch[:, None], poses_torch[:, None], intrinsics_torch[:, None])[0]
    # world_coords torch.Size([10, 1, 480, 640, 3])
    
    pointmap = world_coords[0]
    
    # resize to be 1/28 of the original size
    pointmap = F.interpolate(pointmap.flatten(0, 1).permute(0, 3, 1, 2), size=(17, 23), mode='nearest').permute(0, 2, 3, 1).reshape(B, -1, 17, 23, 3) # torch.Size([10, 1, 17, 23, 3])
    rgb_images_torch_resized = F.interpolate(rgb_images_torch.permute(0, 3, 2, 1), size=(17, 23), mode='bilinear').permute(0, 3, 2, 1)
    
    # voxelize the point cloud
    p2v = voxelization(pointmap.reshape(-1, 3)[None], 0.02) # [1, 3910] denoting the voxel index of each point 10*17*23
    voxelized_pc = scatter_mean(pointmap.reshape(-1, 3)[None], p2v, 1) # B, N, 3 # 1, 3367, 3 (reduced)
    
    rgb_voxelized_pc = scatter_mean(rgb_images_torch_resized.reshape(-1, 3)[None], p2v, 1) # B, N, 3
    
    rgb_xyz_pc = torch.cat((voxelized_pc, rgb_voxelized_pc), -1)
    # (Pdb) p rgb_xyz_pc.shape
    # torch.Size([1, 3367, 6]) xyz + rgb
    
    # WARNING: it only gets the pc for the first image!
    image_pc = pointmap[0].reshape(-1, 3)
    image_rgb = rgb_images_torch_resized[0].reshape(-1, 3)
    image_rgb_xyz_pc = torch.cat((image_pc, image_rgb), -1) # [391,6] xyz + rgb
    
    # project the 3D point cloud to a new intrinsic
    # pointmap = pointmap[0][None]
    # rgb_images_torch = rgb_images_torch[0][None]
    # p2v = torch.arange((image_pc.shape[0]))[None].to(p2v)
    
    
    # points_cam, valid_mask = point_sampling(
    #     image_pc[None],
    #     poses_torch[0][None, None],
    #     intrinsics_torch[0][None, None],
    #     H, W
    # )
    
    # y = torch.arange(0, 17).float().cuda() / 17
    # x = torch.arange(0, 23).float().cuda() / 23
    # y, x = torch.meshgrid(y, x)
    # points_cam = torch.stack((x, y), -1).reshape(-1, 2)[None, None]
    # valid_mask = torch.ones_like(valid_mask)
    
    
    # voxelized_pc: torch.Size([1, 3367, 3])
    # I should select best camera here!
    # Select the best camera based on point visibility
    best_camera_idx, points_cam, valid_mask = select_best_camera(
        voxelized_pc, poses_torch, intrinsics_torch, H, W
    )

    # visualize all the points which are visible
    visible_rgb_xyz_pc = rgb_xyz_pc[valid_mask[0]]  # Now using the best camera's mask
    
    print (f"Best camera index: {best_camera_idx}")
    print (f"Visible points: {visible_rgb_xyz_pc.shape[0]}")
    
    zoomed_intrinsics = intrinsics_torch[best_camera_idx].clone()
    zoomed_intrinsics[0, 0] /= 2.0  # fx /= 2
    zoomed_intrinsics[1, 1] /= 2.0  # fy /= 2
    
    # Project using zoomed intrinsics
    zoomed_points_cam, zoomed_valid_mask = point_sampling(
        voxelized_pc,
        poses_torch[best_camera_idx][None, None],
        zoomed_intrinsics[None, None],
        H, W
    )
    
    zoomed_visible_count = zoomed_valid_mask.sum().item()
    print(f"Zoomed-out camera {best_camera_idx}: {zoomed_visible_count} visible points")
    print(f"Zoom improves visibility by: {zoomed_visible_count - visible_rgb_xyz_pc.shape[0]} points")
    
    
    # points_cam, valid_mask = point_sampling(voxelized_pc, poses_torch[0][None, None], intrinsics_torch[0][None, None], H, W)
    
    # # visualize all the points which are visible
    # visible_rgb_xyz_pc = rgb_xyz_pc[valid_mask[0]] # 1160,6
    
    output_text = run_qwen(
    model, processor, rgb_images_torch, best_camera_idx, zoomed_points_cam, zoomed_valid_mask, p2v
)

    output_text_image_only = run_qwen(model, processor, rgb_images_torch, best_camera_idx)
    
    output_text = f"All images text: {output_text[0]}, \n\n Image only text: {output_text_image_only[0]}"
    
    wandb.log({
        "visibe pc": wandb.Object3D(visible_rgb_xyz_pc.cpu().numpy()),
        "whole pc": wandb.Object3D(rgb_xyz_pc[0].cpu().numpy()),
        "image pc": wandb.Object3D(image_rgb_xyz_pc.cpu().numpy()),
        "image": wandb.Image(rgb_images[0], caption=output_text),
        "all_images": wandb.Image(np.concatenate(rgb_images, 1), caption=output_text),
        # "output_text": wandb.Table(data=[output_text], columns=["output_text"])
    })
    
    

    # render the 2D positions of the 3D points

def select_best_camera(voxelized_pc, poses_torch, intrinsics_torch, H, W):
    """
    Selects the camera that has the most visible points.
    
    Args:
        voxelized_pc: Point cloud tensor [B, N, 3]
        poses_torch: Camera poses [num_cameras, 4, 4]
        intrinsics_torch: Camera intrinsics [num_cameras, 3, 3]
        H, W: Image height and width
    
    Returns:
        best_camera_idx: Index of the best camera
        points_cam: Projected points for the best camera
        valid_mask: Valid mask for the best camera
    """
    num_cameras = poses_torch.shape[0]
    best_visible_count = -1
    best_camera_idx = 0
    best_points_cam = None
    best_valid_mask = None
    
    # For each camera
    for camera_idx in range(num_cameras):
        # Project the points to this camera view
        points_cam_i, valid_mask_i = point_sampling(
            voxelized_pc, 
            poses_torch[camera_idx][None, None],
            intrinsics_torch[camera_idx][None, None],
            H, W
        )
        
        # Count visible points
        visible_count = valid_mask_i.sum().item()
        
        print(f"Camera {camera_idx}: {visible_count} visible points")
        
        # Update if this camera is better
        if visible_count > best_visible_count:
            best_visible_count = visible_count
            best_camera_idx = camera_idx
            best_points_cam = points_cam_i
            best_valid_mask = valid_mask_i
    
    print(f"Selected camera {best_camera_idx} with {best_visible_count} visible points")
    return best_camera_idx, best_points_cam, best_valid_mask

if __name__=="__main__":
    # run_qwen()
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_Projected3D.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    
    wandb.init(project="qwen_vl")
    scene_names = ["scene0191_00", "scene0087_00", "scene0172_00", "scene0631_01"]
    for scene_name in scene_names:
        load_scannet(scene_name, model=model, processor=processor)
    st()
    print("Done")