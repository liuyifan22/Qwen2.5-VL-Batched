# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Tuple
import copy
import random
from datetime import datetime
import os
import detectron2.utils.comm as comm
import ipdb
import numpy as np
import torch
import re
import json
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import wandb
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.comm import get_world_size
from univlg.data_video.sentence_utils import (
    convert_grounding_to_od_logits,
    convert_grounding_to_od_logits_ref
)
from univlg.modeling.transformer_decoder.video_mask2former_transformer_decoder import (
    build_transformer_decoder,
)
from univlg.modeling.visual_backbone import UniVLGVisualBackbone
from univlg.global_vars import (
    LEARNING_MAP_INV,
    MATTERPORT_ALL_CLASSES_TO_21,
    SCANNET200_LEARNING_MAP_INV,
)
from univlg.modeling.backproject.backproject import (
    interpolate_feats_3d,
    multiscsale_voxelize,
    voxel_map_to_source,
    voxelization,
    backprojector,
)
from univlg.utils import vis_utils
from univlg.utils.misc import is_dist_avail_and_initialized
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from transformers import AutoTokenizer, RobertaTokenizerFast

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom
from typing import TYPE_CHECKING, Any
from univlg.model_data import expand_tensor, load_scannet_data, prepare_targets, slice_tensor
from scannet_tools.scannet_dataset import ScanNetDataset
from scannet_tools.single_turn_exploration_scannet import MultiTurnExplorationFramework
from scannet_tools.agent_qwen import AgentQwen2_5
from scannet_tools.agent_gemini import GeminiAgent

from feature_map_tools.qwen_multicam import Qwen2_5_Projected3D
from transformers import AutoProcessor
from feature_map_tools.explore_qwen_vl import run_qwen_for_boundingbox, run_qwen

from torch_scatter import scatter_mean
from torch.nn import functional as F

if TYPE_CHECKING:
    from univlg.modeling.transformer_decoder.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder

logger = logging.getLogger(__name__)


st = ipdb.set_trace


@META_ARCH_REGISTRY.register()
class UniVLG(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        visual_backbone: nn.Module,
        mask_decoder: nn.Module,
        criterion: VideoSetCriterion,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        decoder_3d,
        supervise_sparse,
        eval_sparse,
        cfg,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        
        self.DEBUG = False
        self.USE_ZOOM_OUT = True
        self.BIRD_VIEW_CAM = False
        self.VOXELIZE = False
        self.EXPLORE = True
        self.USE_BOUNDING_BOX = True
        self.ThreeD_BBOX = False
        self.GEMINI = False
        self.ZOOM_RATIO = 0.6
        self.USE_OCCLUSION = False
        self.USE_FIRST_CAMERA = True
        # Generate timestamp only on the master process
        if not is_dist_avail_and_initialized() or comm.is_main_process():
            self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Experiment timestamp: {self.experiment_timestamp}")
        else:
            self.experiment_timestamp = None
            
        # Broadcast the timestamp from the main process to all processes
        if is_dist_avail_and_initialized():
            # Determine device for distributed communication
            current_device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
            
            if comm.is_main_process():
                timestamp_tensor = torch.tensor([ord(c) for c in self.experiment_timestamp], dtype=torch.long, device=current_device)
                timestamp_length = torch.tensor([len(self.experiment_timestamp)], dtype=torch.long, device=current_device)
            else:
                timestamp_length = torch.tensor([0], dtype=torch.long, device=current_device)
                
            # Broadcast the length first
            torch.distributed.broadcast(timestamp_length, 0)
            
            if not comm.is_main_process():
                # Create the tensor with the right size on other processes
                timestamp_tensor = torch.zeros(timestamp_length.item(), dtype=torch.long, device=current_device)
                
            # Broadcast the actual timestamp
            torch.distributed.broadcast(timestamp_tensor, 0)
            
            if not comm.is_main_process():
                # Convert back to string on other processes
                self.experiment_timestamp = ''.join([chr(i) for i in timestamp_tensor.cpu().tolist()])
        
        
        if cfg.USE_WANDB:
            print(f"Using wandb: {cfg.USE_WANDB}")
            if not is_dist_avail_and_initialized() or comm.is_main_process():
                name = (
                    cfg.OUTPUT_DIR.split("/")[-1]
                    if cfg.WANDB_NAME is None
                    else cfg.WANDB_NAME
                )

                kwargs = dict()
                kwargs['id'] = name

                wandb.init(
                    entity=cfg.WANDB_ENTITY,
                    project=cfg.WANDB_PROJECT,
                    sync_tensorboard=True,
                    name=name,
                    resume="allow",
                    config=cfg,
                    mode='online',
                    **kwargs,
                )

        # self.visual_backbone: UniVLGVisualBackbone = visual_backbone
        # self.mask_decoder: VideoMultiScaleMaskedTransformerDecoder = mask_decoder
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.visual_backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_feature_levels = 3

        self.decoder_3d = decoder_3d
        self.supervise_sparse = supervise_sparse
        self.eval_sparse = eval_sparse
        self.cfg = cfg
        
        TOP_N = 1
        temperature = 1.0 if TOP_N > 1 else 0.0
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("No GPU available, using CPU.")
        gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else None
        print(f"GPU ID: {gpu_id}")
        if self.EXPLORE:
            if not self.GEMINI:
                self.agent = AgentQwen2_5(temperature=temperature, gpu_id=gpu_id)
            else:
                self.agent = GeminiAgent(
                    api_key='AIzaSyAMcwWOQ17KKXvgnBPpXb92VwCZlkcphfk',
                    model_name="models/gemini-2.5-pro-preview-03-25",
                    temperature=temperature,
                )
        else:
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            self.model = Qwen2_5_Projected3D.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=f"cuda:{gpu_id}" if gpu_id is not None else "auto",
                attn_implementation="flash_attention_2",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)

        # self.categories = {k: v for k, v in enumerate(self.metadata.thing_classes)}

        if self.cfg.MODEL.OPEN_VOCAB:
            if self.cfg.TEXT_ENCODER_TYPE == "clip":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
            elif self.cfg.TEXT_ENCODER_TYPE == "jina":
                self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-clip-v1', trust_remote_code=True)
            else:
                self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.answer_tokenizer = AutoTokenizer.from_pretrained('t5-small')

        if self.cfg.LOG_GRADIENTS:
            assert self.cfg.USE_WANDB
            wandb.watch(self, log="all", log_freq=1)

    
    @classmethod
    def from_config(cls, cfg):
        visual_backbone = UniVLGVisualBackbone(
            cfg, freeze_backbone=cfg.MODEL.FREEZE_BACKBONE)
        
        mask_decoder = build_transformer_decoder(
            cfg,
            cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_classification=True,
        )

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        box_weight = cfg.MODEL.MASK_FORMER.BOX_WEIGHT
        giou_weight = cfg.MODEL.MASK_FORMER.GIOU_WEIGHT
        generation_weight = cfg.MODEL.MASK_FORMER.GENERATION_WEIGHT

        matching_class_weight = class_weight if cfg.MATCHING_CLASS_WEIGHT is None else cfg.MATCHING_CLASS_WEIGHT
        matching_dice_weight = dice_weight if cfg.MATCHING_DICE_WEIGHT is None else cfg.MATCHING_DICE_WEIGHT
        matching_mask_weight = mask_weight if cfg.MATCHING_MASK_WEIGHT is None else cfg.MATCHING_MASK_WEIGHT
        matching_box_weight = box_weight if cfg.MATCHING_BOX_WEIGHT is None else cfg.MATCHING_BOX_WEIGHT
        matching_giou_weight = giou_weight if cfg.MATCHING_GIOU_WEIGHT is None else cfg.MATCHING_GIOU_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=matching_class_weight,
            cost_mask=matching_mask_weight,
            cost_dice=matching_dice_weight,
            cost_bbox=matching_box_weight,
            cost_giou=matching_giou_weight,
            cost_mask_det=cfg.MODEL.MASK_FORMER.MASK_WEIGHT,
            cost_class_det=cfg.MODEL.MASK_FORMER.CLASS_WEIGHT,
            cost_dice_det=cfg.MODEL.MASK_FORMER.DICE_WEIGHT,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D,
            supervise_sparse=cfg.MODEL.SUPERVISE_SPARSE,
            cfg=cfg,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "loss_bbox": box_weight,
            "loss_giou": giou_weight,
            "loss_generation": generation_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        if cfg.USE_BOX_LOSS:
            losses.append("bboxes")

        if cfg.GENERATION:
            losses.append("generation")

        criterion_fn = VideoSetCriterion

        criterion = criterion_fn(
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_2D,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            supervise_sparse=cfg.MODEL.SUPERVISE_SPARSE,
            cfg=cfg,
        )

        return {
            "visual_backbone": visual_backbone,
            "mask_decoder": mask_decoder,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN) > 0 else MetadataCatalog.get(cfg.DATASETS.TRAIN_2D[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video/3d
            # "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "decoder_3d": cfg.MODEL.DECODER_3D,
            "supervise_sparse": cfg.MODEL.SUPERVISE_SPARSE,
            "eval_sparse": cfg.TEST.EVAL_SPARSE,
            "cfg": cfg,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def load_3d_data(self, batched_inputs, images_shape):
        valids = None
        multiview_data = None
        bs, v = images_shape[:2]
        h, w = batched_inputs[0]["images"][0].shape[-2:]
        
        multiview_data = {}
        multiview_data["multi_scale_xyz"] = [
            torch.stack(
                [batched_inputs[i]["multi_scale_xyz"][j] for i in range(bs)], dim=0
            ).to(self.device)
            for j in range(len(batched_inputs[0]["multi_scale_xyz"]))
        ]

        voxel_size = self.cfg.INPUT.VOXEL_SIZE[::-1]

        if self.cfg.INPUT.VOXELIZE:
            multiview_data["multi_scale_p2v"] = multiscsale_voxelize(
                multiview_data["multi_scale_xyz"], voxel_size
            )
        return valids, multiview_data
    
    def compute_aabb(self,point_cloud):
        """
        Compute the axis-aligned bounding box (AABB) for a given point cloud.

        Args:
            point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud,
                                        where N is the number of points.

        Returns:
            dict: A dictionary containing the min and max coordinates of the bounding box:
                {
                    "min_x": float,
                    "min_y": float,
                    "min_z": float,
                    "max_x": float,
                    "max_y": float,
                    "max_z": float
                }
        """
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("Point cloud must be a tensor of shape (N, 3).")

        # Compute the min and max coordinates along each axis
        min_coords = torch.min(point_cloud, dim=0).values
        max_coords = torch.max(point_cloud, dim=0).values

        # Return the bounding box as a dictionary
        return {
            "min_x": min_coords[0].item(),
            "min_y": min_coords[1].item(),
            "min_z": min_coords[2].item(),
            "max_x": max_coords[0].item(),
            "max_y": max_coords[1].item(),
            "max_z": max_coords[2].item(),
        }
        
    def compute_2d_iou(self, box1, box2):
        """
        Compute IoU between two 2D bounding boxes
        
        Args:
            box1, box2: Dictionaries with keys min_x, min_y, max_x, max_y
            
        Returns:
            float: IoU score
        """
        # Calculate intersection area
        inter_min_x = max(box1["min_x"], box2["min_x"])
        inter_min_y = max(box1["min_y"], box2["min_y"])
        inter_max_x = min(box1["max_x"], box2["max_x"])
        inter_max_y = min(box1["max_y"], box2["max_y"])
        
        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return 0.0  # No intersection
        
        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        
        # Calculate union area
        box1_area = (box1["max_x"] - box1["min_x"]) * (box1["max_y"] - box1["min_y"])
        box2_area = (box2["max_x"] - box2["min_x"]) * (box2["max_y"] - box2["min_y"])
        union_area = box1_area + box2_area - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou
        
    def get_point_cloud_in_2d_bounding_box(self, image_index, bounding_box, scannet_dataset):
        # Extract depth, intrinsics, and pose for the given image index
        depth = scannet_dataset.depth_images[image_index]
        intrinsic = scannet_dataset.intrinsic_depth
        pose = scannet_dataset.poses[image_index]
        pose = self._invert_pose(pose) # get camera to world pose

        # Convert bounding box to pixel coordinates
        min_x, min_y, max_x, max_y = (
            int(bounding_box["min_x"]),
            int(bounding_box["min_y"]),
            int(bounding_box["max_x"]),
            int(bounding_box["max_y"]),
        )
        min_x = max(min_x, 0)
        min_y = max(min_y, 0)
        max_x = min(max_x, depth.shape[1])
        max_y = min(max_y, depth.shape[0])

        # Crop the depth image to the bounding box
        cropped_depth = depth[min_y:max_y, min_x:max_x]

        # Generate pixel coordinates for the cropped region
        h, w = cropped_depth.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h), torch.arange(w), indexing="ij"
        )
        
        # Offset by the bounding box origin
        x_coords = x_coords.flatten() + min_x  
        y_coords = y_coords.flatten() + min_y
        z_coords = cropped_depth.flatten()

        # Filter out invalid depth values
        valid_mask = z_coords > 0
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        z_coords = z_coords[valid_mask]

        # # Check if we have any valid points - if not, return a default bounding box
        if len(x_coords) == 0 or len(y_coords) == 0 or len(z_coords) == 0:
            print(f"Warning: No valid depth values found in bounding box {bounding_box}")
            default_bbox = {
                "min_x": -0.1, "max_x": 0.1,
                "min_y": -0.1, "max_y": 0.1,
                "min_z": -0.1, "max_z": 0.1
            }
            return {
                "point_cloud": torch.zeros((0, 3)),  # Empty point cloud
                "refined_bounding_box": default_bbox
            }

        # Get intrinsic parameters (ensure we use the 3×3 intrinsic matrix)
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # Backproject to 3D camera coordinates directly using the camera equation
        x_cam = (x_coords - cx) * z_coords / fx
        y_cam = (y_coords - cy) * z_coords / fy
        camera_coords = torch.stack([x_cam, y_cam, torch.tensor(z_coords)], dim=1)

        # Transform to world coordinates using the pose matrix
        pose_matrix = torch.tensor(pose).float()
        
        # Add homogeneous coordinate for matrix multiplication
        ones = torch.ones_like(x_cam).unsqueeze(1)
        camera_coords_homogeneous = torch.cat([camera_coords, ones], dim=1)
        
        # Full transformation to world coordinates
        world_coords_homogeneous = torch.matmul(pose_matrix, camera_coords_homogeneous.T).T
        world_coords = world_coords_homogeneous[:, :3]

        # Compute the refined bounding box
        refined_bounding_box = self.compute_refined_bounding_box(world_coords)

        return {
            "point_cloud": world_coords,
            "refined_bounding_box": refined_bounding_box,
        }
    def _invert_pose(self, pose):
        """
        Invert a 4x4 pose matrix to convert between world-to-camera and camera-to-world
        """
        return pose
        # Extract rotation and translation
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Invert
        R_inv = R.T
        t_inv = -R_inv @ t
        
        # Create new pose matrix
        pose_inv = np.eye(4)
        pose_inv[:3, :3] = R_inv
        pose_inv[:3, 3] = t_inv
        
        return pose_inv

    def compute_refined_bounding_box(self, point_cloud):
        """
        Compute a refined bounding box containing the middle 70% of the points in the point cloud.

        Args:
            point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.

        Returns:
            dict: A dictionary containing the refined bounding box:
                {
                    "min_x": float,
                    "min_y": float,
                    "min_z": float,
                    "max_x": float,
                    "max_y": float,
                    "max_z": float
                }
        """
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("Point cloud must be a tensor of shape (N, 3).")

        # Sort points along each axis
        sorted_x = torch.sort(point_cloud[:, 0]).values
        sorted_y = torch.sort(point_cloud[:, 1]).values
        sorted_z = torch.sort(point_cloud[:, 2]).values

        # Compute 15th and 85th percentiles
        lower_idx = int(0.1 * len(sorted_x))
        upper_idx = int(0.9 * len(sorted_x))

        refined_bounding_box = {
            "min_x": sorted_x[lower_idx].item(),
            "max_x": sorted_x[upper_idx].item(),
            "min_y": sorted_y[lower_idx].item(),
            "max_y": sorted_y[upper_idx].item(),
            "min_z": sorted_z[lower_idx].item(),
            "max_z": sorted_z[upper_idx].item(),
        }

        return refined_bounding_box
    
    def compute_iou(self, box1, box2):
        """
        Compute the IoU (Intersection over Union) between two 3D bounding boxes.

        Args:
            box1 (dict): First bounding box with keys: min_x, min_y, min_z, max_x, max_y, max_z.
            box2 (dict): Second bounding box with keys: min_x, min_y, min_z, max_x, max_y, max_z.

        Returns:
            float: IoU value between the two bounding boxes.
        """
        # Compute the intersection box
        inter_min_x = max(box1["min_x"], box2["min_x"])
        inter_min_y = max(box1["min_y"], box2["min_y"])
        inter_min_z = max(box1["min_z"], box2["min_z"])
        inter_max_x = min(box1["max_x"], box2["max_x"])
        inter_max_y = min(box1["max_y"], box2["max_y"])
        inter_max_z = min(box1["max_z"], box2["max_z"])

        # Compute the intersection volume
        inter_width = max(0, inter_max_x - inter_min_x)
        inter_height = max(0, inter_max_y - inter_min_y)
        inter_depth = max(0, inter_max_z - inter_min_z)
        inter_volume = inter_width * inter_height * inter_depth

        # Compute the volume of each box
        box1_volume = (
            (box1["max_x"] - box1["min_x"])
            * (box1["max_y"] - box1["min_y"])
            * (box1["max_z"] - box1["min_z"])
        )
        box2_volume = (
            (box2["max_x"] - box2["min_x"])
            * (box2["max_y"] - box2["min_y"])
            * (box2["max_z"] - box2["min_z"])
        )

        # Compute the union volume
        union_volume = box1_volume + box2_volume - inter_volume

        # Compute IoU
        iou = inter_volume / union_volume if union_volume > 0 else 0.0
        return iou


    def find_highest_iou_label(self, pred_bounding_box, label_to_data, top_k=5):
        """
        Find the labels with the highest IoU between the predicted bounding box and all labels' bounding boxes.

        Args:
            pred_bounding_box (dict): Predicted bounding box with keys: min_x, min_y, min_z, max_x, max_y, max_z.
            label_to_data (dict): Dictionary containing labels and their bounding boxes.
            top_k (int): Number of top matches to return.

        Returns:
            dict: Dictionary with "ids" and "ious" lists containing the top-k matches.
        """
        # Store all IoUs and their corresponding labels
        all_ious = []

        for label, data in label_to_data.items():
            label_bounding_box = data["bounding_box"]
            iou = self.compute_iou(pred_bounding_box, label_bounding_box)
            all_ious.append({"id": label, "iou": iou})
        
        # Sort by IoU in descending order
        all_ious.sort(key=lambda x: x["iou"], reverse=True)
        
        # Get top-k results (or fewer if there aren't enough labels)
        top_results = all_ious[:min(top_k, len(all_ious))]
        
        # Print the top results
        for result in top_results:
            print(f"Label: {result['id']}, IoU: {result['iou']}")

        return {
            "ids": [result["id"] for result in top_results],
            "ious": [result["iou"] for result in top_results]
        }
    
    
    def save_point_cloud_to_ply(self, coords, colors, output_path):
        """
        Save point cloud coordinates and colors to a PLY file.
        
        Args:
            coords (torch.Tensor): Point cloud coordinates with shape (N, 3)
            colors (torch.Tensor): Point cloud colors with shape (N, 3)
            output_path (str): Path to save the PLY file
        """
        import os
        import numpy as np
        
        # Convert tensors to numpy if they're not already
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
            
        # Make sure colors are in the range 0-255
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write the PLY file
        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(coords)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Write vertex data
            for i in range(len(coords)):
                x, y, z = coords[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        
        print(f"Point cloud saved to {output_path}")
    
    def visualize_3d_bounding_boxes(self, gt_boxes_dict, pred_box, output_path=None, scannet_dataset=None, highlight_index=None, gt_id=None):
        """
        Visualize ground truth bounding boxes in blue, predicted bounding box in red, and camera poses in 3D space.
        The ground truth box that matches gt_id will be highlighted with a bright color.
        
        Args:
            gt_boxes_dict (dict): Dictionary of label -> bounding box data
            pred_box (dict): Predicted bounding box
            output_path (str, optional): Path to save the visualization. If None, the plot will be shown.
            scannet_dataset (ScanNetDataset, optional): Dataset containing camera pose information.
            highlight_index (int, optional): Index of the camera to highlight (typically the selected view).
            gt_id (int, optional): ID of the ground truth box to highlight.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Function to create vertices of a bounding box
        def get_box_vertices(box):
            x_min, y_min, z_min = box["min_x"], box["min_y"], box["min_z"]
            x_max, y_max, z_max = box["max_x"], box["max_y"], box["max_z"]
            
            vertices = [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max]
            ]
            return np.array(vertices)
        
        # Function to create faces of a bounding box from vertices
        def get_box_faces(vertices):
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom face
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top face
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front face
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back face
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right face
                [vertices[0], vertices[3], vertices[7], vertices[4]]   # left face
            ]
            return faces
        
        # Plot ground truth boxes in blue
        gt_box_plotted = False  # Flag to check if any regular GT box was plotted
        target_gt_plotted = False  # Flag to check if target GT box was plotted
        all_min_coords = []
        all_max_coords = []
        
        for label, data in gt_boxes_dict.items():
            box = data["bounding_box"]
            vertices = get_box_vertices(box)
            faces = get_box_faces(vertices)
            
            # Store min and max for setting axis limits
            all_min_coords.append([box["min_x"], box["min_y"], box["min_z"]])
            all_max_coords.append([box["max_x"], box["max_y"], box["max_z"]])
            
            if gt_id is not None and label == gt_id:
                # Highlight the target ground truth box with bright yellow and higher alpha
                ax.add_collection3d(Poly3DCollection(faces, alpha=0.9, facecolor='yellow', 
                                                    edgecolor='gold', linewidth=2, label='Target GT'))
                target_gt_plotted = True
            else:
                # Other ground truth boxes in blue with lower alpha
                if not gt_box_plotted:
                    # First box gets a label for the legend
                    ax.add_collection3d(Poly3DCollection(faces, alpha=0.05, facecolor='blue', 
                                                    edgecolor='blue', label='Other Ground Truth'))
                    gt_box_plotted = True
                else:
                    # Subsequent boxes don't need labels
                    ax.add_collection3d(Poly3DCollection(faces, alpha=0.05, facecolor='blue', edgecolor='blue'))
        
        # Plot predicted box in red
        vertices = get_box_vertices(pred_box)
        faces = get_box_faces(vertices)
        
        # Store min and max for setting axis limits
        all_min_coords.append([pred_box["min_x"], pred_box["min_y"], pred_box["min_z"]])
        all_max_coords.append([pred_box["max_x"], pred_box["max_y"], pred_box["max_z"]])
        
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.9, facecolor='red', edgecolor='red', label='Prediction'))
        
        # Plot camera poses if provided
        if scannet_dataset is not None and hasattr(scannet_dataset, 'poses'):
            # Function to create camera frustum
            def create_camera_frustum(pose, scale=0.001):
                # Camera position is the translation vector from the pose matrix
                
                position = pose[:3, 3]
                
                # Camera orientation vectors from pose matrix
                forward = -pose[:3, 2]  # Z-axis (negative because camera looks along negative Z)
                right = pose[:3, 0]     # X-axis
                up = pose[:3, 1]        # Y-axis
                
                # Scale the vectors to be visible
                forward = forward * scale
                right = right * scale * 0.6  # Slightly smaller for aesthetics
                up = up * scale * 0.6
                
                # Create frustum vertices
                vertices = [
                    position,  # Camera center
                    position + forward + right + up,    # Top-right corner
                    position + forward - right + up,    # Top-left corner
                    position + forward - right - up,    # Bottom-left corner
                    position + forward + right - up     # Bottom-right corner
                ]
                return np.array(vertices)
            
            # Draw camera frustums
            camera_shown = False
            highlight_shown = False
            
            for i, pose in enumerate(scannet_dataset.poses):
                pose = self._invert_pose(pose)  # Invert pose to get camera to world
                frustum = create_camera_frustum(pose)
                pos = pose[:3, 3]

                
                # Store camera position for axis limits
                all_min_coords.append(pos)
                all_max_coords.append(pos)
                
                if i == highlight_index:  # Highlight the selected camera
                    # Draw a special marker for the highlighted camera
                    ax.scatter(pos[0], pos[1], pos[2], color='lime', s=100, marker='*', label='Selected Camera' if not highlight_shown else None)
                    highlight_shown = True
                    
                    # Draw frustum lines for highlighted camera
                    ax.plot([pos[0], frustum[1][0]], [pos[1], frustum[1][1]], [pos[2], frustum[1][2]], 'g-')
                    ax.plot([pos[0], frustum[2][0]], [pos[1], frustum[2][1]], [pos[2], frustum[2][2]], 'g-')
                    ax.plot([pos[0], frustum[3][0]], [pos[1], frustum[3][1]], [pos[2], frustum[3][2]], 'g-')
                    ax.plot([pos[0], frustum[4][0]], [pos[1], frustum[4][1]], [pos[2], frustum[4][2]], 'g-')
                    
                    # Connect the frustum corners
                    ax.plot([frustum[1][0], frustum[2][0]], [frustum[1][1], frustum[2][1]], [frustum[1][2], frustum[2][2]], 'g-')
                    ax.plot([frustum[2][0], frustum[3][0]], [frustum[2][1], frustum[3][1]], [frustum[2][2], frustum[3][2]], 'g-')
                    ax.plot([frustum[3][0], frustum[4][0]], [frustum[3][1], frustum[4][1]], [frustum[3][2], frustum[4][2]], 'g-')
                    ax.plot([frustum[4][0], frustum[1][0]], [frustum[4][1], frustum[1][1]], [frustum[4][2], frustum[1][2]], 'g-')
                else:
                    # Regular cameras with gray frustums
                    color = 'gray'
                    ax.scatter(pos[0], pos[1], pos[2], color=color, s=20, 
                            label='Other Cameras' if not camera_shown else None)
                    camera_shown = True
                    
                    # Draw frustum lines for other cameras with gray color
                    ax.plot([pos[0], frustum[1][0]], [pos[1], frustum[1][1]], [pos[2], frustum[1][2]], color=color, alpha=0.5, linewidth=0.8)
                    ax.plot([pos[0], frustum[2][0]], [pos[1], frustum[2][1]], [pos[2], frustum[2][2]], color=color, alpha=0.5, linewidth=0.8)
                    ax.plot([pos[0], frustum[3][0]], [pos[1], frustum[3][1]], [pos[2], frustum[3][2]], color=color, alpha=0.5, linewidth=0.8)
                    ax.plot([pos[0], frustum[4][0]], [pos[1], frustum[4][1]], [pos[2], frustum[4][2]], color=color, alpha=0.5, linewidth=0.8)
                    
                    # Connect the frustum corners with thinner, semi-transparent lines
                    ax.plot([frustum[1][0], frustum[2][0]], [frustum[1][1], frustum[2][1]], [frustum[1][2], frustum[2][2]], color=color, alpha=0.5, linewidth=0.8)
                    ax.plot([frustum[2][0], frustum[3][0]], [frustum[2][1], frustum[3][1]], [frustum[2][2], frustum[3][2]], color=color, alpha=0.5, linewidth=0.8)
                    ax.plot([frustum[3][0], frustum[4][0]], [frustum[3][1], frustum[4][1]], [frustum[3][2], frustum[4][2]], color=color, alpha=0.5, linewidth=0.8)
                    ax.plot([frustum[4][0], frustum[1][0]], [frustum[4][1], frustum[1][1]], [frustum[4][2], frustum[1][2]], color=color, alpha=0.5, linewidth=0.8)
        
        # Set axis limits based on all boxes and cameras
        all_min_coords = np.min(np.array(all_min_coords), axis=0)
        all_max_coords = np.max(np.array(all_max_coords), axis=0)
        
        # Add some padding
        padding = max(
            (all_max_coords[0] - all_min_coords[0]),
            (all_max_coords[1] - all_min_coords[1]),
            (all_max_coords[2] - all_min_coords[2])
        ) * 0.1
        
        ax.set_xlim([all_min_coords[0] - padding, all_max_coords[0] + padding])
        ax.set_ylim([all_min_coords[1] - padding, all_max_coords[1] + padding])
        ax.set_zlim([all_min_coords[2] - padding, all_max_coords[2] + padding])
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Bounding Boxes and Camera Poses Visualization')
        ax.legend()
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Save or show
        if output_path is not None:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    
    def find_nearest_boxes_for_points(self, points, label_to_data):
        """
        For each point, find the 3 nearest bounding boxes and apply weighted voting.
        
        Args:
            points: List of 3D points (torch.Tensor)
            label_to_data: Dictionary mapping labels to data containing bounding boxes
        
        Returns:
            Dictionary with ranked labels and their scores
        """
        # Weights for the 3 nearest boxes
        weights = [1.0, 0.4, 0.2, 0.1, 0.05]
        weighted_votes = {}
        
        # Function to check if point is inside a bounding box
        def is_point_in_box(point, box):
            return (point[0] >= box["min_x"] and point[0] <= box["max_x"] and
                    point[1] >= box["min_y"] and point[1] <= box["max_y"] and
                    point[2] >= box["min_z"] and point[2] <= box["max_z"])
        
        # Function to calculate distance from point to box center
        def distance_to_box_center(point, box):
            center_x = (box["min_x"] + box["max_x"]) / 2
            center_y = (box["min_y"] + box["max_y"]) / 2
            center_z = (box["min_z"] + box["max_z"]) / 2
            
            return ((point[0] - center_x)**2 + 
                    (point[1] - center_y)**2 + (point[2] - center_z)**2)**0.5
        
        for i, point in enumerate(points):
            # Track boxes that contain this point
            containing_boxes = {}
            
            # Find all boxes that contain this point
            for label, data in label_to_data.items():
                if is_point_in_box(point, data["bounding_box"]):
                    containing_boxes[label] = distance_to_box_center(point, data["bounding_box"])
            
            # Calculate distances to all box centers for this point
            distances = {}
            for label, data in label_to_data.items():
                distances[label] = distance_to_box_center(point, data["bounding_box"])
            
            # Sort boxes by distance (prioritizing containing boxes)
            if containing_boxes:
                # Sort containing boxes by distance
                sorted_labels = sorted(containing_boxes.items(), key=lambda x: x[1])
                
                # Add non-containing boxes after containing boxes
                non_containing = {k: v for k, v in distances.items() if k not in containing_boxes}
                sorted_labels.extend(sorted(non_containing.items(), key=lambda x: x[1]))
            else:
                # If no box contains the point, sort all by distance
                sorted_labels = sorted(distances.items(), key=lambda x: x[1])
            
            # Apply weights to the top 3 boxes
            for rank, (label, _) in enumerate(sorted_labels[:5]):
                if label not in weighted_votes:
                    weighted_votes[label] = 0.0
                weighted_votes[label] += weights[rank]
        
        # Sort labels by weighted votes in descending order
        ranked_labels = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Create results
        results = {
            "ranked_labels": [{"id": label, "score": score} for label, score in ranked_labels],
            "most_common_label": ranked_labels[0][0] if ranked_labels else None,
            "all_scores": weighted_votes
        }
        
        return results
    
    
    def visualize_nearby_points_in_pointcloud(self, scannet_coords, scannet_colors, nearby_points, output_dir, target_name, camera_position=None):
        """
        Create a visualization of the pointcloud with nearby points highlighted and camera position marked.
        
        Args:
            scannet_coords: Original ScanNet point cloud coordinates
            scannet_colors: Original ScanNet point cloud colors
            nearby_points: Points identified near the selected pixel
            output_dir: Directory to save visualization
            target_name: Name of the target object for filename
            camera_position: Position of the camera to visualize (optional)
        """
        import os
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
        
        # Convert tensors to numpy arrays
        if isinstance(scannet_coords, torch.Tensor):
            scannet_coords = scannet_coords.cpu().numpy()
        if isinstance(scannet_colors, torch.Tensor):
            scannet_colors = scannet_colors.cpu().numpy()
        if isinstance(nearby_points, torch.Tensor):
            nearby_points = nearby_points.cpu().numpy()
        if isinstance(camera_position, torch.Tensor):
            camera_position = camera_position.cpu().numpy()
        
        # Normalize colors if needed
        if scannet_colors.max() <= 1.0:
            scannet_colors = scannet_colors * 255
        
        # Create a copy of the original colors
        modified_colors = scannet_colors.copy()
        
        # Find the nearest points in the full pointcloud to our nearby_points
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(scannet_coords)
        distances, indices = nbrs.kneighbors(nearby_points)
        
        # Make highlighted points bright gold
        highlight_color = np.array([255, 215, 0])  # Bright gold
        
        # Add a larger sphere for each highlighted point
        sphere_radius = 0.1  # Size of highlight sphere
        num_sphere_points = 200  # Points per sphere for visualization
        sphere_points = []
        sphere_colors = []
        
        for idx in indices.flatten():
            # Highlight the original point
            modified_colors[idx] = highlight_color
            
            # Create a small sphere around each point
            center = scannet_coords[idx]
            for i in range(num_sphere_points):
                # Simple algorithm to distribute points in sphere-like shape
                theta = np.pi * np.random.random()
                phi = 2 * np.pi * np.random.random()
                r = sphere_radius * np.random.random()
                
                x = center[0] + r * np.sin(theta) * np.cos(phi)
                y = center[1] + r * np.sin(theta) * np.sin(phi)
                z = center[2] + r * np.cos(theta)
                
                sphere_points.append([x, y, z])
                sphere_colors.append(highlight_color)
        
        # Add camera position markers if provided
        camera_sphere_points = []
        camera_sphere_colors = []
        
        if camera_position is not None:
            # Make camera points blue
            camera_color = np.array([30, 144, 255])  # Dodger Blue
            camera_radius = 0.1  # Smaller radius for camera points
            
            # Add 30 dense points around the camera position
            for i in range(200):
                # Distribute points in a small sphere around camera position
                theta = np.pi * np.random.random()
                phi = 2 * np.pi * np.random.random()
                r = camera_radius * np.random.random()
                
                x = camera_position[0] + r * np.sin(theta) * np.cos(phi)
                y = camera_position[1] + r * np.sin(theta) * np.sin(phi)
                z = camera_position[2] + r * np.cos(theta)
                
                camera_sphere_points.append([x, y, z])
                camera_sphere_colors.append(camera_color)
        
        # Combine all points and colors
        combined_points = []
        combined_colors = []
        
        # Start with original point cloud
        combined_points.append(scannet_coords)
        combined_colors.append(modified_colors)
        
        # Add highlight spheres if any
        if len(sphere_points) > 0:
            combined_points.append(np.array(sphere_points))
            combined_colors.append(np.array(sphere_colors))
        
        # Add camera position markers if any
        if len(camera_sphere_points) > 0:
            combined_points.append(np.array(camera_sphere_points))
            combined_colors.append(np.array(camera_sphere_colors))
        
        # Stack all points and colors
        all_coords = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)
        
        # Save the visualization
        output_path = os.path.join(output_dir, f"{target_name.replace(' ', '_')}_nearby_points.ply")
        self.save_point_cloud_to_ply(all_coords, all_colors, output_path)
        print(f"Visualization with highlighted points and camera position saved to {output_path}")
        
        return output_path
    
    def get_3d_points_near_pixel(self, pixel_coords, points_cam, valid_mask, depth_values, voxelized_pc, p2v, radius=0.02, c2w=None, intrinsics=None, H=480, W=640):
        """
        Find 3D points in the neighborhood of a given pixel coordinate.
        
        Args:
            pixel_coords: [x, y] pixel coordinates
            points_cam: Camera-space points from zoomed_points_cam
            valid_mask: Visibility mask from zoomed_valid_mask
            depth_values: Depth values from zoomed_depth_values
            voxelized_pc: Voxelized point cloud
            p2v: Point to voxel mapping
            radius: Search radius in normalized coordinates
            H, W: Image dimensions
        
        Returns:
            List of 3D voxel coordinates near the pixel
        """
        # Convert pixel coordinates to normalized [0,1] coordinates
        x_norm = pixel_coords[0] / W
        y_norm = pixel_coords[1] / H
        
        # Get valid points
        valid_points = valid_mask.squeeze(0).squeeze(0)  # [num_points]
        
        # Get normalized camera coordinates for valid points
        points = points_cam[0, 0][valid_points]  # [num_valid, 2]
        
        # Calculate distance from the normalized pixel to each point
        distances = torch.sqrt((points[:, 0] - x_norm)**2 + (points[:, 1] - y_norm)**2)
        
        # Find points within the radius
        nearby_indices = torch.where(distances < radius)[0]
        print(f"Found {len(nearby_indices)} points within radius {radius}")
        
        # Get original indices in the voxelized point cloud
        original_indices = torch.where(valid_points)[0][nearby_indices]
        
        ###### try unprojecting the points to 3D space directly#######
        # Get the depth values for these points
        depth_values = depth_values[0,0][valid_points] # all valid depth values
        
        assert torch.all(depth_values > 0), "All depth values must be greater than 0"
        
        nearby_depths = depth_values[nearby_indices]
        
        # Unproject the points to 3D space
        
        nearby_2d_points = points[nearby_indices]
        
        def unproject_to_3d(normalized_coords, depths, intrinsics, c2w, H=480, W=640):
            """
            Unproject 2D points to 3D world coordinates
            
            Args:
                normalized_coords: Normalized coordinates in range [0,1] (N, 2)
                depths: Depth values (N)
                intrinsics: Camera intrinsic matrix (4x4 or 3x3)
                c2w: Camera-to-world pose matrix (4x4)
                H, W: Image dimensions
            
            Returns:
                World coordinates (N, 3)
            """
            # Convert normalized coordinates to pixel coordinates
            pixel_x = normalized_coords[:, 0] * W
            pixel_y = normalized_coords[:, 1] * H
            
            # Extract intrinsic parameters
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            
            # Unproject to camera space
            x_cam = (pixel_x - cx) * depths / fx
            y_cam = (pixel_y - cy) * depths / fy
            z_cam = depths
            
            # Create camera space points
            camera_points = torch.stack([x_cam, y_cam, z_cam, torch.ones_like(x_cam)], dim=1)
            
            # Transform to world coordinates
            world_points_h = torch.matmul(c2w, camera_points.transpose(0, 1)).transpose(0, 1)
            
            # Return 3D world coordinates
            return world_points_h[:, :3]
        
        unprojected_points = unproject_to_3d(
            nearby_2d_points,
            nearby_depths.squeeze(-1),
            intrinsics,
            c2w,
            H=H,
            W=W
        )

        # Get the corresponding voxel indices
        voxel_indices = p2v[0][original_indices]
        
        # Get unique voxel indices (multiple points may map to the same voxel)
        unique_voxel_indices = torch.unique(voxel_indices)
        
        # Get the 3D coordinates of these voxels
        nearby_points = voxelized_pc[0, unique_voxel_indices]
        
        print(f"Original nearby points shape: {nearby_points.shape}")
        print(f"Unprojected points shape: {unprojected_points.shape}")

        # Verify if they are close
        if len(nearby_points) == len(unprojected_points):
            distances = torch.norm(nearby_points - unprojected_points, dim=1)
            mean_distance = distances.mean().item()
            print(f"Mean distance between voxelized and unprojected points: {mean_distance:.4f}")

        # Use unprojected points instead of voxelized points for greater precision
        precise_3d_points = unprojected_points
        
        # import pdb;pdb.set_trace()
        
        
        return precise_3d_points, unique_voxel_indices
    
    # def get_3d_points_near_pixel(self, pixel_coords, points_cam, valid_mask, depth_values, voxelized_pc, p2v, radius=0.02, H=480, W=640):
        
    #     # pixel_coords: [312, 125], agent's output
    #     x_norm = pixel_coords[0] / W
    #     y_norm = pixel_coords[1] / H
        
    #     # Get valid points
    #     valid_points = valid_mask.squeeze(0).squeeze(0)  # [num_points]
    #     points = points_cam[0, 0][valid_points]  # [num_valid, 2]
    #     distances = torch.sqrt((points[:, 0] - x_norm)**2 + (points[:, 1] - y_norm)**2)
    #     # radius: 0.02
    #     nearby_indices = torch.where(distances < radius)[0]
    #     # import pdb; pdb.set_trace()
    #     original_indices = torch.where(valid_points)[0][nearby_indices]
        
    #     # Get the corresponding voxel indices
    #     voxel_indices = p2v[0][original_indices]
    #     unique_voxel_indices = torch.unique(voxel_indices)
        
    #     # Get the 3D coordinates of these voxels
    #     nearby_points = voxelized_pc[0, unique_voxel_indices]
        
    #     return nearby_points, unique_voxel_indices
    
    
    
    
    
    def forward(self, batched_inputs):
        """
            Args:
                batched_inputs (list[dict]): Batched inputs to the model. Each item in the list is a dict and corresponds to input for one batch element.
                    Since batched_inputs[1] has the same structure as batched_inputs[0], we describe batched_inputs[0] below.

                    Each dict contains the following keys:
                        height (int): Height of the image, always 448.
                        width (int): Width of the image, always 448.
                        segment_file_names (list[str]): List of segment file names, length = frames.
                        valid_file_names (list[str]): List of valid file names, length = frames.
                        length (int): Length of the video (number of frames), always 15.
                        image_id (str): Image ID.
                        decoder_3d (bool): Flag indicating if 3D decoder is used.
                        actual_decoder_3d (bool): Flag indicating if actual 3D decoder is used.
                        images (list[torch.Tensor]): List of images tensors, length = frames.
                            Each tensor has shape (3, H, W).
                        padding_masks (torch.Tensor): Padding mask for the image, shape (H, W).
                        image (torch.Tensor): Single image tensor, shape (3, H, W).
                        instances (NoneType): Always None.
                        instances_all (list[empty]): Always an empty list.
                        file_names (list[str]): List of file names, length = frames.
                        image_ids (list[int]): List of image IDs, length = frames.
                        file_name (str): File name.
                        valid_class_ids (np.ndarray): Numpy array of valid class IDs, shape (num_valid_classes,), where num_valid_classes is e.g., 20.
                        multiplier (int): Multiplier value.
                        do_camera_drop (bool): Flag for camera drop augmentation.
                        camera_drop_prob (float): Probability of camera drop.
                        camera_drop_min_frames_keep (int): Minimum frames to keep after camera drop.
                        always_keep_first_frame (bool): Flag to always keep the first frame.
                        max_frames (int): Maximum number of frames, always 15.
                        use_ghost (bool): Flag for using ghost augmentation.
                        pseudo_2d_aug (bool): Flag for pseudo 2D augmentation.
                        instances_all_full (list[empty]): Always an empty list.
                        depths (list[torch.Tensor]): List of depth tensors, length = frames.
                            Each tensor has shape (H, W).
                        poses (list[torch.Tensor]): List of pose tensors, length = frames.
                            Each tensor has shape (4, 4). Camera to world pose.
                        intrinsics (list[torch.Tensor]): List of intrinsics tensors, length = frames.
                            Each tensor has shape (4, 4).
                        depth_file_names (list[str]): List of depth file names, length = frames.
                        pose_file_names (list[str]): List of pose file names, length = frames.
                        new_image_shape (tuple[int, int]): Tuple representing the new image shape, e.g., (H, W).
                        original_all_classes (dict[int, str]): Dictionary mapping class IDs to class names for original classes.
                        all_classes (dict[int, str]): Dictionary mapping class IDs to class names for all classes.
                        scannet_coords (torch.Tensor): ScanNet point cloud coordinates, shape (num_points, 3).
                        scannet_color (torch.Tensor): ScanNet point cloud colors, shape (num_points, 3).
                        scannet_labels (torch.Tensor): ScanNet point cloud labels, shape (num_points, 2).
                        scannet_segments (torch.Tensor): ScanNet point cloud segments, shape (num_points,).
                        dataset_name (str): Name of the dataset.
                        num_classes (int): Number of classes.
                        full_scene_dataset (bool): Flag indicating if it's a full scene dataset.
                        multi_scale_xyz (list[torch.Tensor]): List of multi-scale point clouds. Length is num_scales (e.g., 4).
                            Each tensor has shape (frames, small_H, small_W, 3), where small_H and small_W vary across scales.
                            For the first scale, e.g., we might have small_H=32, small_W=32.
                        original_xyz (torch.Tensor): Original point cloud in image space, shape (frames, H, W, 3).
                        relevant_ids (list[int]): List of relevant IDs. # perhaps they are OBJECT ids, not frame ids.
                        sr3d_data (list[dict]): List of SR3D data dictionaries, length = 1.
                            Each inner dict contains:
                                text_caption (str): Text caption.
                                target_id (int): Target object ID.
                                anchor_ids (list[int]): List of anchor object IDs.
                                target_name (str): Target object name.
                                anchors_names (list[str]): List of anchor object names.
                                tokens_positive (list[list[list[int]]]): Positive tokens.
                                tokenized (transformers.tokenization_utils_base.BatchEncoding): Tokenized text.
                                positive_map (torch.Tensor): Positive map, shape (num_queries, sequence_length), e.g., (2, 520).
                                positive_map_od (NoneType): Always None.
                                annotation_id (NoneType): Always None.

        Returns:
            output
                
        """
        # self.save_point_cloud_to_ply(batched_inputs[0]["scannet_coords"], batched_inputs[0]["scannet_color"], "viz/scannet_191.ply")
        
        if 1:
            images = []
            for video in batched_inputs:
                for image in video["images"]:
                    images.append(image.to(self.device))
            bs = len(batched_inputs)
            v = len(batched_inputs[0]["images"])
            
            # if not self.training:
            print(f"Batch size: {bs}, Number of frames per video: {v}")
                # print(batched_inputs[0]["sr3d_data"][0]["target_id"])
                # print(batched_inputs[0]["sr3d_data"][0]["anchor_ids"])
                # import pdb;pdb.set_trace()
            
            
            images = batched_inputs[0]["images"]
            depths = batched_inputs[0]["depths"]
            poses = batched_inputs[0]["poses"]
            intrinsics = batched_inputs[0]["intrinsics"]
            
            # # # save some images to look at
            # img_temp_dir = "./img_temp"
            # os.makedirs(img_temp_dir, exist_ok=True)

            # # Save the image with the scene name prefix
            # scene = batched_inputs[0]["pose_file_names"][0].split("/")[-3]

            # for i, image in enumerate(images):
            #     # Convert tensor to numpy array if needed
            #     if isinstance(image, torch.Tensor):
            #         image = image.permute(1, 2, 0).cpu().numpy()
                
            #     # Save the image
            #     temp_image_path = os.path.join(img_temp_dir, f"{scene}_{i}.png")
            #     Image.fromarray(image).save(temp_image_path)
            
            
            
            
            preload_data_for_scannet = {
                'color_images': [],
                'depth_images': [],
                'poses': [],
                'frame_indices': [],
                'intrinsic_depth': None
            }

            # Process each frame's data
            for i in range(len(images)):
                # Convert tensors to numpy arrays if needed
                color_image = images[i].permute(1,2,0).cpu().numpy() if isinstance(images[i], torch.Tensor) else images[i]
                depth_image = depths[i].cpu().numpy() if isinstance(depths[i], torch.Tensor) else depths[i]
                pose = poses[i].cpu().numpy() if isinstance(poses[i], torch.Tensor) else poses[i]
                
                # Add to lists
                preload_data_for_scannet['color_images'].append(color_image)
                preload_data_for_scannet['depth_images'].append(depth_image)
                preload_data_for_scannet['poses'].append(pose)
                preload_data_for_scannet['frame_indices'].append(i)  # Using index as frame_idx

            # Set intrinsic matrix (assuming it's the same for all frames)
            intrinsic = intrinsics[0].cpu().numpy() if isinstance(intrinsics[0], torch.Tensor) else intrinsics[0]
            preload_data_for_scannet['intrinsic_depth'] = intrinsic

            # Optionally add color_files if they exist
            if "file_names" in batched_inputs[0]:
                preload_data_for_scannet['color_files'] = batched_inputs[0]["file_names"]

            # Now create the ScanNetDataset instance with the preloaded data
            
            scannet_dataset = ScanNetDataset(preloaded_data=preload_data_for_scannet)
            """
                Now we have the scannet dataset, we can do the multi-turn exploration
            """
            steps =8
            RESIZE_LENGTH=640
            scene = batched_inputs[0]["pose_file_names"][0].split("/")[-3]
            output_dir = os.path.join("output", f"scannet_exp_{self.experiment_timestamp}", f"scene_{scene}")
            base_path = "./scannet_data_golden"
            os.makedirs(base_path, exist_ok=True)
            
            
            # original_target
            if not self.DEBUG:
                target = batched_inputs[0]["sr3d_data"][0]["text_caption"]
            else:
                target = batched_inputs[0]["sr3d_data"][0]["target_name"]
            # gt_id = batched_inputs[0]["sr3d_data"][0]["target_id"]
            # anchor_ids = batched_inputs[0]["sr3d_data"][0]["anchor_ids"]
            # print (f"Target: {target}")
            # print (f"GT ID: {gt_id}")
            # print (f"Anchor IDs: {anchor_ids}")
            
            # print("All classes in the dataset:")
            # for class_id, class_name in batched_inputs[0]["all_classes"].items():
            #     print(f"classes {class_id}: {class_name}")
                
            # for class_id, class_name in batched_inputs[0]["original_all_classes"].items():
            #     print(f"original classes {class_id}: {class_name}")
            # import pdb;pdb.set_trace()
            
            
            
            
            description_path = os.path.join(base_path, scene, f"{target.replace(' ', '_')}")
            
            
            EXPLORE = self.EXPLORE
            if EXPLORE:
                multi_turn_explore = MultiTurnExplorationFramework(
                    scannet_dataset, 
                    self.agent, 
                    target, 
                    max_turns=steps, 
                    resize_length=RESIZE_LENGTH, 
                    cur_path=output_dir, 
                    need_caption=False, 
                    description_path=description_path, 
                    use_bounding_box=self.USE_BOUNDING_BOX, 
                    use_gemini=self.GEMINI,
                    scannet_coords=batched_inputs[0]["scannet_coords"],
                    scannet_color=batched_inputs[0]["scannet_color"],
                    use_zoom_out=self.USE_ZOOM_OUT,  # Add this parameter
                    zoom_distance=1.25,  # Optional: can be configurable
                    zoom_factor=0.5     # Optional: can be configurable
                )
                
                agent_answer, selected_index, zoomed_view_data = multi_turn_explore.explore()
            else:
                # use the feature field solution
                
                # Step 1: Convert images, depths, and poses to tensors if they're not already
                rgb_images = batched_inputs[0]["images"]
                rgb_images_torch = torch.stack(rgb_images).float().to(self.device)
                depth_images = batched_inputs[0]["depths"]
                depth_images_torch = torch.stack(depth_images).float().to(self.device) # already in meters
                
                poses_torch = torch.stack([pose.to(self.device) for pose in batched_inputs[0]["poses"]])
                intrinsics_torch = torch.stack([intr.to(self.device) for intr in batched_inputs[0]["intrinsics"]])
                
                # Step 2: Back-project to 3D
                B, H, W = depth_images_torch.shape
                
                # depth_images_torch = torch.where(depth_images_torch < 0.2, torch.tensor(1.0, device=depth_images_torch.device), depth_images_torch)
                
                world_coords = backprojector(
                    [[B, 3, H, W]], depth_images_torch[:, None], poses_torch[:, None], intrinsics_torch[:, None])[0]
                # [[1,1,480,640,3]]
                
                # Step 3: Downsample point cloud to reduce computation
                downsample_factor = 28
                ds_h, ds_w = 17, 23
                pointmap = world_coords[0]
                pointmap = F.interpolate(
                    pointmap.flatten(0, 1).permute(0, 3, 1, 2),
                    size=(ds_h, ds_w), 
                    mode='nearest'
                ).permute(0, 2, 3, 1).reshape(B, -1, ds_h, ds_w, 3)
                
                if self.VOXELIZE:
                    # Step 4: Voxelize the point cloud
                    voxel_size = 0.02
                    flat_pointmap = pointmap.reshape(-1, 3)[None]
                    p2v = voxelization(flat_pointmap, voxel_size) # [1, 391]
                    voxelized_pc = scatter_mean(flat_pointmap, p2v, 1) # [1, 391, 3]
                else:
                    voxelized_pc = pointmap.reshape(-1, 3)[None]
                    # Create indices tensor with the right shape for expansion
                    p2v = torch.arange(voxelized_pc.shape[1], device=voxelized_pc.device).unsqueeze(0)

                
                # Also voxelize RGB values
                # rgb_voxelized_pc = scatter_mean(rgb_images_torch_resized.reshape(-1, 3)[None], p2v, 1)
                
                # Step 5: Define point sampling function for camera projection
                def point_sampling(points, poses, intrinsics, height, width, use_occlusion=False):
                    """Project 3D points to camera views and check visibility"""
                    assert points.dtype == torch.float32
                    assert poses.dtype == torch.float32
                    
                    # Get camera-to-world projection matrix
                    depth2img = intrinsics @ poses.inverse()
                    
                    
                    # Add homogeneous coordinate
                    points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
                    
                    
                    B, num_query = points.shape[:2]
                    num_cam = depth2img.shape[1]
                    
                    points = points[:, None].repeat(1, num_cam, 1, 1)
                    
                    # Transform points to camera space
                    points_cam = torch.einsum('bnij,bnmj->bnmi', depth2img, points)
                    
                    eps = 1e-5
                    
                    # Check which points are in front of camera
                    valid_mask = (points_cam[..., 2:3] > eps)
                    
                    # Store the depth values before normalization
                    depth_values = points_cam[..., 2:3].clone()
                    
                    # Normalize by depth
                    points_cam = points_cam[..., 0:2] / torch.maximum(
                        points_cam[..., 2:3], torch.ones_like(points_cam[..., 2:3]) * eps)
                    
                    # Normalize to [0, 1] range
                    points_cam[..., 0] /= width
                    points_cam[..., 1] /= height
                    
                    # Check which points are within image bounds
                    valid_mask = (valid_mask & 
                                (points_cam[..., 1:2] > -0.5) &
                                (points_cam[..., 1:2] < 1.05) &
                                (points_cam[..., 0:1] < 1.05) &
                                (points_cam[..., 0:1] > -0.5))
                    
                    valid_mask = torch.nan_to_num(valid_mask).squeeze(-1)
                    
                    # we should do a check on the occlusion in the pixels: our feature map should be 17 in height and 23 in width. Project the feature cloud to the image and get each pixel with the smallest depth to be the valid one
                    if use_occlusion:
                        # Create a depth buffer initialized with infinity for the feature map
                        B, num_cam, num_points = valid_mask.shape
                        depth_buffer = torch.full((B, num_cam, ds_h, ds_w), float('inf'), device=points_cam.device)
                        
                        # FIRST PASS: Fill the depth buffer with minimum depths
                        for b in range(B):
                            for c in range(num_cam):
                                for p in range(num_points):
                                    if valid_mask[b, c, p]:
                                        # Convert normalized coordinates to pixel coordinates in feature map
                                        px = int((points_cam[b, c, p, 0] * ds_w).round())
                                        py = int((points_cam[b, c, p, 1] * ds_h).round())
                                        
                                        # Ensure we're within bounds
                                        if 0 <= px < ds_w and 0 <= py < ds_h:
                                            # Get current depth of this point
                                            current_depth = depth_values[b, c, p, 0]
                                            
                                            # Update depth buffer with minimum depth
                                            depth_buffer[b, c, py, px] = min(depth_buffer[b, c, py, px], current_depth)
                        
                        # SECOND PASS: Check each point against the finalized depth buffer
                        occlusion_mask = torch.zeros_like(valid_mask)
                        for b in range(B):
                            for c in range(num_cam):
                                for p in range(num_points):
                                    if valid_mask[b, c, p]:
                                        px = int((points_cam[b, c, p, 0] * ds_w).round())
                                        py = int((points_cam[b, c, p, 1] * ds_h).round())
                                        
                                        if 0 <= px < ds_w and 0 <= py < ds_h:
                                            current_depth = depth_values[b, c, p, 0]
                                            
                                            # If this point's depth matches the minimum depth at this pixel,
                                            # mark it as visible (not occluded)
                                            if abs(current_depth - depth_buffer[b, c, py, px]) < 1e-5:
                                                occlusion_mask[b, c, p] = 1
                        
                        # Update the valid mask to consider occlusion
                        valid_mask = valid_mask & (occlusion_mask == 1)
                        
                        
                    
                    return points_cam, valid_mask, depth_values
                
                # Step 6: Select the best camera based on visibility
                def select_best_camera(voxelized_pc, poses, intrinsics, height, width):
                    """Select the camera with most visible points"""
                    num_cameras = poses.shape[0]
                    best_visible_count = -1
                    best_camera_idx = 0
                    best_points_cam = None
                    best_valid_mask = None
                    
                    range_ids=1 if self.USE_FIRST_CAMERA else num_cameras
                    # Check each camera
                    for camera_idx in range(range_ids):
                        points_cam_i, valid_mask_i, _ = point_sampling(
                            voxelized_pc, 
                            poses[camera_idx][None, None],
                            intrinsics[camera_idx][None, None],
                            height, width
                        )
                        
                        visible_count = valid_mask_i.sum().item()
                        
                        if visible_count > best_visible_count:
                            best_visible_count = visible_count
                            best_camera_idx = camera_idx
                            best_points_cam = points_cam_i
                            best_valid_mask = valid_mask_i
                    
                    print(f"Selected camera {best_camera_idx} with {best_visible_count} visible points")
                    return best_camera_idx, best_points_cam, best_valid_mask
                
                # Step 7: Find the best camera
                best_camera_idx, points_cam, valid_mask = select_best_camera(
                    voxelized_pc, poses_torch, intrinsics_torch, H, W
                )
                
                zoomed_intrinsics = intrinsics_torch[best_camera_idx].clone().to(voxelized_pc.device)
                zoomed_intrinsics[0, 0] *= self.ZOOM_RATIO  # Reduce focal length (zoom out) to see more
                zoomed_intrinsics[1, 1] *= self.ZOOM_RATIO
                
                
                # Step 8: Apply zoom-in by modifying the intrinsics for better visibility
                if not self.BIRD_VIEW_CAM:
                    
                    
                    # Project using zoomed intrinsics
                    # import ipdb; ipdb.set_trace()
                    zoomed_points_cam, zoomed_valid_mask, zoomed_depth_values = point_sampling(
                        voxelized_pc,
                        poses_torch[best_camera_idx][None, None],
                        zoomed_intrinsics[None, None],
                        H, W,
                        use_occlusion=self.USE_OCCLUSION
                    )
                    # import ipdb; ipdb.set_trace()
                    # If zoom improves visibility significantly, use it
                    zoomed_visible_count = zoomed_valid_mask.sum().item()
                    print (f"Zoomed camera {best_camera_idx} has {zoomed_visible_count} un-occluded visible points")
                    zoomed_extrinsics = poses_torch[best_camera_idx].clone()
                else: # USE BIRD VIEW CAMERA
                    # lets get the bounding box for the whole scene (voxelized_pc)
                    # import ipdb; ipdb.set_trace()
                    # get the min and max coordinates of the voxelized_pc
                    # import pdb; pdb.set_trace()
                    min_coords = torch.min(voxelized_pc, dim=1).values
                    min_coords = min_coords[0]
                    max_coords = torch.max(voxelized_pc, dim=1).values
                    max_coords = max_coords[0]
                    # get the center of the bounding box
                    center = (min_coords + max_coords) / 2
                    # select an appropriate z height out of the bounding box and look downwards
                    # create a new camera pose, looking downwards
                    def create_downward_camera_extrinsics(center, z_height):
                        """
                        Create a camera extrinsics matrix for a camera looking downwards.

                        Args:
                            center (torch.Tensor): The center of the scene (x, y, z) as a tensor of shape (3,).
                            z_height (float): The height of the camera above the center.

                        Returns:
                            torch.Tensor: A 4x4 extrinsics matrix for the downward-looking camera.
                        """
                        # Camera position (above the center)
                        camera_position = torch.tensor([center[0], center[1], center[2]+z_height])

                        # Camera orientation (looking downwards along -Z axis)
                        # Rotation matrix for looking downwards
                        rotation_matrix = torch.tensor([
                            [1,  0,  0],  # X-axis remains the same
                            [0, -1,  0],  # Y-axis is flipped
                            [0,  0, -1]   # Z-axis points downwards
                        ])

                        # Combine rotation and translation into a 4x4 extrinsics matrix
                        extrinsics = torch.eye(4)
                        extrinsics[:3, :3] = rotation_matrix
                        extrinsics[:3, 3] = camera_position

                        return extrinsics

                    # Example usage
                    z_height = 3.0 # Height above the center
                    downward_extrinsics = create_downward_camera_extrinsics(center, z_height)
                    print("Downward-looking camera extrinsics:\n", downward_extrinsics)
                    zoomed_extrinsics = downward_extrinsics.to(voxelized_pc.device)
                    
                    # use point sampling 
                    zoomed_points_cam, zoomed_valid_mask, zoomed_depth_values = point_sampling(
                        voxelized_pc,
                        zoomed_extrinsics[None, None],
                        zoomed_intrinsics[None, None],
                        H, W,
                        use_occlusion=self.USE_OCCLUSION
                    )
                    zoomed_visible_count = zoomed_valid_mask.sum().item()
                    print (f"Zoomed camera {best_camera_idx} has {zoomed_visible_count} un-occluded visible points. Overall points: {zoomed_valid_mask.shape}")
                
                
                points_cam = zoomed_points_cam
                valid_mask = zoomed_valid_mask
                depth_values = zoomed_depth_values
                
                depth_map = torch.zeros((H, W), device=self.device, dtype=depth_values.dtype)

                # Convert normalized coordinates to pixel coordinates
                pixel_x = (points_cam[0, 0, valid_mask.squeeze(0, 1), 0] * W).long()
                pixel_y = (points_cam[0, 0, valid_mask.squeeze(0, 1), 1] * H).long()

                # Ensure pixel coordinates are within bounds
                valid_pixels = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
                pixel_x = pixel_x[valid_pixels]
                pixel_y = pixel_y[valid_pixels]

                # Get corresponding depth values
                depths = depth_values[0, 0, valid_mask.squeeze(0, 1)][valid_pixels]

                # Fill in the depth map at valid pixel locations
                depth_map[pixel_y, pixel_x] = depths.squeeze(-1)

                # Optionally apply a filter to fill in holes
                # This is a simple approach - more advanced hole filling could be used
                # if 1:
                #     depth_map = cv2.medianBlur(depth_map.cpu().numpy(), 5)
                #     depth_map = torch.from_numpy(depth_map).to(self.device)
                    
                # do this here
                
                # output_caption = run_qwen(self.model, 
                #     self.processor,
                #     rgb_images_torch, 
                #     best_camera_idx=best_camera_idx,
                #     points_cam=points_cam,
                #     valid_mask=valid_mask,
                #     p2v=p2v
                # )[0]
                output_caption = "Skipped the caption generation step for now."
                # Save the caption to a local text file with the target name
                os.makedirs(output_dir, exist_ok=True)
                caption_file_path = os.path.join(output_dir, f"{target.replace(' ', '_')}.txt")
                with open(caption_file_path, "w") as caption_file:
                    caption_file.write(output_caption)       
                
                output_text = run_qwen_for_boundingbox(
                    self.model, 
                    self.processor,
                    rgb_images_torch, 
                    target,  # target description from input
                    best_camera_idx=best_camera_idx,
                    points_cam=points_cam,
                    valid_mask=valid_mask,
                    p2v=p2v
                )
                agent_answer = output_text[0] if output_text else ""
                print("Agent answer: ", agent_answer)
                
                # Append the agent's answer to the output text file
            
                with open(caption_file_path, "a") as output_txt_file:
                    output_txt_file.write(f"\n\nAgent Answer:\n{agent_answer}\n")
                print(f"Saved agent answer to: {caption_file_path}")
                
                answer_match = re.search(r'<answer>(.*)', agent_answer, re.DOTALL)
                if answer_match:
                    agent_answer = answer_match.group(1).strip()
                
                # image should be in size 640, 480

            if "Invalid" in agent_answer or "Error" in agent_answer:
                print("Error: Invalid view number detected in agent answer")
                return 1000, 0.0  # Error code 1000: Invalid view number
            elif "No reply" in agent_answer:
                print("Error: No reply within specified steps")
                return 2000, 0.0  # Error code 2000: Steps timeout
            #----------------------------- formatting and visualization ---------------------------------
            
            label_to_data = {}

            data = batched_inputs[0]
            # scannet_coords: (122497,3)
            # scannet_labels: (122497,2)
            
            scannet_labels = data["scannet_labels"][:,-1] # this is the instance label of each object
            scannet_coords = data["scannet_coords"]
            
            unique_labels = torch.unique(scannet_labels)
            
            
            for label in unique_labels:
                # Mask to filter points belonging to the current label
                mask = scannet_labels == label
                # Get the point cloud coordinates for the current label
                label_coords = scannet_coords[mask]
                # Compute the axis-aligned bounding box (AABB) for the current label
                
                min_coords = torch.min(label_coords, dim=0).values
                max_coords = torch.max(label_coords, dim=0).values
                bounding_box = {
                    "min_x": min_coords[0].item(),
                    "min_y": min_coords[1].item(),
                    "min_z": min_coords[2].item(),
                    "max_x": max_coords[0].item(),
                    "max_y": max_coords[1].item(),
                    "max_z": max_coords[2].item(),
                }
                # Store the point cloud coordinates and bounding box in the dictionary
                label_to_data[int(label.item())] = {
                    "bounding_box": bounding_box,
                }
            
            
            
            USE_BOUNDING_BOX = self.USE_BOUNDING_BOX
            if USE_BOUNDING_BOX:
                try:
                    # First try to extract JSON using the ```json format
                    json_str = re.search(r"```json\n([\s\S]*?)\n```", agent_answer)
                    if json_str:
                        json_str = json_str.group(1)
                        bbox_data = json.loads(json_str)
                        bounding_box = bbox_data[0]["bbox_2d"]  # Assuming the first entry is the correct one
                    else:
                                # Try to extract JSON without code blocks
                        json_match = re.search(r'({[\s\S]*?"bbox_2d"[\s\S]*?\})', agent_answer)
                        if json_match:
                            json_str = json_match.group(1)
                            bbox_data = json.loads(json_str)
                            bounding_box = bbox_data["bbox_2d"]
                        else:
                            # Try directly extracting the array
                            bbox_match = re.search(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', agent_answer)
                            if bbox_match:
                                bounding_box = [int(bbox_match.group(1)), int(bbox_match.group(2)), 
                                                int(bbox_match.group(3)), int(bbox_match.group(4))]
                            else:
                                raise Exception("Could not find JSON in the response")
                    
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    try:
                        # Fallback to the old format if JSON parsing fails
                        bounding_box = re.search(
                            r"Bounding Box:?[\n\s]*(?:\[|\(|{)?\s*(\d+)\s*[,;\s]\s*(\d+)\s*[,;\s]\s*(\d+)\s*[,;\s]\s*(\d+)\s*(?:\]|\)|})?", 
                            agent_answer, re.IGNORECASE
                        ).groups()
                        if self.GEMINI:
                            H, W = scannet_dataset.depth_images[selected_index].shape  # Get image dimensions
                            bounding_box = (
                                int(float(bounding_box[0])),  # xmin * width/1000
                                int(float(bounding_box[1])),  # ymin * height/1000
                                int(float(bounding_box[2])),  # xmax * width/1000
                                int(float(bounding_box[3]))   # ymax * height/1000
                            )
                            # bounding_box = (
                            #     int(float(bounding_box[0]) * W / 1000),  # xmin * width/1000
                            #     int(float(bounding_box[1]) * H / 1000),  # ymin * height/1000
                            #     int(float(bounding_box[2]) * W / 1000),  # xmax * width/1000
                            #     int(float(bounding_box[3]) * H / 1000)   # ymax * height/1000
                            # )
                        else:
                            bounding_box = tuple(int(coord) for coord in bounding_box)
                    except:
                        print("Error: Invalid bounding box format or agent refused to provide one")
                        bounding_box = [0, 0, RESIZE_LENGTH, RESIZE_LENGTH]
                        return 3000, 0.0

                print("Bounding box: ", bounding_box)
                # if output bounding box is not in the format [x1, y1, x2, y2], convert it
                if len(bounding_box) == 2:
                    bounding_box = [bounding_box[0] - 20, bounding_box[1]-20, bounding_box[0] + 20, bounding_box[1] + 20]
                elif len(bounding_box) != 4:
                    # If the bounding box is not in the expected format, set it to a default value
                    bounding_box = [0, 0,RESIZE_LENGTH ,RESIZE_LENGTH]
                pred_2d_bounding_box = {"min_x": bounding_box[0], "min_y": bounding_box[1], "max_x": bounding_box[2], "max_y": bounding_box[3]}
                
                if self.ThreeD_BBOX:
                    if EXPLORE:
                        selected_image = scannet_dataset.color_images[selected_index]
                        if isinstance(selected_image, Image.Image):
                            pil_image = selected_image
                        else:
                            pil_image = Image.fromarray(selected_image)
                        draw = ImageDraw.Draw(pil_image)
                        img_width, img_height = pil_image.size

                        # Rescale bounding box from RESIZE_LENGTHxRESIZE_LENGTH to the actual image size
                        try:
                            scale_x = 1
                            scale_y = 1
                            left, up, right, down = map(int, bounding_box)
                            left_px = int(left * scale_x)
                            right_px = int(right * scale_x)
                            up_px = int(up * scale_y)
                            down_px = int(down * scale_y)
                            
                            if left_px > right_px:
                                left_px, right_px = right_px, left_px
                            if up_px > down_px:
                                up_px, down_px = down_px, up_px
                        except:
                            left_px = 1
                            right_px = img_width - 1
                            up_px = 1
                            down_px = img_height - 1

                        try:
                            # Draw the bounding box
                            draw.rectangle([(left_px, up_px), (right_px, down_px)], outline="red", width=3)
                            selected_image = np.array(pil_image)

                            # Save the image with the bounding box
                            output_image_path = os.path.join(output_dir, f"{target.replace(' ', '_')}.png")
                            pil_image.save(output_image_path)
                            print(f"Saved image with bounding box to: {output_image_path}")
                        except Exception as e:
                            print(f"Error: Invalid bounding box or failed to save image. {e}")
                            return 3000, 0.0
                    
                    

                    # Create a dictionary to store point cloud coordinates for each label
                    # Create a dictionary to store point cloud coordinates and bounding boxes for each label
                    
                        
                    # we have image_index, bounding_box and scannet_dataset(depth, intrinsics, poses)
                    # want to get the point cloud in the bounding box

                    if EXPLORE:
                        predicted_3d_bounding_box = self.get_point_cloud_in_2d_bounding_box(
                            selected_index, pred_2d_bounding_box, modified_dataset
                        )
                    else:
                        # Please convert the bounding box and depth_map to this bounding box for non EXPLORE 
                        modified_dataset = copy.deepcopy(scannet_dataset)

                        # Replace the depth image at best_camera_idx with our computed depth map
                        modified_dataset.depth_images[best_camera_idx] = depth_map.cpu().numpy() if isinstance(depth_map, torch.Tensor) else depth_map
                        
                        # Replace the intrinsics with our zoomed intrinsics for proper backprojection
                        original_intrinsic = modified_dataset.intrinsic_depth.copy()
                        modified_intrinsic = original_intrinsic.copy()
                        # Apply the same zoom factor as was used to generate the depth map
                        modified_intrinsic[0, 0] *= self.ZOOM_RATIO  # Reduce focal length (zoom out)
                        modified_intrinsic[1, 1] *= self.ZOOM_RATIO
                        modified_dataset.intrinsic_depth = modified_intrinsic
                        modified_dataset.poses[best_camera_idx] = zoomed_extrinsics.cpu().numpy() if isinstance(zoomed_extrinsics, torch.Tensor) else zoomed_extrinsics
                        
                        # Get the point cloud and 3D bounding box using the best camera view, generated depth map, and modified intrinsics
                        predicted_3d_bounding_box = self.get_point_cloud_in_2d_bounding_box(
                            best_camera_idx, pred_2d_bounding_box, modified_dataset
                        )
                    # import pdb;pdb.set_trace()
                    
                    pred_bounding_box = predicted_3d_bounding_box["refined_bounding_box"]

                    # Find the label with the highest IoU
                    best_labels_with_iou = self.find_highest_iou_label(pred_bounding_box, label_to_data)

                    # Near line 797 in your forward method
                    scene = batched_inputs[0]["pose_file_names"][0].split("/")[-3]
                    output_dir = os.path.join("output", f"scannet_exp_{self.experiment_timestamp}", f"scene_{scene}")
                    viz_output_path = os.path.join(output_dir, f"{target.replace(' ', '_')}_3d_boxes_with_cameras.png")

                    gt_id = batched_inputs[0]["sr3d_data"][0]["target_id"]
                    
                    selected_index = best_labels_with_iou["ids"][0]
                    VIZ_BBOXES = False
                    if VIZ_BBOXES:
                        print(f"Visualizing 3D bounding boxes with camera poses...")
                        self.visualize_3d_bounding_boxes(
                            label_to_data, 
                            pred_bounding_box, 
                            viz_output_path,
                            scannet_dataset=scannet_dataset,
                            highlight_index=selected_index,
                            gt_id=gt_id
                        )
                        print(f"Visualization saved to: {viz_output_path}")
                
                    
                    
                    print(f"Best labels with IoU: {best_labels_with_iou}")
                    
                    # get gt
                    
                    print(f"GT ID: {gt_id}")
                    
                    output = [best_labels_with_iou]
                # import pdb;pdb.set_trace()
                
                else: # not threeD bbox
                    # not using 3D bbox
                    # instead, we are going to project all the gt_bboxes back to 2D on this camera
                    # and then get the best labels with the top-5 highest IoU
                    
                    # get pointcloud for object_ids:
                    
                    print("Projecting all object point clouds to 2D bounding boxes...")
                    
                    if EXPLORE:
                        camera_idx = selected_index
                        if zoomed_view_data is not None:
                            # Use the zoomed camera parameters from exploration
                            zoomed_pose, zoomed_intrinsics = zoomed_view_data
                            camera_pose = zoomed_pose.to(self.device) if isinstance(zoomed_pose, torch.Tensor) else torch.tensor(zoomed_pose).float().to(self.device)
                            camera_intrinsics = zoomed_intrinsics.to(self.device) if isinstance(zoomed_intrinsics, torch.Tensor) else torch.tensor(zoomed_intrinsics).float().to(self.device)
                            
                            # Render a new image from the point cloud using the zoomed-out camera
                            H, W = scannet_dataset.depth_images[camera_idx].shape
                            rendered_image = multi_turn_explore.render_image_from_point_cloud(
                                zoomed_pose, zoomed_intrinsics, H, W
                            )
                            
                            if rendered_image is not None:
                                # Use the rendered image instead of the original
                                selected_image = rendered_image
                                print("Using rendered image from zoomed-out camera view")
                            else:
                                # Fall back to original image if rendering fails
                                selected_image = scannet_dataset.color_images[camera_idx]
                                print("Failed to render image from point cloud, using original image")
                        else:
                            # Use the original camera parameters and image
                            camera_pose = torch.tensor(scannet_dataset.poses[camera_idx]).float().to(self.device)
                            camera_intrinsics = torch.tensor(scannet_dataset.intrinsic_depth).float().to(self.device)
                            selected_image = scannet_dataset.color_images[camera_idx]
                            print("Using original camera parameters and image (no zoomed data)")
                    else:
                        camera_idx = best_camera_idx
                        camera_pose = zoomed_extrinsics
                        camera_intrinsics = zoomed_intrinsics
                        
                        # For non-EXPLORE mode, we should also render from point cloud
                        H, W = scannet_dataset.depth_images[camera_idx].shape
                        rendered_image = multi_turn_explore.render_image_from_point_cloud(
                            camera_pose.cpu().numpy(), camera_intrinsics.cpu().numpy(), H, W
                        )
                        
                        if rendered_image is not None:
                            selected_image = rendered_image
                            print("Using rendered image from zoomed-out camera view")
                        else:
                            selected_image = scannet_dataset.color_images[camera_idx]
                            print("Failed to render image from point cloud, using original image")
                    
                    # Get unique object IDs
                    unique_object_ids = [label for label in label_to_data.keys() if label != -1]
                    
                    # Dictionary to store 2D bounding boxes for each object
                    object_2d_bboxes = {}
                    
                    H, W = scannet_dataset.depth_images[selected_index].shape
                    # Image dimensions
                    img_height, img_width = H, W
                    
                    # Project each object's point cloud to 2D
                    for obj_id in unique_object_ids:
                        # Get 3D points for this object
                        obj_mask = scannet_labels == obj_id
                        obj_points = scannet_coords[obj_mask]
                        
                        if len(obj_points) == 0:
                            continue
                            
                        # Add homogeneous coordinate for transformation
                        obj_points_homogeneous = torch.cat(
                            [obj_points.to(self.device), torch.ones(obj_points.shape[0], 1, device=self.device)], dim=1
                        )
                        
                        # Transform points from world to camera space
                        cam_to_world = camera_pose.to(self.device)
                        world_to_cam = torch.inverse(cam_to_world).to(self.device)
                        camera_points = (world_to_cam @ obj_points_homogeneous.T).T
                        
                        # Filter points in front of camera
                        front_mask = camera_points[:, 2] > 0
                        if not front_mask.any():
                            continue  # Skip if no points are in front of the camera
                            
                        camera_points = camera_points[front_mask]
                        
                        # Project to 2D using camera intrinsics
                        points_2d = torch.zeros((camera_points.shape[0], 2), device=self.device)
                        points_2d[:, 0] = camera_intrinsics[0, 0] * (camera_points[:, 0] / camera_points[:, 2]) + camera_intrinsics[0, 2]
                        points_2d[:, 1] = camera_intrinsics[1, 1] * (camera_points[:, 1] / camera_points[:, 2]) + camera_intrinsics[1, 2]
                        
                        # Filter points within image bounds
                        in_image_mask = (
                            (points_2d[:, 0] >= 0) & 
                            (points_2d[:, 0] < img_width) & 
                            (points_2d[:, 1] >= 0) & 
                            (points_2d[:, 1] < img_height)
                        )
                        
                        if not in_image_mask.any():
                            continue  # Skip if no points are within image bounds
                            
                        valid_points_2d = points_2d[in_image_mask]
                        
                        # Compute 2D bounding box
                        min_x = valid_points_2d[:, 0].min().item()
                        min_y = valid_points_2d[:, 1].min().item()
                        max_x = valid_points_2d[:, 0].max().item()
                        max_y = valid_points_2d[:, 1].max().item()
                        
                        # Store the 2D bounding box
                        object_2d_bboxes[obj_id] = {
                            "min_x": min_x,
                            "min_y": min_y,
                            "max_x": max_x,
                            "max_y": max_y,
                            "visible_points": len(valid_points_2d),
                            "total_points": len(obj_points)
                        }
                    
                    # Get ground truth object ID
                    gt_id = batched_inputs[0]["sr3d_data"][0]["target_id"]
                    
                    # Check if the ground truth object is visible in the selected view
                    if gt_id in object_2d_bboxes:
                        gt_bbox = object_2d_bboxes[gt_id]
                        print(f"Ground truth object {gt_id} is visible with bbox: {gt_bbox}")
                        
                        # Compute IoU between predicted 2D bbox and all object bboxes
                        iou_scores = {}
                        for obj_id, obj_bbox in object_2d_bboxes.items():
                            iou = self.compute_2d_iou(pred_2d_bounding_box, obj_bbox)
                            iou_scores[obj_id] = iou
                            
                        # Sort by IoU in descending order
                        sorted_obj_ids = sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)
                        
                        # Create result in the same format as before
                        best_labels_with_iou = {
                            "ids": [obj_id for obj_id, _ in sorted_obj_ids[:5]],
                            "ious": [iou for _, iou in sorted_obj_ids[:5]]
                        }
                        
                        output = [best_labels_with_iou]
                        
                        # Visualize all projected bounding boxes (optional)
                        if 1:
                            # Create a copy of the image for visualization
                            
                            if isinstance(selected_image, Image.Image):
                                viz_image = selected_image.copy()
                            else:
                                viz_image = Image.fromarray(selected_image.copy())
                            
                            draw = ImageDraw.Draw(viz_image)
                            
                            # Draw each object's bounding box with different colors
                            colors = {
                                gt_id: (255, 0, 0),  # Red for ground truth
                                best_labels_with_iou["ids"][0]: (0, 255, 0)  # Green for top prediction
                            }
                            
                            for obj_id, bbox in object_2d_bboxes.items():
                                color = colors.get(obj_id, (0, 0, 255))  # Blue for other objects
                                draw.rectangle(
                                    [(bbox["min_x"], bbox["min_y"]), (bbox["max_x"], bbox["max_y"])],
                                    outline=color,
                                    width=2
                                )
                                draw.text((bbox["min_x"], bbox["min_y"]), f"ID: {obj_id}", fill=color)
                            
                            # Save visualization
                            viz_path = os.path.join(output_dir, f"{target.replace(' ', '_')}_all_projections.png")
                            viz_image.save(viz_path)
                            print(f"Saved visualization of all projected objects to: {viz_path}")
                    else:
                        print(f"Ground truth object {gt_id} is not visible in the selected view")

                        if isinstance(selected_image, Image.Image):
                            viz_image = selected_image.copy()
                        else:
                            viz_image = Image.fromarray(selected_image.copy())
                        viz_path = os.path.join(output_dir, f"{target.replace(' ', '_')}_no_gt.png")
                        viz_image.save(viz_path)
                        
                        # add a fall back
                        output = [{"ids": [-1, -1, -1, -1, -1], "ious": [0.0, 0.0, 0.0, 0.0, 0.0]}]
                        
                        return 4000, 0.0  # Error code 4000: Wrong image (GT not visible)
                        
                    
                    
                    
                    
            
            else: # not USE_BOUNDING_BOX, only use a point
                # get the point in the picture
                
                coordinates_match = re.search(r'\((\d+),\s*(\d+)\)', agent_answer)
                coordinates_match_2 = re.search(r'\[(\d+),\s*(\d+)\]', agent_answer)
                if coordinates_match:
                    x_coord = int(coordinates_match.group(1))
                    y_coord = int(coordinates_match.group(2))
                    print(f"Extracted coordinates: x={x_coord}, y={y_coord}")
                    coordinates = [x_coord, y_coord]
                elif coordinates_match_2:
                    x_coord = int(coordinates_match_2.group(1))
                    y_coord = int(coordinates_match_2.group(2))
                    print(f"Extracted coordinates: x={x_coord}, y={y_coord}")
                    coordinates = [x_coord, y_coord]
                else:
                    # Try to extract bounding box format
                    try:
                        # First try to find JSON format
                        json_match = re.search(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', agent_answer)
                        if json_match:
                            x_min = int(json_match.group(1))
                            y_min = int(json_match.group(2))
                            x_max = int(json_match.group(3))
                            y_max = int(json_match.group(4))
                        else:
                            # Try to find regular bbox format
                            bbox_match = re.search(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', agent_answer)
                            if bbox_match:
                                x_min = int(bbox_match.group(1))
                                y_min = int(bbox_match.group(2))
                                x_max = int(bbox_match.group(3))
                                y_max = int(bbox_match.group(4))
                            else:
                                raise ValueError("No bounding box found")
                        
                        # Calculate middle point
                        x_coord = (x_min + x_max) // 2
                        y_coord = (y_min + y_max) // 2
                        print(f"Extracted middle point from bounding box: x={x_coord}, y={y_coord}")
                        coordinates = [x_coord, y_coord]
                    except:
                        print("No coordinates or bounding box found")
                        # Fallback to default coordinates
                        x_coord = 320
                        y_coord = 240
                        coordinates = [320, 240]  # Default to image center
                
                if EXPLORE:
                    print("coordinates: ", coordinates)
                    
                    # Get the depth map from the selected camera view
                    selected_depth = scannet_dataset.depth_images[selected_index]
                    
                    # Get camera intrinsics and pose
                    intrinsic = scannet_dataset.intrinsic_depth
                    c2w = torch.tensor(scannet_dataset.poses[selected_index]).float().to(self.device)
                    
                    # Ensure coordinates are within bounds
                    img_height, img_width = selected_depth.shape
                    x_coord = max(0, min(x_coord, img_width - 1))
                    y_coord = max(0, min(y_coord, img_height - 1))
                    
                    # Draw and save the point on the selected image
                    selected_image = scannet_dataset.color_images[selected_index]
                    if isinstance(selected_image, Image.Image):
                        pil_image = selected_image
                    else:
                        pil_image = Image.fromarray(selected_image)
                    
                    draw = ImageDraw.Draw(pil_image)
                    point_radius = 5
                    draw.ellipse(
                        [(x_coord - point_radius, y_coord - point_radius), 
                        (x_coord + point_radius, y_coord + point_radius)],
                        fill="red", outline="red"
                    )
                    
                    # Save the image with the point
                    output_image_path = os.path.join(output_dir, f"{target.replace(' ', '_')}_selected_point.png")
                    pil_image.save(output_image_path)
                    print(f"Saved image with selected point to: {output_image_path}")
                    
                    # Get depth at the selected pixel
                    depth_value = selected_depth[y_coord, x_coord]
                    
                    # If depth is invalid at the exact pixel, find nearest valid depth
                    if depth_value <= 0:
                        search_radius = 1
                        valid_depth_found = False
                        
                        while not valid_depth_found and search_radius < 20:
                            for dy in range(-search_radius, search_radius + 1):
                                for dx in range(-search_radius, search_radius + 1):
                                    ny, nx = y_coord + dy, x_coord + dx
                                    if 0 <= ny < img_height and 0 <= nx < img_width:
                                        if selected_depth[ny, nx] > 0:
                                            depth_value = selected_depth[ny, nx]
                                            valid_depth_found = True
                                            break
                                if valid_depth_found:
                                    break
                            search_radius += 1
                        
                        if not valid_depth_found:
                            print("Could not find valid depth near the selected point")
                            # Fallback to a reasonable default depth
                            depth_value = 2.0
                    
                    # Unproject to 3D space
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    
                    # Get 3D point in camera coordinates
                    x_cam = (x_coord - cx) * depth_value / fx
                    y_cam = (y_coord - cy) * depth_value / fy
                    z_cam = depth_value
                    
                    # Convert to world coordinates
                    point_cam = torch.tensor([[x_cam, y_cam, z_cam, 1.0]]).float().to(self.device)
                    point_world = torch.matmul(c2w, point_cam.T).T
                    point_3d = point_world[0, :3].unsqueeze(0)  # Shape [1, 3]
                    
                    print(f"Unprojected 3D point: {point_3d}")
                    
                    # Find the nearest boxes for this 3D point
                    nearest_boxes = self.find_nearest_boxes_for_points(point_3d, label_to_data)
                    
                    # Filter out -1 ID (background/invalid)
                    filtered_predictions = [item for item in nearest_boxes["ranked_labels"] if item["id"] != -1]
                    nearest_boxes["ranked_labels"] = filtered_predictions
                    
                    # Get top prediction and ranked predictions
                    top_prediction = nearest_boxes["most_common_label"]
                    ranked_predictions = nearest_boxes["ranked_labels"]
                    
                    # Print results for debugging
                    gt_id = batched_inputs[0]["sr3d_data"][0]["target_id"]
                    print(f"Ground truth ID: {gt_id}")
                    print(f"Top prediction: {top_prediction}")
                    print(f"Top 5 predictions: {ranked_predictions[:5]}")
                    
                    # Format output similar to the non-EXPLORE case
                    best_labels_with_iou = {
                        "ids": [item["id"] for item in nearest_boxes["ranked_labels"][:5]],
                        "ious": [item["score"] for item in nearest_boxes["ranked_labels"][:5]]
                    }
                    
                    # Return in the same format as the non-EXPLORE case
                    output = [best_labels_with_iou]
                    return output
                
                
                
                
                else:
                    # Draw the point on the selected camera's image and save it
                    selected_image = scannet_dataset.color_images[best_camera_idx]
                    if isinstance(selected_image, Image.Image):
                        pil_image = selected_image
                    else:
                        pil_image = Image.fromarray(selected_image)

                    draw = ImageDraw.Draw(pil_image)
                    img_width, img_height = pil_image.size

                    # Ensure the coordinates are within bounds
                    x_coord = max(0, min(x_coord, img_width - 1))
                    y_coord = max(0, min(y_coord, img_height - 1))

                    # Draw the point
                    point_radius = 5
                    draw.ellipse(
                        [(x_coord - point_radius, y_coord - point_radius), (x_coord + point_radius, y_coord + point_radius)],
                        fill="red",
                        outline="red",
                    )

                    # Save the image with the point
                    output_image_path = os.path.join("./viz_pc_occluded", f"{target.replace(' ', '_')}_selected_point.png")
                    if self.DEBUG:
                        pil_image.save(output_image_path)
                        print(f"Saved image with selected point to: {output_image_path}")
                    
                    
                    # Find nearby 3D points
                    nearby_points = []
                    radius = 0.03  # Adjust radius as needed
                    c2w = zoomed_extrinsics
                    intrinsics = zoomed_intrinsics
                    
                    while len(nearby_points) == 0 and radius < 0.3:
                        nearby_points, voxel_indices = self.get_3d_points_near_pixel(
                            coordinates, 
                            zoomed_points_cam,
                            zoomed_valid_mask, 
                            zoomed_depth_values,
                            voxelized_pc, 
                            p2v,
                            radius=radius,  # Adjust radius as needed
                            c2w=c2w,
                            intrinsics=intrinsics,
                        )
                        radius += 0.01
                    
                    
                    print(f"Found {len(nearby_points)} unique 3D points near pixel ({x_coord}, {y_coord})")
                    if len(nearby_points) == 0:
                        # dummy point
                        nearby_points = torch.tensor([[0, 0, 0]]).float().to(self.device)
                    
                    if self.DEBUG:
                        """Do re-projection"""
                        def point_reprojection(points, poses, intrinsics):
                            """Project 3D points to camera views and check visibility"""
                            if points.dtype != torch.float32:
                                points = points.float()
                            if poses.dtype != torch.float32:
                                poses = poses.float()
                            
                            # Get camera-to-world projection matrix
                            depth2img = intrinsics @ poses.inverse()
                            
                            # Add homogeneous coordinate
                            points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
                            
                            
                            B, num_query = points.shape[:2]
                            num_cam = depth2img.shape[1]
                            
                            points = points[:, None].repeat(1, num_cam, 1, 1)
                            
                            # Transform points to camera space
                            points_cam = torch.einsum('bnij,bnmj->bnmi', depth2img, points)
                            
                            eps = 1e-5
                            
                            # Check which points are in front of camera
                            valid_mask = (points_cam[..., 2:3] > eps)
                            
                            # Store the depth values before normalization
                            depth_values = points_cam[..., 2:3].clone()
                            
                            # Normalize by depth
                            points_cam = points_cam[..., 0:2] / torch.maximum(
                                points_cam[..., 2:3], torch.ones_like(points_cam[..., 2:3]) * eps)

                            return points_cam, valid_mask, depth_values
                        
                        c2w = zoomed_extrinsics
                        intrinsics = zoomed_intrinsics
                        points_cam, valid_mask, depth_values = point_reprojection(
                            nearby_points, 
                            c2w[None, None],
                            intrinsics[None, None]
                        )
                        print(points_cam)
                        print(valid_mask)
                        print(depth_values)
                        # import pdb;pdb.set_trace()
                        
                        print(zoomed_intrinsics==intrinsics)

                    """Do re-projection end"""

                    if len(nearby_points) > 0:
                        
                        # Create indices for sampling then use them to select from tensor
                        indices = random.sample(range(len(nearby_points)), min(20, len(nearby_points)))
                        nearby_points = nearby_points[indices]
                        
                        if self.DEBUG:
                            camera_position = zoomed_extrinsics.cpu()

                            c2w = zoomed_extrinsics
                            intrinsics = zoomed_intrinsics
                            
                            
                            # Call the function with camera position
                            viz_path = self.visualize_nearby_points_in_pointcloud(
                                batched_inputs[0]["scannet_coords"],
                                batched_inputs[0]["scannet_color"],
                                nearby_points,
                                "./viz_pc_occluded",
                                target,
                                camera_position=camera_position
                            )

                    nearest_boxes = self.find_nearest_boxes_for_points(nearby_points, label_to_data)
                    filtered_predictions = [item for item in nearest_boxes["ranked_labels"] if item["id"] != -1]
                    nearest_boxes["ranked_labels"] = filtered_predictions
                    top_prediction = nearest_boxes["most_common_label"]
                    ranked_predictions = nearest_boxes["ranked_labels"]

                    # Compare with ground truth
                    gt_id = batched_inputs[0]["sr3d_data"][0]["target_id"]
                    print(f"Ground truth ID: {gt_id}")
                    print(f"Top prediction: {top_prediction}")
                    print(f"Top 5 predictions: {ranked_predictions[:5]}")
                    
                    # import pdb;pdb.set_trace()
                
                best_labels_with_iou = {
                    "ids": [item["id"] for item in nearest_boxes["ranked_labels"][:5]],
                    "ious": [item["score"] for item in nearest_boxes["ranked_labels"][:5]]
                }

                # Return in the same format as the bounding box approach
                output = [best_labels_with_iou]
                    
                    
            
            return output

        