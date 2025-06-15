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

from feature_map_tools.qwen_multicam import Qwen2_5_Projected3D
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from feature_map_tools.explore_qwen_vl import run_qwen_for_boundingbox, run_qwen


from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import cv2


import google.generativeai as genai
import vertexai
import time
import json
import time


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
        self.BIRD_VIEW_CAM = True
        self.VOXELIZE = False
        self.EXPLORE = True
        self.USE_BOUNDING_BOX = False
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

                

                # Add timestamp to the name for better tracking
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_with_timestamp = f"{name}_{timestamp}"
                
                kwargs = dict()
                kwargs['id'] = name_with_timestamp

                wandb.init(
                    entity=cfg.WANDB_ENTITY,
                    project=cfg.WANDB_PROJECT,
                    sync_tensorboard=True,
                    name=name_with_timestamp,
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
        
        if 0:
            if self.EXPLORE:
                self.agent = AgentQwen2_5(temperature=temperature, gpu_id=gpu_id)
            else:
                model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
                self.model = Qwen2_5_Projected3D.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=f"cuda:{gpu_id}" if gpu_id is not None else "auto",
                    attn_implementation="flash_attention_2",
                )
                self.processor = AutoProcessor.from_pretrained(model_name)
        else: # use gemini api

            self.GOOGLE_API_KEY = 'AIzaSyAMcwWOQ17KKXvgnBPpXb92VwCZlkcphfk'
            genai.configure(api_key=self.GOOGLE_API_KEY)
            project_id = "geminiplanning"
            vertexai.init(project=project_id, location="us-central1")
            
            # Initialize the Gemini model
            self.gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
            # Setup output directory for results
            os.makedirs("gemini_results", exist_ok=True)
            self.output_json = "gemini_results/gemini_outputs.json"
            
            # Create output JSON if it doesn't exist
            if not os.path.exists(self.output_json):
                with open(self.output_json, 'w') as f:
                    json.dump({}, f)
        # self.categories = {k: v for k, v in enumerate(self.metadata.thing_classes)}
        
        
        self.grounding_dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self.grounding_dino_model = AutoModelForZeroShotObjectDetection .from_pretrained("IDEA-Research/grounding-dino-base").to(torch.cuda.current_device())
        

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
        
        
        if 1:
            # images = []
            # for video in batched_inputs:
            #     for image in video["images"]:
            #         images.append(image.to(self.device))
            bs = len(batched_inputs)
            
            assert bs == 1, "Batch size should be 1 for now"
            v = len(batched_inputs[0]["images"])
            
            if not self.training:
                print(v)
                # print(batched_inputs[0]["sr3d_data"][0]["target_id"])
                # print(batched_inputs[0]["sr3d_data"][0]["anchor_ids"])
                # import pdb;pdb.set_trace()
            
            
            
            images = batched_inputs[0]["images"]
            depths = batched_inputs[0]["depths"]
            poses = batched_inputs[0]["poses"]
            intrinsics = batched_inputs[0]["intrinsics"]
            
            target_name = batched_inputs[0]["sr3d_data"][0]["target_name"]
            anchor_names = batched_inputs[0]["sr3d_data"][0]["anchors_names"]

            # Prepare text prompts for Grounding DINO
            prompts = [target_name] + anchor_names
            text = " ".join([f"{prompt.lower()}." for prompt in prompts])
            
            # text = text.replace(". ", ".")
            # text = text.replace(" ", "_")
            # text = text.replace(".", ". ")
            text = text.replace("picture", "painting")

            
            answer_list = []
            # Iterate over images and generate bounding boxes
            for i, image in enumerate(images):
                # Convert image to PIL format
                pil_image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype("uint8"))

                # Preprocess the image and text prompts

                inputs = self.grounding_dino_processor(images=pil_image, text=text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.grounding_dino_model(**inputs)

                results = self.grounding_dino_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[pil_image.size[::-1]]
                )
                
                answer_list.append(results)
                # [{'scores': tensor([0.6549], device='cuda:0'), 'boxes': tensor([[438.4309,  87.5565, 639.1539, 326.4796]], device='cuda:0'), 'text_labels': ['plant'], 'labels': ['plant']}]?
                
                # [{'scores': tensor([0.6449, 0.4100], device='cuda:0'), 'boxes': tensor([[427.1261, 188.8412, 632.6815, 410.9197], [123.1609, 271.0362, 255.4935, 477.6711]], device='cuda:0'), 'text_labels': ['plant', 'chair'], 'labels': ['plant', 'chair']}]
                
                os.makedirs("detection_results", exist_ok=True)
    
                # Generate output path
                output_path = f"detection_results/frame_{i}_{text.replace(' ', '_').replace('.', '')}_detection.png"
                
                # Visualize and save detections
                # visualize_and_save_detections(pil_image, results, output_path)
                # import pdb;pdb.set_trace()
            
            
            
            assert len(answer_list) == v
            # get a canvas
            import pdb;pdb.set_trace()
            CANVAS_TIMES = 2
            H,W = images[0].shape[-2:]
            canvas = torch.zeros((CANVAS_TIMES*H,CANVAS_TIMES*W,3), dtype=torch.float32, device=images[0].device)
            
            canvas_occupancy = torch.zeros((CANVAS_TIMES*H,CANVAS_TIMES*W), dtype=torch.long, device=images[0].device)
            
            # Start with the first camera's parameters
            
           
            canvas_intrinsics = intrinsics[0].clone()

            # 1. Adjust intrinsics to increase FOV (reduce focal length)
            canvas_intrinsics[0, 0] = canvas_intrinsics[0, 0] / CANVAS_TIMES  # fx
            canvas_intrinsics[1, 1] = canvas_intrinsics[1, 1] / CANVAS_TIMES  # fy

            # 2. Adjust principal point for the larger canvas
            canvas_intrinsics[0, 2] = canvas_intrinsics[0, 2] * CANVAS_TIMES  # cx
            canvas_intrinsics[1, 2] = canvas_intrinsics[1, 2] * CANVAS_TIMES  # cy

            # 3. Move the camera backwards along its viewing direction
            # Extract the camera's forward direction (negative z-axis in camera coordinates)
            if 0:
                canvas_camera = poses[0].clone()  # c2w transformation
                camera_forward = -canvas_camera[:3, 2]  # This is the viewing direction
                camera_forward = camera_forward / torch.norm(camera_forward)  # Normalize
                # Distance to move backward - proportional to canvas size
                move_distance = 0.0 * CANVAS_TIMES  # You can adjust this factor as needed # currently zero
                # Update camera position by moving backwards
                canvas_camera[:3, 3] = canvas_camera[:3, 3] - camera_forward * move_distance
            else:
                # use camera that is high above in the sky
                # take the average of all camera positions
                camera_positions = torch.stack([pose[:3, 3] for pose in poses], dim=0)
                camera_positions = camera_positions.mean(dim=0)
                # Move the camera to a fixed height above the average position
                fixed_height = 2.0  # Adjust this value as needed
                camera_positions[1] += fixed_height
                # Create a new camera pose matrix
                canvas_camera = torch.eye(4, device=camera_positions.device)

                # Set the translation component (camera position)
                canvas_camera[:3, 3] = camera_positions

                # Create rotation matrix for looking down:
                # - Forward direction (Z) points downward (negative Y in world space)
                # - Up direction points toward negative Z in world space
                # - Right direction points toward positive X in world space

                # Forward direction (Z-axis) points down (-Y in world)
                canvas_camera[:3, 2] = torch.tensor([0.0, -1.0, 0.0], device=camera_positions.device)
                # Right direction (X-axis) points right (+X in world)
                canvas_camera[:3, 0] = torch.tensor([1.0, 0.0, 0.0], device=camera_positions.device) 
                # Up direction (Y-axis) points forward (+Z in world) - calculated as cross product
                canvas_camera[:3, 1] = torch.cross(canvas_camera[:3, 2], canvas_camera[:3, 0])
                

            canvas_objects = []
            for i in range(v):
                # Extract current image, depth, pose and detected objects
                current_image = images[i]
                current_depth = depths[i]
                current_pose = poses[i]  # Camera to world transformation
                current_intrinsic = intrinsics[i]
                current_detections = answer_list[i][0]  # Get detections for this view
                
                # Process each bounding box in the current view
                for box_idx, (box, score, label) in enumerate(zip(
                        current_detections['boxes'], 
                        current_detections['scores'], 
                        current_detections['text_labels'])):
                    
                    # Skip low confidence detections
                    if score < 0.4:
                        continue
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.cpu().tolist()
                    
                    # Calculate center point of the bounding box
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Get depth at the center point (handle edge cases)
                    center_x_int = min(max(int(center_x), 0), W-1)
                    center_y_int = min(max(int(center_y), 0), H-1)
                    depth_value = current_depth[center_y_int, center_x_int]
                    
                    # Skip if depth is invalid (zero or very large)
                    if depth_value < 0.01 or depth_value > 10.0:
                        continue
                    
                    # 1. Unproject to 3D space
                    # Convert to homogeneous coordinates
                    point_2d = torch.tensor([center_x, center_y, 1.0], device=current_depth.device)
                    
                    # Get 3D point in camera coordinates
                    point_3d_cam = torch.matmul(torch.inverse(current_intrinsic[:3, :3]), point_2d) * depth_value
                    point_3d_cam = torch.cat([point_3d_cam, torch.ones(1, device=point_3d_cam.device)])
                    
                    # Transform to world coordinates
                    point_3d_world = torch.matmul(current_pose, point_3d_cam)
                    
                    # 2. Project to canvas camera
                    # Transform from world to canvas camera
                    point_3d_canvas_cam = torch.matmul(torch.inverse(canvas_camera), point_3d_world)
                    
                    # Skip if point is behind the canvas camera
                    if point_3d_canvas_cam[2] <= 0:
                        continue
                    
                    # Project to canvas image plane
                    point_2d_canvas_homogeneous = torch.matmul(canvas_intrinsics[:3, :3], 
                                                            point_3d_canvas_cam[:3] / point_3d_canvas_cam[2])
                    canvas_x = point_2d_canvas_homogeneous[0].item()
                    canvas_y = point_2d_canvas_homogeneous[1].item()
                    
                    # 3. Check if point is inside canvas
                    if (canvas_x >= 0 and canvas_x < CANVAS_TIMES*W and 
                        canvas_y >= 0 and canvas_y < CANVAS_TIMES*H):
                        
                        # Calculate size on canvas (keep relative size based on depth)
                        depth_ratio = point_3d_canvas_cam[2] / depth_value
                        box_width = (x2 - x1) / depth_ratio
                        box_height = (y2 - y1) / depth_ratio
                        
                        box_width/= 2
                        box_height/= 2
                        
                        # Calculate canvas coordinates for the box
                        canvas_x1 = max(0, int(canvas_x - box_width/2))
                        canvas_y1 = max(0, int(canvas_y - box_height/2))
                        canvas_x2 = min(CANVAS_TIMES*W-1, int(canvas_x + box_width/2))
                        canvas_y2 = min(CANVAS_TIMES*H-1, int(canvas_y + box_height/2))
                        
                        # Check if the region is already occupied
                        # I think checking the mid point is enough
                        canvas_x1_m = max(0, int(canvas_x - box_width/4))
                        canvas_y1_m = max(0, int(canvas_y - box_height/4))
                        canvas_x2_m = min(CANVAS_TIMES*W-1, int(canvas_x + box_width/4))
                        canvas_y2_m = min(CANVAS_TIMES*H-1, int(canvas_y + box_height/4))
                        
                        region_occupied = canvas_occupancy[canvas_y1_m:canvas_y2_m, canvas_x1_m:canvas_x2_m].float().mean() > 0.5
                        
                        if not region_occupied and (canvas_x2 - canvas_x1) > 5 and (canvas_y2 - canvas_y1) > 5:
                            # Extract the object from the original image
                            obj_img = current_image[:, int(y1):int(y2), int(x1):int(x2)].permute(1, 2, 0)
                            
                            # Resize to fit the canvas region
                            obj_img_resized = F.interpolate(
                                obj_img.permute(2, 0, 1).unsqueeze(0),
                                size=(canvas_y2-canvas_y1, canvas_x2-canvas_x1),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(0).permute(1, 2, 0)
                            
                            # Paste the object onto the canvas
                            canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = obj_img_resized
                            
                            # Mark this region as occupied
                            canvas_occupancy[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = i + 1  # Using view index + 1 as marker
                            # Add text label in the center of the image crop
                            # Mark this region as occupied with a unique ID
                            region_id = len(canvas_objects) + 1  # Use a unique ID for each object
                            canvas_occupancy[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = region_id
                            
                            # Store object information
                            canvas_objects.append({
                                'region_id': region_id,
                                'view_idx': i,
                                'box_idx': box_idx,
                                'label': label,
                                'score': score.item(),
                                'coords': (canvas_y1, canvas_x1, canvas_y2, canvas_x2)
                            })
                            # text_color = torch.tensor([1.0, 1.0, 1.0], device=canvas.device)  # White text
                            # bg_color = torch.tensor([0.0, 0.0, 0.0], device=canvas.device)    # Black background

                            # # Calculate center position for the text
                            # text_y = canvas_y1 + (canvas_y2 - canvas_y1) // 2
                            # text_x = canvas_x1 + (canvas_x2 - canvas_x1) // 2

                            # # Create a semi-transparent background for text
                            # label_width = min(100, canvas_x2 - canvas_x1)
                            # label_height = 20
                            # bg_y1 = max(0, text_y - label_height // 2)
                            # bg_y2 = min(CANVAS_TIMES*H-1, text_y + label_height // 2)
                            # bg_x1 = max(0, text_x - label_width // 2)
                            # bg_x2 = min(CANVAS_TIMES*W-1, text_x + label_width // 2)

                            # # Create semi-transparent black background for text
                            # alpha = 0.5
                            # original = canvas[bg_y1:bg_y2, bg_x1:bg_x2].clone()
                            # overlay = bg_color.unsqueeze(0).unsqueeze(0).expand_as(original)
                            # canvas[bg_y1:bg_y2, bg_x1:bg_x2] = original * (1 - alpha) + overlay * alpha

                            # # Add the label text (using color as marker, since we can't render text directly in tensors)
                            # # In final saved image, you could overlay actual text using PIL or OpenCV
                            # label_indicator = text_color.unsqueeze(0).unsqueeze(0).expand(5, label_width // 2, 3)
                            # text_y_pos = bg_y1 + (bg_y2 - bg_y1) // 2 - 2
                            # text_x_pos = bg_x1 + (bg_x2 - bg_x1) // 2 - label_width // 4
                            # if (text_y_pos + 5 <= bg_y2) and (text_x_pos + label_width // 2 <= bg_x2):
                            #     canvas[text_y_pos:text_y_pos+5, text_x_pos:text_x_pos+label_width//2] = label_indicator

                            # # After generating the canvas, before saving:
                            # # You can add text to the final image using PIL
                            # # Create a method to add text to the saved image with the actual label name
                            
                            # # Draw a small label above the object
                            # label_height = min(20, canvas_y1)
                            # if canvas_y1 > 5:
                            #     # Simple colored rectangle as label (would need PIL for actual text)
                            #     canvas[canvas_y1-label_height:canvas_y1, canvas_x1:min(canvas_x1+50, canvas_x2), :] = torch.tensor(
                            #         [0.9, 0.5, 0.1], device=canvas.device)  # Orange label
            
            # here my canvas is ready
            # save to local
                        
            # Create output directory if it doesn't exist
            os.makedirs("canvas_results", exist_ok=True)

            # Convert canvas tensor to numpy array
            canvas_np = canvas.cpu().numpy()

            # Scale to 0-255 range and convert to uint8
            if canvas_np.max() <= 1.0:
                canvas_np = (canvas_np * 255).astype(np.uint8)
            else:
                canvas_np = canvas_np.astype(np.uint8)

            # Convert to PIL for text rendering
            pil_image = Image.fromarray(canvas_np)
            draw = ImageDraw.Draw(pil_image)

            # Try to load a font
            try:
                # Try multiple common font files, with increased size
                font = ImageFont.truetype("arial.ttf", 24)  # Increased from 24 to 36
            except IOError:
                try:
                    # Try DejaVuSans which is often available on Linux
                    font = ImageFont.truetype("DejaVuSans.ttf", 24)
                except IOError:
                    try:
                        # Try another common font
                        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 24)
                    except IOError:
                        # If all else fails, use the default font but try to make it larger
                        font = ImageFont.load_default()
                        # Note: The default font might not support size adjustments

            # Add text labels for each occupied region
            # please sort the canvas_objects by their region area, the bigger, the first
            canvas_objects = sorted(canvas_objects, key=lambda x: (x['coords'][2] - x['coords'][0]) * (x['coords'][3] - x['coords'][1]), reverse=True)
            
            for index_num, obj_info in enumerate(canvas_objects):
                y1, x1, y2, x2 = obj_info['coords']
                
                # Get center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get the label for this specific object
                label_text = obj_info['label']
                label_text = f"{index_num}: {label_text}"
                
                # Measure the text size to properly position it
                text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:4]
                
                # Move text to the left by offsetting from center
                # You can adjust the offset value (currently 40% of text width)
                offset = int(text_width * 0.4)
                text_x = center_x - offset
                text_y = center_y
                
                # Draw text outline for better visibility
                draw.text((text_x-1, text_y-1), label_text, font=font, fill=(0,0,0))
                draw.text((text_x+1, text_y-1), label_text, font=font, fill=(0,0,0))
                draw.text((text_x-1, text_y+1), label_text, font=font, fill=(0,0,0))
                draw.text((text_x+1, text_y+1), label_text, font=font, fill=(0,0,0))
                
                # Draw actual text
                draw.text((text_x, text_y), label_text, font=font, fill=(255,255,255))

            # Create a filename with the timestamp and target object
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"canvas_results/canvas_{timestamp}_{target_name}.png"

            # Save the canvas image with text
            pil_image.save(filename)
            print(f"Canvas saved to {filename}")
            
            canvas_pil = pil_image.copy()
            
            
            observation = []
            sr3d_text = batched_inputs[0]["sr3d_data"][0]["text_caption"]
            # text = f"You are looking for {sr3d_text}. Now you are looking at an abstract map containing some relevant objects. Please tell me whether the target you are looking for is in the map or not. If yes, please tell me the object ID of the target you are looking for. Think step by step and caerefully verify whether your choice satisfies the question: {sr3d_text}."
            
            text = f"""You are given an abstract map showing objects from different camera views combined into a single image. Note that the patches precisely indicate the relative positions. If you see a patch between two other patches, it means that the object is located in the center of them in the real world.

            Your task: Identify if the target object described as '{sr3d_text}' appears in this map.

            Please analyze step-by-step:
            1. Review all labeled objects in the map (each has a number followed by its name)
            2. Consider which objects match or relate to the description '{sr3d_text}'
            3. Evaluate each candidate carefully and consider the relative positions of the patches

            If the target is not present:
            - Clearly state that the target object is not found
            - Explain which objects you considered and why they don't match
            
            If you find the target:
            - Specify which object ID number corresponds to '{sr3d_text}'
            - Explain why this object matches the description

            Be thorough and precise in your reasoning."""
            
            
            observation.append(pil_image)
            observation.append(text)
            
            # agent_answer = self.agent.generate(observation)
            
            chat_model = self.gemini_model.start_chat(history=[])
            
            # Upload the image to Gemini
            try:
                image_file = genai.upload_file(path=filename)
                
                # Wait for processing if needed
                while image_file.state.name == "PROCESSING":
                    print('.', end='')
                    time.sleep(3)
                    image_file = genai.get_file(image_file.name)
                
                if image_file.state.name == "FAILED":
                    raise ValueError(f"File upload failed: {image_file.state.name}")
                
                print(f"Uploaded canvas image as: {image_file.uri}")
                
                # Send the image and prompt to Gemini
                response = chat_model.send_message([image_file, text], request_options={"timeout": 300})
                
                # Get the response text
                agent_answer = response.candidates[0].content.parts[0].text
                
                # Save to output JSON
                with open(self.output_json, 'r') as f:
                    output_data = json.load(f)
                
                output_data[filename] = agent_answer
                
                with open(self.output_json, 'w') as f:
                    json.dump(output_data, f, indent=4)
                
                # Clean up
                genai.delete_file(image_file.name)

            except Exception as e:
                agent_answer = f"Error using Gemini API: {str(e)}"
                print(f"Gemini API error: {str(e)}")
            
            
            print (agent_answer)
            txt_filename = filename.replace('.png', '.txt')
            with open(txt_filename, 'w') as f:
                f.write(agent_answer)
            print(f"Agent answer saved to {txt_filename}")
            
            if self.cfg.USE_WANDB and (not is_dist_avail_and_initialized() or comm.is_main_process()):
                # Convert PIL image to numpy array for wandb
                canvas_np = np.array(canvas_pil)
                
                # Get the text caption from the input data
                text_caption = batched_inputs[0]["sr3d_data"][0]["text_caption"]
                
                # Create a unique step ID based on timestamp
                step_id = int(time.time())
                
                # Create a combined caption with both question and answer
                combined_caption = f"Question: {text_caption}\n\nAgent Response:\n{agent_answer}"
                
                # Log to wandb with the combined caption
                wandb.log({
                    "canvas_analysis": wandb.Image(
                        canvas_np, 
                        caption=combined_caption
                    ),
                    # Optional: Keep the HTML versions for better formatting in the UI
                    "text_and_response": wandb.Html(
                        f"<div style='margin-bottom:10px'><b>Question:</b> {text_caption}</div>"
                        f"<div><b>Agent Response:</b><br>{agent_answer}</div>"
                    ),
                }, step=step_id)
                
                print(f"Logged results to wandb (step: {step_id})")
            
        
            output = []
            
            return output













def visualize_and_save_detections(image, results, output_path, confidence_threshold=0.3):
    """
    Visualize the detected objects and save the image with bounding boxes.
    
    Args:
        image: PIL Image or tensor
        results: Detection results from Grounding DINO
        output_path: Path to save the output image
        confidence_threshold: Threshold for confidence scores
    """
    # Convert image from tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Convert to uint8 if normalized
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise TypeError("Unsupported image type")
    
    # Create a copy to draw on
    draw_image = pil_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # Color mapping for different classes (for consistent colors)
    color_map = {}
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Green (dark)
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
    ]
    
    for result in results:
        boxes = result['boxes'].cpu()
        scores = result['scores'].cpu()
        labels = result['text_labels']
        
        for box, score, label in zip(boxes, scores, labels):
            # Skip if confidence is below threshold
            if score < confidence_threshold:
                continue
                
            # Assign a consistent color for the class
            if label not in color_map:
                color_map[label] = colors[len(color_map) % len(colors)]
            color = color_map[label]
            
            # Extract box coordinates
            x1, y1, x2, y2 = box.tolist()
            
            # Draw bounding box
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            
            # Prepare label text with confidence score
            label_text = f"{label}: {score:.2f}"
            
            # Calculate text size and position it above the box
            text_width, text_height = draw.textbbox((0, 0), label_text, font=font)[2:4]
            text_position = (x1, max(0, y1 - text_height - 2))
            
            # Draw text background
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill=color
            )
            
            # Draw text
            draw.text(text_position, label_text, font=font, fill=(255, 255, 255))
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    draw_image.save(output_path)
    
    print(f"Visualization saved to {output_path}")
    return draw_image

# Example usage inside the model
def process_and_save_detections(images, results, output_dir="./detection_results"):
    """Process multiple images and save detection results"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (image, result) in enumerate(zip(images, results)):
        output_path = os.path.join(output_dir, f"detection_{i}.png")
        visualize_and_save_detections(image, [result], output_path)