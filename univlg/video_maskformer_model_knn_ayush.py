# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Tuple
import copy
import random


import detectron2.utils.comm as comm
import ipdb
import numpy as np
import torch
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
)
from univlg.utils import vis_utils
from univlg.utils.misc import is_dist_avail_and_initialized
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean
from transformers import AutoTokenizer,AutoProcessor

from .modeling.criterion import VideoSetCriterion
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom
from typing import TYPE_CHECKING, Any
from univlg.model_data import expand_tensor, load_scannet_data, prepare_targets, slice_tensor

if TYPE_CHECKING:
    from univlg.modeling.transformer_decoder.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder

logger = logging.getLogger(__name__)

from univlg.modeling_qwen2_5_vl_3dknn_ayush_advice import Qwen2_5_VLForConditionalGeneration
# from transformers import Qwen2_5_VLVisionModel, Qwen2_5_VLProcessor
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import StaticCache
from torch.nn import CrossEntropyLoss
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)

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
        USE_QWEN_3D = True
        
        self.this_device = torch.cuda.current_device()
        
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


        self.use_qwen_3d = USE_QWEN_3D
        if self.use_qwen_3d:
            from feature_map_tools.qwen3d_encoder import Qwen3DEncoder
            
            self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype=torch.bfloat16, 
                device_map=self.this_device,
                attn_implementation="flash_attention_2" #"flash_attention_2"
            )
            # for param in self.qwen_model.parameters():
            #     param.requires_grad = False
            # self.vision = Qwen2_5_VLVisionModel.from_pretrained("Qwen2.5-VL-3B-Instruct-Vision")
            # processor = Qwen2_5_VLProcessor.from_pretrained("Qwen2.5-VL-3B-Instruct-Vision")
            # # Initialize the 3D encoder
            
            # #############################################
            
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=16,                       # low-rank dimension
                lora_alpha=32,              # scaling
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj", "qkv", "visual.blocks.*.attn.proj", 
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            # wrap the model—only LoRA parameters will be trainable
            # check what parameter names are in the model
            
            self.qwen_model = get_peft_model(self.qwen_model, lora_config)
            
            # #############################################
            vision_encoder = self.qwen_model.visual.to(torch.bfloat16)
            # for param in vision_encoder.parameters():
            #     param.requires_grad = False
            self.qwen_3d_encoder = Qwen3DEncoder(
                model=vision_encoder,
                processor=self.qwen_processor,
                voxel_size=cfg.QWEN_3D.VOXEL_SIZE if hasattr(cfg, 'QWEN_3D') else 0.05,
                feature_dim=self.qwen_model.config.hidden_size,
                min_points_per_voxel=cfg.QWEN_3D.MIN_POINTS_PER_VOXEL if hasattr(cfg, 'QWEN_3D') else 1,
                device=self.this_device
            )

        # use Qwen instead
        self.visual_backbone: UniVLGVisualBackbone = visual_backbone
        # reduce 2048 to 256
        # self.connector = nn.Linear(
            # 2048, 256, bias=False
        # )
        self.connector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        self.mask_decoder: VideoMultiScaleMaskedTransformerDecoder = mask_decoder
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



    def encode_scene_3d(self, batched_inputs):
        """
        Process a batch of images with the Qwen3DEncoder.
        
        Args:
            batched_inputs: List of dictionaries containing batch data
            
        Returns:
            Tuple of (point_cloud, feature_cloud)
        """
        
        if not hasattr(self, 'qwen_3d_encoder'):
            raise AttributeError("Qwen3D encoder not initialized. Set USE_QWEN_3D=True in config.")
        
        images = []
        depths = []
        intrinsics = []
        extrinsics = []
        
        for video in batched_inputs:
            for i, image in enumerate(video["images"]):
                # Normalize images following the Qwen preprocessing
                images.append(image.to(self.device))
                depths.append(video["depths"][i].to(self.device))
                intrinsics.append(video["intrinsics"][i].to(self.device))
                extrinsics.append(video["poses"][i].to(self.device))
        
        # Optional text prompt
        text_prompt = None
        if 'sr3d_data' in batched_inputs[0] and len(batched_inputs[0]['sr3d_data']) > 0:
            text_prompt = batched_inputs[0]['sr3d_data'][0].get('text_caption', None)
        
        
        batch_size = 16 if self.training else 128
        
        # Process in batches
        point_cloud, feature_cloud = self.qwen_3d_encoder.process_scene(
            images=images,
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            text=text_prompt,
            batch_size=batch_size,
        )
        
        return point_cloud, feature_cloud

    def process_pointcloud_in_chunks(self, pointcloud, featurecloud, target_sentence, chunk_size=4096):
        """
        Process point cloud in chunks through Qwen decoder to handle memory constraints
        while preserving all points and their spatial relationships.
        
        Args:
            pointcloud: Point cloud coordinates [B, N, 3]
            featurecloud: Point features [B, N, C] 
            
            target_sentence: Text query
            chunk_size: Maximum points per chunk (default 4096)
            
        Returns:
            Tuple of (processed_pointcloud, processed_featurecloud)
        """
        batch_size, num_points, feature_dim = featurecloud.shape
        device = pointcloud.device
        
        # import pdb; pdb.set_trace() # k=8
        
        if num_points <= chunk_size:
            # attention_mask: Attention mask [B, N, k] where k is the number of neighbors
            attention_mask = self.build_knn_attention_mask_indices(pointcloud, k=8)
            
            
            # If small enough, process normally
            qwen_outputs = self.process_point_cloud_and_text_with_qwen(
                featurecloud, pointcloud, attention_mask, target_sentence
            )
            return pointcloud, qwen_outputs['point_features']
        
        # Calculate number of chunks needed
        num_chunks = (num_points + chunk_size - 1) // chunk_size
        
        processed_features_list = []
        processed_points_list = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_points)
            
            # Extract chunk
            chunk_points = pointcloud[:, start_idx:end_idx, :]
            chunk_features = featurecloud[:, start_idx:end_idx, :]
            
            # Create attention mask for this chunk
            # For simplicity, we'll rebuild the attention mask for each chunk
            # This maintains local spatial relationships within each chunk
            chunk_attention_mask = self.build_knn_attention_mask_indices(chunk_points, k=8)
            
            # Process chunk through Qwen
            if 1:
                qwen_outputs = self.process_point_cloud_and_text_with_qwen(
                    chunk_features, 
                    chunk_points, 
                    chunk_attention_mask, 
                    target_sentence
                )
                
                processed_chunk_features = qwen_outputs['point_features']
                
                processed_features_list.append(processed_chunk_features)
                processed_points_list.append(chunk_points)
                

        
        # Concatenate all processed chunks back together
        final_pointcloud = torch.cat(processed_points_list, dim=1)
        final_featurecloud = torch.cat(processed_features_list, dim=1)
        
        return final_pointcloud, final_featurecloud

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



    def upsample_pred_masks(
        self,
        mask_pred_results,
        batched_inputs,
        multiview_data,
        shape,
        downsample=False,
        interp="bilinear",
    ):
        bs, v, H_padded, W_padded = shape
        assert bs == 1
        if interp == "trilinear":
            target_xyz = batched_inputs["original_xyz"][None].to(self.device)
            if downsample:
                target_xyz = (
                    F.interpolate(
                        target_xyz[0].permute(0, 3, 1, 2),
                        scale_factor=0.5,
                        mode="nearest",
                    )
                    .permute(0, 2, 3, 1)
                    .reshape(
                        bs,
                        v,
                        target_xyz.shape[2] // 2,
                        target_xyz.shape[3] // 2,
                        target_xyz.shape[4],
                    )
                )
            target_p2v = torch.arange(
                target_xyz.flatten(1, 3).shape[1], device=self.device
            )[None]
            source_xyz = multiview_data["multi_scale_xyz"]
            source_p2v = multiview_data["multi_scale_p2v"]

            mask_pred_results = mask_pred_results[:, source_p2v][None]

            mask_pred_results = mask_pred_results.permute(0, 2, 1)
            source_xyz = source_xyz.flatten(0, 2)[None]
            source_p2v = source_p2v[None]
            B, _, Q = mask_pred_results.shape

            mask_pred_results = (
                interpolate_feats_3d(
                    mask_pred_results,
                    source_xyz,
                    source_p2v,
                    target_xyz,
                    target_p2v,
                    shape=[bs, v],
                    num_neighbors=self.cfg.INTERP_NEIGHBORS,
                    voxelize=True,
                )
                .reshape(
                    target_xyz.shape[1], Q, target_xyz.shape[-3], target_xyz.shape[-2]
                )
                .permute(1, 0, 2, 3)
                .to(mask_pred_results.dtype)
            )
        elif interp == "bilinear":
            Q, N, H, W = mask_pred_results.shape
            img_size = (H_padded, W_padded)
            if downsample:
                img_size = (img_size[0] // 2, img_size[1] // 2)

            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(img_size[0], img_size[1]),
                mode="bilinear",
                align_corners=False,
            )
        else:
            raise NotImplementedError(
                f"interp must be either trilinear or bilinear, got {interp}"
            )

        return mask_pred_results

    def export_pred_benchmark(self, processed_results, scene_name, dataset_name):
        # for instance segmentation
        # root_path = f"/path/to/language_grounding/benchmark_evaluations"
        if "scannet200" in dataset_name:
            learning_map_inv = SCANNET200_LEARNING_MAP_INV
            root_path = "/path/to/language_grounding/benchmark_evaluations/scannet200_80_4"
        else:
            root_path = "/path/to/language_grounding/benchmark_evaluations/scannet_52_10.5"
            learning_map_inv = LEARNING_MAP_INV

        base_path = f"{root_path}/instance_evaluation"
        pred_mask_path = f"{base_path}/pred_mask"

        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        file_name = scene_name
        with open(f"{base_path}/{file_name}.txt", "w") as fout:
            real_id = -1
            pred_classes = (
                processed_results["instances_3d"]["pred_classes"].cpu().numpy()
            )
            scores = processed_results["instances_3d"]["pred_scores"].cpu().numpy()
            pred_masks = processed_results["instances_3d"]["pred_masks"].cpu().numpy()
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                pred_class = learning_map_inv[pred_class]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                np.savetxt(
                    f"{pred_mask_path}/{file_name}_{real_id}.txt", mask, fmt="%d"
                )
                fout.write(
                    f"pred_mask/{file_name}_{real_id}.txt {pred_class} {score}\n"
                )

        # for semantic segmentation
        base_path = f"{root_path}/semantic_evaluation"
        Path(base_path).mkdir(parents=True, exist_ok=True)

        pred_mask_path = f"{base_path}/{scene_name}.txt"

        with open(pred_mask_path, "w") as fout:
            pred_mask = processed_results["semantic_3d"].cpu().numpy()
            pred_mask = np.array([learning_map_inv[x + 1] for x in pred_mask])
            np.savetxt(pred_mask_path, pred_mask, fmt="%d")

        torch.cuda.empty_cache()

    def eval_ghost(
        self,
        mask_cls_results,
        mask_pred_results,
        batched_inputs,
        scannet_gt_target_dicts,
        scannet_p2v,
        num_classes,
        scannet_idxs,
        segments,
        scannet_all_masks_batched=None,
        actual_decoder_3d=True,
    ):
        processed_results = []

        if (not self.cfg.USE_GT_MASKS or 'ref' not in batched_inputs[0]['dataset_name']) and self.cfg.USE_SEGMENTS:
            mask_pred_results = voxel_map_to_source(
                mask_pred_results.permute(0, 2, 1), segments if not self.cfg.USE_GT_MASKS else scannet_all_masks_batched
            ).permute(0, 2, 1)

        pred_masks = mask_pred_results
        for i, pred_mask in enumerate(pred_masks):
            if self.cfg.USE_GT_MASKS and 'ref' in batched_inputs[i]['dataset_name']:
                max_valid_point = len(scannet_gt_target_dicts[i]['all_scannet_masks'].unique())
                pred_mask = pred_mask[:, :max_valid_point]
            else:
                pred_mask = pred_mask[:, scannet_p2v[i]]

                # remove padding
                max_valid_point = scannet_gt_target_dicts[i]["max_valid_points"]
                pred_mask = pred_mask[:, :max_valid_point]

            if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                if 'ref' in batched_inputs[i]['dataset_name']:
                    processed_3d = {
                        "pred_scores": mask_cls_results[i],
                        "pred_masks": pred_mask,
                        "scannet_idxs": scannet_idxs[i] if len(scannet_idxs) > 0 else None,
                        "reduced_scene_id_to_original_id": scannet_gt_target_dicts[i]["reduced_scene_id_to_original_id"] if self.cfg.USE_GT_MASKS else None,
                    }
                else:
                    processed_3d = self.inference_scannet_ghost(
                        pred_mask, mask_cls_results[i], num_classes=num_classes
                    )

                if "test" not in batched_inputs[i]["dataset_name"]:
                    processed_3d["scannet_gt_masks"] = scannet_gt_target_dicts[i][
                        "masks"
                    ]
                    processed_3d["scannet_gt_classes"] = (
                        scannet_gt_target_dicts[i]["labels"] + 1
                    )
                    processed_3d["max_valid_points"] = max_valid_point
                processed_3d = {"instances_3d": processed_3d}

            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                semantic_r = retry_if_cuda_oom(self.inference_scannet_ghost_semantic)(
                    mask_cls_results[i], pred_mask
                )
                processed_3d["semantic_3d"] = semantic_r

            if self.cfg.MATTERPORT_ALL_CLASSES_TO_21:
                matterport_all_classes_to_21 = torch.tensor(
                    list(MATTERPORT_ALL_CLASSES_TO_21.values()), device=pred_mask.device
                )
                processed_3d["instances_3d"][
                    "pred_classes"
                ] = matterport_all_classes_to_21[
                    processed_3d["instances_3d"]["pred_classes"] - 1
                ]
                processed_3d["semantic_3d"] = (
                    matterport_all_classes_to_21[processed_3d["semantic_3d"]] - 1
                )

            processed_results.append(processed_3d)

            if self.cfg.EXPORT_BENCHMARK_DATA:
                self.export_pred_benchmark(
                    processed_results[-1],
                    batched_inputs[i]["file_name"].split("/")[-3],
                    dataset_name=batched_inputs[i]["dataset_name"],
                )
                return None

            if self.cfg.VISUALIZE:
                self.visualize_pred_on_scannet(
                    batched_inputs[i],
                    processed_results[i],
                    scannet_gt_target_dicts,
                    index=i,
                    scannet_idxs=scannet_idxs[i] if len(scannet_idxs) > 0 else None,
                )
        torch.cuda.empty_cache()
        return processed_results

    def eval_normal(
        self,
        mask_cls_results,
        mask_pred_results,
        batched_inputs,
        images,
        shape,
        num_classes,
        decoder_3d,
        multiview_data,
        actual_decoder_3d=False,
    ):
        bs, v, H_padded, W_padded = shape
        processed_results = []

        for i, (
            mask_cls_result,
            mask_pred_result,
            input_per_image,
            image_size,
        ) in enumerate(
            zip(mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes)
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            shape_ = [1, v, H_padded, W_padded]

            multiview_data_ = None
            if multiview_data is not None:
                multiview_data_ = {}
                multiview_data_["multi_scale_xyz"] = multiview_data["multi_scale_xyz"][
                    -1
                ][i]
                if self.cfg.INPUT.VOXELIZE:
                    multiview_data_["multi_scale_p2v"] = multiview_data[
                        "multi_scale_p2v"
                    ][-1][i]

            if self.eval_sparse:
                valids = input_per_image.get("valids")
                valids = torch.stack(valids).reshape(v, height, width)

            processed_results.append({})

            if self.cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON and decoder_3d:
                semantic_r = self.inference_video_semantic(
                    mask_cls_result,
                    mask_pred_result,
                    image_size,
                    valids if self.eval_sparse else None,
                    batched_inputs[i],
                    multiview_data_,
                    shape_,
                )
                processed_results[-1]["semantic_3d"] = semantic_r
               
            output_img_size = None

            if "coco" in input_per_image["dataset_name"] or "sam" in input_per_image['dataset_name']:
                output_img_size = [
                    input_per_image.get("height"),
                    input_per_image.get("width"),
                ]
                height, width = image_size

            if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                instance_r = self.inference_video(
                    mask_cls_result,
                    mask_pred_result,
                    height,
                    width,
                    valids if self.eval_sparse else None,
                    decoder_3d=decoder_3d,
                    num_classes=num_classes,
                    batched_inputs=batched_inputs[i],
                    multiview_data=multiview_data_,
                    shape=shape_,
                    output_img_size=output_img_size,
                    actual_decoder_3d=actual_decoder_3d,
                )

                if decoder_3d:
                    processed_results[-1]["instances_3d"] = instance_r["3d"]

                if not decoder_3d or not actual_decoder_3d:
                    processed_results[-1]["instances"] = instance_r["2d"]
        return processed_results
    
    def duplicate_ref(
        self, batched_inputs, sr3d_data, valids, segments, multiview_data
    ):
        bs = len(sr3d_data)
        batched_inputs = [copy.copy(batched_inputs[0]) for i in range(bs)]
        for i in range(bs):
            batched_inputs[i]['sr3d_data'] = [sr3d_data[i]]
        if valids is not None:
            valids = valids.repeat(bs, 1, 1, 1)
        if segments is not None:
            segments = segments.repeat(bs, 1, 1)
        multiview_data['multi_scale_xyz'] = [multiview_data['multi_scale_xyz'][i].repeat(bs, 1, 1, 1, 1) for i in range(len(multiview_data['multi_scale_xyz']))]
        multiview_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][i].repeat(bs, 1) for i in range(len(multiview_data['multi_scale_p2v']))]
        return valids, segments, multiview_data, batched_inputs

    def add_boxes_to_targets(self, targets, pcs):
        """
            boxes are in the format [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        for i, target in enumerate(targets):
            assert not self.cfg.USE_GT_MASKS
            masks = target['masks'].to(torch.bool)
            pc = pcs[i, :masks.shape[1]]
            object_bbox = torch.tensor(
                [pc[mask].min(0)[0].tolist() + pc[mask].max(0)[0].tolist() if mask.sum() > 1 else [0.0, 0.0, 0.0, 1e-2, 1e-2, 1e-2] for mask in masks], device=masks.device
            )
            targets[i]['boxes'] = object_bbox
        return targets

    def add_3d_rope_encoding(self, points, features, dim_model=None, base=10000.0):
        """
        Add 3D Rotary Position Embeddings (ROPE) to point cloud features
        
        Args:
            points: Point cloud coordinates [B, N, 3]
            features: Point features [B, N, C]
            dim_model: Feature dimension (if None, uses features.shape[-1])
            base: Base value for frequency calculation
            
        Returns:
            Enhanced features with positional information
        """
        if dim_model is None:
            dim_model = features.shape[-1]
            
        batch_size, num_points, _ = points.shape
        device = points.device
        
        dim_per_coord = dim_model // 3  # Split dimensions  x,y,z
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_per_coord, 2).float() / dim_per_coord)).to(device)
        enhanced_features = features.clone()
        
        for dim_idx in range(3): 
            pos = points[:, :, dim_idx].unsqueeze(-1) 
            freqs = torch.einsum('bn,d->bnd', pos.squeeze(-1), inv_freq)  
            emb = torch.cat((freqs, freqs), dim=-1) 
            sin = emb.sin()  
            cos = emb.cos()  
            start_idx = dim_idx * dim_per_coord
            end_idx = start_idx + dim_per_coord
            
            end_idx = min(end_idx, dim_model)
            actual_dim = end_idx - start_idx
            
            if actual_dim > 0:
                feat_part = features[:, :, start_idx:end_idx]
                if actual_dim % 2 == 0:  
                    feat_part_1 = feat_part[:, :, :actual_dim//2]
                    feat_part_2 = feat_part[:, :, actual_dim//2:]
                    
                    # Apply rotary transformation
                    rotated_feat = torch.cat([
                        feat_part_1 * cos[:, :, :actual_dim//2] - feat_part_2 * sin[:, :, :actual_dim//2],
                        feat_part_2 * cos[:, :, :actual_dim//2] + feat_part_1 * sin[:, :, :actual_dim//2]
                    ], dim=-1)
                    
                    enhanced_features[:, :, start_idx:end_idx] = rotated_feat
                else:
                    enhanced_features[:, :, start_idx:end_idx] = feat_part + sin[:, :, :actual_dim]
        
        return enhanced_features
    
    
    def random_sample_pointcloud(self, pointcloud, feature_cloud, target_points):
        """Simple random sampling - very fast but less structure-preserving"""
        batch_size, num_points, _ = pointcloud.shape
        device = pointcloud.device
        
        downsampled_pc = []
        downsampled_features = []
        
        target_points = min(target_points, num_points)
        
        for b in range(batch_size):
            # Random sampling (much faster than FPS)
            idx = torch.randperm(num_points, device=device)[:target_points]
            downsampled_pc.append(pointcloud[b, idx])
            downsampled_features.append(feature_cloud[b, idx])
        
        return torch.stack(downsampled_pc), torch.stack(downsampled_features)

    def build_knn_attention_mask(self, pointcloud, k=128, include_self=True):
        """
        Build kNN attention mask for point cloud features
        
        Args:
            pointcloud: Point cloud coordinates [B, N, 3]
            k: Number of nearest neighbors per point
            include_self: Whether to include self in the neighbors
            
        Returns:
            attention_mask: Binary attention mask [B, N, N] where 1 indicates connection
        """
        batch_size, num_points, _ = pointcloud.shape
        device = pointcloud.device
        
        # Initialize attention masks
        attention_masks = []
        
        for b in range(batch_size):
            # Compute pairwise distances between all points
            dist = torch.cdist(pointcloud[b], pointcloud[b])
            
            if not include_self:
                # Set self-distances to a large value to exclude them
                eye = torch.eye(num_points, device=device)
                dist = dist + eye * 1e5
            
            # Get indices of k nearest neighbors for each point
            _, knn_idx = torch.topk(dist, k=min(k, num_points), dim=1, largest=False)
            
            # Create binary mask where 1 indicates neighbor relationship
            mask = torch.zeros(num_points, num_points, device=device)
            for i in range(num_points):
                mask[i, knn_idx[i]] = 1.0
            
            attention_masks.append(mask)
        
        return torch.stack(attention_masks)
    
    def build_knn_attention_mask_indices(self, pointcloud, k=8, include_self=True):
        """
        Build kNN attention mask as indices for point cloud features
        Compatible with the modified Qwen2.5-VL KNN attention
        
        Args:
            pointcloud: Point cloud coordinates [B, N, 3]
            k: Number of nearest neighbors per point (default 8)
            include_self: Whether to include self in the neighbors
            
        Returns:
            knn_mask: KNN neighbor indices [B, N, k] for points only
        """
        batch_size, num_points, _ = pointcloud.shape
        device = pointcloud.device
        
        # Initialize KNN masks
        knn_masks = []
        
        for b in range(batch_size):
            # Compute pairwise distances between all points
            dist = torch.cdist(pointcloud[b], pointcloud[b])
            
            if not include_self:
                # Set self-distances to a large value to exclude them
                eye = torch.eye(num_points, device=device)
                dist = dist + eye * 1e5
            
            # Get indices of k nearest neighbors for each point
            _, knn_idx = torch.topk(dist, k=min(k, num_points), dim=1, largest=False)
            
            knn_masks.append(knn_idx)
        
        return torch.stack(knn_masks)  # [B, N, k]

    def process_point_cloud_and_text_with_qwen(self, feature_cloud, pointcloud, attention_mask_indices, target_sentence):
        """
        Process 3D point cloud features and text query through Qwen model
        
        Args:
            feature_cloud: Point features [B, N, C] - e.g. [1, 16384, 2048]
            pointcloud: Point coordinates [B, N, 3]
            attention_mask_indices: KNN neighbor indices [B, N, k] where k is number of neighbors
            target_sentence: Text query to process alongside points
            
        Returns:
            hidden_states: Last hidden states from Qwen model
        """
        batch_size, num_points, feature_dim = feature_cloud.shape
        device = feature_cloud.device
        k = attention_mask_indices.shape[-1]  # Number of neighbors
        
        # Simplified text input - just use target sentence directly
        text_inputs = self.qwen_processor.tokenizer(
            target_sentence,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=128  # Reduced from 512 since we're not using chat template
        ).to(device)
        
        qwen_embed_dim = self.qwen_model.config.hidden_size  # Should be 4096 for Qwen2.5-3B
        
        text_length = text_inputs.input_ids.shape[1]
        total_length = text_length + num_points
        
        attention_mask_indices = attention_mask_indices + text_length  # Shift indices for point cloud tokens

        # Setup position IDs
        # Generate position IDs for text tokens
        position_ids = torch.arange(0, text_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Extend position IDs for point cloud tokens
        point_position_ids = torch.full((batch_size, num_points), fill_value=text_length, dtype=torch.long, device=device)

        # Concatenate position IDs for text and point cloud tokens
        position_ids = torch.cat([position_ids, point_position_ids], dim=1).unsqueeze(0).expand(3, batch_size, -1)
        
        text_embeds = self.qwen_model.get_input_embeddings()(text_inputs.input_ids)
        point_embeddings = feature_cloud
        combined_embeds = torch.cat([text_embeds, point_embeddings], dim=1)
        # Forward pass through transformer layers
        hidden_states = combined_embeds
        
        # Get rotary embeddings
        rotary_emb = self.qwen_model.base_model.model.model.rotary_emb
        cos, sin = rotary_emb(hidden_states, position_ids=position_ids)
        cos, sin = cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        position_embeddings = (cos, sin)
        
        # Process through each transformer layer
        
        # import pdb; pdb.set_trace()
        
        for layer in self.qwen_model.base_model.model.model.layers: # Qwen2_5_VLDecoderLayer
            layer_outputs = layer(
                hidden_states,
                attention_mask_indices=attention_mask_indices,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                position_embeddings=position_embeddings,
                text_length=text_length,
                use_text = True
            ) # hope flash attn
            hidden_states = layer_outputs[0]
        

        hidden_states = self.qwen_model.base_model.model.model.norm(hidden_states)
        point_token_representations = hidden_states[:, text_length:, :]
        
        text_token_representations = hidden_states[:, :text_length, :]
        
        return {
            'point_features': point_token_representations,
            'text_features': text_token_representations,
            'combined_features': hidden_states
        }

    def upsample_point_features(self, source_pc, source_features, target_pc, k_neighbors=3):
        """
        Upsample point cloud features from a sparse point cloud to a denser target point cloud.
        
        Args:
            source_pc: Source sparse point cloud [B, N_source, 3]
            source_features: Source point features [B, N_source, C]
            target_pc: Target dense point cloud [B, N_target, 3]
            k_neighbors: Number of nearest neighbors for interpolation
            
        Returns:
            Upsampled features [B, N_target, C]
        """
        batch_size = source_pc.shape[0]
        device = source_pc.device
        
        # Create source and target point IDs for the interpolation function
        source_p2v = torch.arange(source_pc.shape[1], device=device).unsqueeze(0).repeat(batch_size, 1)
        target_p2v = torch.arange(target_pc.shape[1], device=device).unsqueeze(0).repeat(batch_size, 1)
        
        # Prepare for interpolation - convert to [B, C, N] format
        interp_features = source_features.float()
        source_pc = source_pc.float()
        target_pc = target_pc.float()
        
        # The shape parameter should be [batch_size, num_views]
        # In our case with flat point clouds, we use [batch_size, 1]
        shape = [batch_size, 1]
        
        # Use the 3D interpolation function
        upsampled = interpolate_feats_3d(
            source_feats=interp_features,   # [B, N_source, C]
            source_xyz=source_pc,    # [B, N_source, 3]
            source_p2v=source_p2v, # [B, N_source]
            target_xyz=target_pc,  # [B, N_target, 3]
            target_p2v=target_p2v, # [B, N_target]
            shape=shape, # [B, N_source]
            num_neighbors=k_neighbors,
            voxelize=True,
        )
        
        return upsampled.permute(0, 2, 1)

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
                            Each tensor has shape (4, 4).
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
                        relevant_ids (list[int]): List of relevant IDs.
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
        
        images = []
        for video in batched_inputs:
            for image in video["images"]:
                images.append(image.to(self.device))
        bs = len(batched_inputs)
        v = len(batched_inputs[0]["images"])
        
        if not self.training:
            # print(f"using {v} views for eval!")
            # print(self.cfg.USE_GT_MASKS)
            evaling = 1
            print(f"using {v} views for eval")
        
        
        ##########################################################################################
        ####  start the qwen3D design
        ####  replace the DINOv2 Encoder and MsDeformable Attn
        ##########################################################################################
        assert bs == 1, "Batch size should be 1 for our model, or the memory will explode"
        
        # print("starting to process point cloud")
        import time
        time_start = time.time()
        pointcloud, featurecloud = self.encode_scene_3d(batched_inputs)
        # torch.cuda.empty_cache()  # Clear cache after heavy 3D processing
        # print("pc.shape", pointcloud.shape) # [1, 12961, 3]
        # print("contextualized_point_features.shape", featurecloud.shape) # [1, 12961, 2048]
        # import pdb; pdb.set_trace()
        t1= time.time()
        # print("encoding time", t1 - time_start)
        
        # pointcloud = process_point_features(pointcloud.unsqueeze(0))
        # featurecloud = process_point_features(featurecloud.unsqueeze(0)) # avoid nan
        pointcloud = pointcloud.unsqueeze(0)  # Add batch dimension
        featurecloud = featurecloud.unsqueeze(0)
        
        # save point cloud
        
        target_sentence = batched_inputs[0]["sr3d_data"][0]["text_caption"]
        
        # save_pointcloud_to_ply(pointcloud, filepath=f"viz_global/scene_pointcloud_{target_name}.ply")
        if torch.isnan(featurecloud).any():
            print("NaN values found in feature cloud BEFORE positional encoding")
            # Handle NaN values as needed (e.g., skip, replace, etc.)
            import pdb; pdb.set_trace()
        
        featurecloud = self.add_3d_rope_encoding(pointcloud, featurecloud)
        
        if torch.isnan(featurecloud).any():
            print("NaN values found in feature cloud AFTER positional encoding")
            # Handle NaN values as needed (e.g., skip, replace, etc.)
            import pdb; pdb.set_trace()
        
        t2 = time.time()
        # print("add_3d_rope_encoding time", t2 - t1)
        # import pdb; pdb.set_trace()
        """no more downsampling for eval"""
        if self.training:
            pointcloud, featurecloud = self.random_sample_pointcloud(
                pointcloud, featurecloud, target_points=4096 # 16384
            )
            print(f"Train: Downsampled point cloud to {pointcloud.shape[1]} points")
        else:
            print(f"Eval: Using full point cloud with {pointcloud.shape[1]} points") # 59 views: 13904
        
        t3 = time.time()
        # print("random_sample_pointcloud time", t3 - t2)
        
        # save_pc = pointcloud.squeeze(0).to(torch.float16).cpu().numpy()
        # # save pointcloud to ply
        # save_pointcloud_to_ply(save_pc, filepath=f"viz/scene_pointcloud_{target_sentence}_train.ply")
        # scannet_coords = batched_inputs[0]["scannet_coords"]
        # scannet_color = batched_inputs[0]["scannet_color"] 
        # scannet_pc = np.concatenate([scannet_coords, scannet_color], axis=-1)
        # save_pointcloud_to_ply(scannet_pc, filepath=f"viz/scene_pointcloud_{target_sentence}_scannet_train.ply")
        # # import pdb; pdb.set_trace()
        
        # attention_mask_indices = self.build_knn_attention_mask_indices(pointcloud, k=8)
        # torch.cuda.empty_cache()  # Clear cache after heavy 3D processing
        # print(f"Point cloud shape: {pointcloud.shape}")
        # print(f"Feature cloud shape: {featurecloud.shape}")
        # print(f"Attention mask shape: {attention_mask.shape}")
        
        t4= time.time()
        # print("build_knn_attention_mask time", t4 - t3)
        
        time_end = time.time()
        # print(f"Time taken for point cloud processing: {time_end - time_start:.2f} seconds")
        
        # use Qwen as a backbone 
        """Inputs: feature cloud (16384*2048), target sentence, attention mask
        Outputs: mid-layers, first try last hidden state
        """
        # NaN is in the feature cloud, so we need to handle it
        
        if torch.isnan(featurecloud).any():
            print("NaN values found in feature cloud")
            # Handle NaN values as needed (e.g., skip, replace, etc.)
            import pdb; pdb.set_trace()
        
        if torch.isnan(pointcloud).any():
            print("NaN values found in point cloud")
            # Handle NaN values as needed (e.g., skip, replace, etc.)
            pointcloud = torch.nan_to_num(pointcloud)
            if torch.isnan(pointcloud).any():
                print("NaN values still found in point cloud after replacement")
                pointcloud = process_point_features(pointcloud)
                if torch.isnan(pointcloud).any():
                    print("Well, well")
                    import pdb; pdb.set_trace()
        
        

        time_start = time.time()
        # with torch.no_grad():
        BYPASS_DECODER = False # decide whether to use Qwen decoder or not
        if not BYPASS_DECODER:
        #     qwen_outputs = self.process_point_cloud_and_text_with_qwen(
        #         featurecloud, 
        #         pointcloud, 
        #         attention_mask, 
        #         target_sentence
        #     )
            chunk_size = 8192 # on H100
            
            # print(f"Processing {pointcloud.shape[1]} points in chunks of {chunk_size}")
            # Random permutation for data augmentation
            
            pointcloud, featurecloud = self.process_pointcloud_in_chunks(
                pointcloud, featurecloud, target_sentence, chunk_size=chunk_size
            )
            contextualized_point_features = featurecloud
        # # import pdb; pdb.set_trace()
        # # Getting the contextualized point features for downstream tasks
        #     contextualized_point_features = qwen_outputs['point_features']
        
        else:
            contextualized_point_features = featurecloud
            # contextualized_point_features = self.upsample_point_features(
            #     pointcloud, featurecloud, pointcloud, k_neighbors=3
            # )
        
        # print(f"Contextualized point features shape: {contextualized_point_features.shape}")
        
        
        if torch.isnan(contextualized_point_features).any():
            print("NaN values found in contextualized point features")
            # Handle NaN values as needed (e.g., skip, replace, etc.)
            import pdb; pdb.set_trace()
        
        

        time_end = time.time()
        # print(f"Time taken for Qwen processing: {time_end - time_start:.2f} seconds")
        
        ##########################################################################################
        ####  change for encoder end
        ##########################################################################################
        
        
        # print(f"Contextualized point features shape: {contextualized_point_features.shape}")
        # import pdb; pdb.set_trace()
        
        # important to check this when joint joint training
        decoder_3d = torch.tensor(
            sum([batched_input["decoder_3d"] for batched_input in batched_inputs]),
            device=self.device,
        )

        if self.cfg.MULTI_TASK_TRAINING and self.training:
            eff_bs = len(batched_inputs)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(decoder_3d)
                eff_bs *= get_world_size()
            decoder_3d = decoder_3d.item()
            assert (
                decoder_3d == eff_bs or decoder_3d == 0
            ), f"All videos must have the same decoder_3d value {decoder_3d}"

        decoder_3d = decoder_3d > 0
        actual_decoder_3d = batched_inputs[0]["actual_decoder_3d"]
        
        if self.training and self.cfg.PSEUDO_2D_AUG and decoder_3d and not actual_decoder_3d:
            if random.random() > 0.5:
                decoder_3d = False

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        H_padded, W_padded = images.tensor.shape[-2:]

        valids, segments = None, None

        # load sensor data
        valids, multiview_data = self.load_3d_data(
            batched_inputs,
            images_shape=[bs, v, H_padded, W_padded],
        )

        # Trick to eval on multiple language utterances in referential grounding with same visual scene
        # We only encode the visual scene once, and then duplicate it for each language utterance
        do_duplicate = False
        if not self.training and (sr3d_data := batched_inputs[0].get('sr3d_data', None)) and isinstance(sr3d_data, list) and len(sr3d_data) > 1:
            assert bs == 1
            do_duplicate = True
            valids, segments, multiview_data, batched_inputs = self.duplicate_ref(
                batched_inputs, sr3d_data, valids, segments, multiview_data
            )
            bs = len(batched_inputs)
        
        # load mesh data
        (
            scannet_pc,
            scannet_gt_target_dicts,
            scannet_p2v,
            scannet_segments_batched,
            scannet_all_masks_batched,
        ) = (None, None, None, None, None)

        if self.cfg.USE_GHOST_POINTS:
            full_scene_dataset = batched_inputs[0]["full_scene_dataset"]
            (
                scannet_pc,
                scannet_p2v,
                scannet_gt_target_dicts,
                scannet_idxs,
                scannet_segments_batched,
                scannet_all_masks_batched,
            ) = load_scannet_data(
                self.cfg,
                batched_inputs,
                multiview_data,
                do_knn=self.training or not full_scene_dataset or self.cfg.FORCE_SUBSAMPLE,
                images=images,
                shape=[bs, v, H_padded, W_padded],
                device=self.device,
                is_training=self.training,
                tokenizer=self.tokenizer,
            )

        
        if do_duplicate and not self.training:
            new_bs = 1
            shape = [bs, v, H_padded, W_padded]
            orig_bs = multiview_data["multi_scale_xyz"][0].shape[0]
            assert orig_bs == bs
            multiview_data = slice_tensor(multiview_data, shape[1], orig_bs)
            scannet_pc = slice_tensor(scannet_pc, shape[1], orig_bs)
            scannet_p2v = slice_tensor(scannet_p2v, shape[1], orig_bs)
            shape = (new_bs, *shape[1:])
            bs = new_bs
        
        ### Set of lines ###
        # call to the model with some inputs
        if not self.use_qwen_3d: 
            mask_features, multi_scale_features = self.visual_backbone(
                images=images.tensor,
                multi_scale_xyz=multiview_data["multi_scale_xyz"] if self.cfg.MODEL.CROSS_VIEW_BACKBONE and decoder_3d else None,
                multi_scale_p2v=multiview_data["multi_scale_p2v"] if self.cfg.MODEL.CROSS_VIEW_BACKBONE and decoder_3d else None,
                shape=[bs, v, H_padded, W_padded],
                decoder_3d=decoder_3d,
                actual_decoder_3d=actual_decoder_3d,
                mesh_pc=scannet_pc,
                mesh_p2v=scannet_p2v
            )
            # mask_features: torch.Size([1, 256, 118475, 1])
            # multi_scale_features: 3 layers of: torch.Size([54, 256, 35, 46]), 54 is image number

            if do_duplicate and not self.training:
                mask_features = expand_tensor(mask_features, shape[1], orig_bs)
                multi_scale_features = expand_tensor(multi_scale_features, shape[1], orig_bs)
                scannet_pc = expand_tensor(scannet_pc, shape[1], orig_bs)
                scannet_p2v = expand_tensor(scannet_p2v, shape[1], orig_bs)
                shape = [orig_bs, *shape[1:]]
                bs = orig_bs

        # mask classification target
        if self.cfg.USE_GHOST_POINTS:
            targets = scannet_gt_target_dicts
        else:
            targets = prepare_targets(
                self.cfg,
                batched_inputs,
                images,
                valids,
                dataset_names=[batched_input["dataset_name"] for batched_input in batched_inputs],
                is_training=self.training,
                tokenizer=self.tokenizer,
                device=self.device,
            )

        if self.cfg.USE_BOX_LOSS and actual_decoder_3d and self.training and "test" not in batched_inputs[0]["dataset_name"]:
            targets = self.add_boxes_to_targets(targets, scannet_pc)

        captions = None
        if self.cfg.MODEL.OPEN_VOCAB:
            captions = [targets[i]["text_caption"] for i in range(len(targets))]
            num_classes = (
                max([batched_input["num_classes"] for batched_input in batched_inputs])
                + 1
            )

        if self.training and self.cfg.GENERATION:
            answers = [random.choice(sample['sr3d_data'][0]['answers']) if 'sr3d_data' in sample and 'answers' in sample['sr3d_data'][0] else '' for sample in batched_inputs]
            tokenized_answers = [self.answer_tokenizer(answer)['input_ids'] for answer in answers]
            answer_len = [len(answer) for answer in tokenized_answers]
            max_answer_len = max(answer_len)
            padded_answers = []
            for i in range(len(tokenized_answers)):
                padded_answers.append(tokenized_answers[i] + [0] * (max_answer_len - len(tokenized_answers[i])))
            padded_answers = torch.tensor(padded_answers).to('cuda')

        if self.cfg.USE_GHOST_POINTS and decoder_3d:
            scannet_pc_ = scatter_mean(scannet_pc, scannet_p2v, dim=1)
            scannet_p2v_ = (
                torch.arange(scannet_pc.shape[1], device=scannet_pc.device)
                .unsqueeze(0)
                .repeat(scannet_pc.shape[0], 1)
            )


        
        
        # contextualized_point_features # torch.Size([1, 12961, 2048])
        # pointcloud (Pdb) pointcloud.shape torch.Size([1, 12961, 3])
        scannet_pc_xyz = scannet_pc_.clone()
        # print("my pc xmax", pointcloud[..., 0].max())
        # print("my pc xmin", pointcloud[..., 0].min())
        # print("my pc ymax", pointcloud[..., 1].max())
        # print("my pc ymin", pointcloud[..., 1].min())
        # print("my pc zmax", pointcloud[..., 2].max())
        # print("my pc zmin", pointcloud[..., 2].min())
        # print("scannet pc xmax", scannet_pc_xyz[..., 0].max())
        # print("scannet pc xmin", scannet_pc_xyz[..., 0].min())
        # print("scannet pc ymax", scannet_pc_xyz[..., 1].max())
        # print("scannet pc ymin", scannet_pc_xyz[..., 1].min())
        # print("scannet pc zmax", scannet_pc_xyz[..., 2].max())
        # print("scannet pc zmin", scannet_pc_xyz[..., 2].min())
        
        # my pc xmax tensor(5.6750, device='cuda:0')
        # my pc xmin tensor(-0.0750, device='cuda:0')
        # my pc ymax tensor(6.3750, device='cuda:0')
        # my pc ymin tensor(0.0250, device='cuda:0')
        # my pc zmax tensor(3.2750, device='cuda:0')
        # my pc zmin tensor(-0.0250, device='cuda:0')
        # scannet pc xmax tensor(5.6224, device='cuda:0')
        # scannet pc xmin tensor(-0.0392, device='cuda:0')
        # scannet pc ymax tensor(6.2310, device='cuda:0')
        # scannet pc ymin tensor(0.0270, device='cuda:0')
        # scannet pc zmax tensor(3.2792, device='cuda:0')
        # scannet pc zmin tensor(-0.0053, device='cuda:0')
        time_start = time.time()

        ##########################################################################################
        ####  start the Upsampling
        ####  Using 3D interpolation to upsample the features, k=12
        ##########################################################################################
        
        upsampled_features = self.upsample_point_features(
            source_pc=pointcloud,
            source_features=contextualized_point_features,
            target_pc=scannet_pc_xyz,
            k_neighbors=8
        )
        torch.cuda.empty_cache()  # Clear cache after heavy 3D processing
        # should save to local
        # save_pointcloud_to_ply(scannet_pc_xyz.squeeze(0).to(torch.float16).detach().cpu().numpy(), filepath=f"viz/upsampled_points_{target_sentence}_val.ply")
        # save_pointcloud_to_ply(pointcloud.squeeze(0).to(torch.float16).detach().cpu().numpy(), filepath=f"viz/the_4096_points_{target_sentence}_val.ply")
        # import pdb; pdb.set_trace()
        
        # project the upsampled features to 256 dimensions
        qwen_mask_features = self.connector(upsampled_features).permute(0, 2, 1).unsqueeze(-1) # [1, 256, 118475, 1]
        
        time_end = time.time()
        # print("upsample time", time_end - time_start)
        
        # what I need to do is to get the features from the Qwen model and then use them as the input to the mask decoder
        # Qwen points are not as dense as the original points, so I kind of need to upsample them to orignal points
        
        # however, qwen features is 2048, this feature is only 256
        # what is a good network architecture to upsample the features?
        
        # use 3d interpolation to upsample the features pointcloud -> scannet_pc_xyz
        # use a simple MLP to downgrade the dim 2048 -> 256
        # Ayush: use function interpolate_feats_3d
        
        time_start = time.time()
        
        ##########################################################################################
        ####  upsampling ends
        ####  The rest is exactly the same as the original code
        ##########################################################################################
        outputs = self.mask_decoder(
            qwen_mask_features, # torch.Size([1, 256, 118475, 1])
            shape=[bs, v],
            mask_features_xyz=scannet_pc_, # torch.Size([1, 118475, 3])
            mask_features_p2v=scannet_p2v_, # torch.Size([1, 167558])
            segments=scannet_segments_batched if self.cfg.USE_GHOST_POINTS else segments,
            decoder_3d=decoder_3d,
            captions=captions,
            actual_decoder_3d=actual_decoder_3d,
            scannet_all_masks_batched=scannet_all_masks_batched,
            max_valid_points=[targets[i]['max_valid_points'] for i in range(len(targets))] if self.cfg.USE_GHOST_POINTS and decoder_3d else None,
            tokenized_answer=padded_answers if (self.cfg.GENERATION and self.training) else None,
            answers=answers if (self.cfg.GENERATION and self.training) else None,
        )
        check_for_nans_in_dict(outputs)
        
        time_end = time.time()
        # print("mask decoder time", time_end - time_start)

        if outputs is None:
            return None

        if self.training:
            if self.cfg.GENERATION:
                outputs['tokenized_answer'] = padded_answers
                for i in range(len(outputs['aux_outputs'])):
                    outputs['aux_outputs'][i]['tokenized_answer'] = padded_answers

            # bipartite matching-based loss
            losses = self.criterion(
                outputs, targets, decoder_3d=decoder_3d, actual_decoder_3d=actual_decoder_3d
            )

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            
            if (
                self.cfg.MULTI_TASK_TRAINING
                and not self.cfg.IID_MULTITASK_TRAINING
                and not (self.cfg.GRAD_ACCUMULATION_STEPS > 1)
            ):
                if actual_decoder_3d:
                    losses["loss_3d"] = sum(losses.values())
                else:
                    losses["loss_2d"] = sum(losses.values())
                    
            return losses
        else: # not training
            generation_language = None
            if 'scanqa' in batched_inputs[0]['dataset_name'] or 'sqa3d' in batched_inputs[0]['dataset_name']:
                generation_language = outputs["generation_language"]
        
            if 'sr3d_data' in batched_inputs[0]:
                num_classes = max([len(batched_input['sr3d_data'][0]['anchor_ids']) + 1 for batched_input in batched_inputs])
            elif 'refcoco' in batched_inputs[0]['dataset_name']:
                num_classes = 2
            else:
                num_classes = max(
                    [batched_input["num_classes"] for batched_input in batched_inputs]
                )

            if self.cfg.MODEL.OPEN_VOCAB:
                outputs["pred_logits"] = outputs["pred_logits"].sigmoid()
                reduce = "mean"
                if 'sr3d_data' in batched_inputs[0]  or 'refcoco' in batched_inputs[0]['dataset_name']:
                    outputs["pred_logits"] = torch.cat(
                        [
                            convert_grounding_to_od_logits_ref(
                                logits=outputs["pred_logits"][i][None],
                                num_class=num_classes + 1,
                                positive_maps=targets[i]["positive_map"],
                                reduce=reduce,
                            )
                            for i in range(bs)
                        ]
                    )
                else:
                    outputs["pred_logits"] = torch.cat(
                        [
                            convert_grounding_to_od_logits(
                                logits=outputs["pred_logits"][i][None],
                                num_class=num_classes + 1,
                                positive_map_od=targets[i]["positive_map_od"],
                                reduce=reduce,
                            )
                            for i in range(bs)
                        ]
                    )

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            del outputs

            # import pdb; pdb.set_trace()
            # Processing for ghost points
            if self.cfg.USE_GHOST_POINTS and actual_decoder_3d:
                processed_results = self.eval_ghost(
                    mask_cls_results,
                    mask_pred_results,
                    batched_inputs,
                    scannet_gt_target_dicts,
                    scannet_p2v,
                    num_classes,
                    scannet_idxs,
                    scannet_segments_batched,
                    scannet_all_masks_batched,
                )
                if generation_language is not None:
                    return generation_language, processed_results
                return processed_results

            # Normal Processing
            processed_results = self.eval_normal(
                mask_cls_results,
                mask_pred_results,
                batched_inputs,
                images,
                [bs, v, H_padded, W_padded],
                num_classes,
                decoder_3d,
                multiview_data,
                actual_decoder_3d=batched_inputs[0]["actual_decoder_3d"],
            )

            if generation_language is not None:
                return generation_language, processed_results
            return processed_results
        
    


    def visualize_pred_on_ours(
        self,
        index,
        images,
        shape,
        input_per_image,
        processed_results,
        targets,
        valids,
        fps_xyz=None,
    ):
        bs, v, H_padded, W_padded = shape
        our_pc = input_per_image["original_xyz"]
        if self.cfg.HIGH_RES_INPUT:
            our_pc = F.interpolate(
                our_pc.float().permute(0, 3, 1, 2), scale_factor=0.5, mode="nearest"
            ).permute(0, 2, 3, 1)
            our_pc = our_pc.cpu().numpy()

            if valids is not None:
                valids = (
                    F.interpolate(
                        valids.float().permute(0, 1, 2).unsqueeze(0),
                        scale_factor=0.5,
                        mode="nearest",
                    )
                    .squeeze(0)
                    .bool()
                )

        if valids is not None:
            our_pc = our_pc[valids]

        else:
            our_pc = our_pc.reshape(-1, 3)

        vis_images = images.tensor * self.pixel_std + self.pixel_mean
        vis_images = vis_images.view(bs, v, 3, H_padded, W_padded)[index]
        if self.cfg.HIGH_RES_INPUT:
            vis_images = F.interpolate(vis_images, scale_factor=0.5, mode="bilinear")

        if valids is not None:
            color = vis_images.permute(0, 2, 3, 1)[valids].cpu().numpy() / 255.0
        else:
            color = vis_images.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy() / 255.0
        color = np.clip(color, 0, 1)

        scene_name = input_per_image["file_name"].split("/")[-3]

        pred_scores = processed_results["instances_3d"]["pred_scores"]
        pred_masks = processed_results["instances_3d"]["pred_masks"]
        pred_labels = processed_results["instances_3d"]["pred_classes"]

        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx].cpu().numpy()
        pred_labels = pred_labels[sort_idx].cpu().numpy()

        # select confident predictions
        pred_scores = pred_scores[sort_idx].cpu().numpy()

        conf = pred_scores > 0.0
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]
        if fps_xyz is not None:
            fps_xyz = fps_xyz[0][sort_idx][conf].cpu().numpy()

        gt_masks = targets[index]["masks"]
        if self.cfg.HIGH_RES_INPUT:
            gt_masks = (
                F.interpolate(gt_masks.float(), scale_factor=0.5, mode="nearest") > 0.5
            )
        else:
            gt_masks = gt_masks.reshape(-1, H_padded // 4, W_padded // 4)
            gt_masks = F.interpolate(
                gt_masks.float().unsqueeze(0), scale_factor=4.0, mode="nearest"
            )[0].bool()

        if valids is not None:
            gt_masks = gt_masks[:, valids].cpu().numpy()
        else:
            gt_masks = gt_masks.flatten(1)
            gt_masks = gt_masks.cpu().numpy()
        gt_labels = targets[index]["labels"].cpu().numpy()

        valids = np.zeros_like(our_pc[:, 0]).astype(bool)

        # valid_idx = np.random.choice(
        #     np.arange(valids.shape[0]), 200000)
        # valids[valid_idx] = True
        valids[:] = True

        dataset_name = input_per_image["dataset_name"]
        vis_utils.plot_3d_offline(
            our_pc.numpy(),
            color,
            masks=pred_masks,
            valids=valids,
            labels=pred_labels,
            gt_masks=gt_masks,
            gt_labels=gt_labels,
            scene_name=scene_name,
            data_dir="vis_sam_3d",
            mask_classes=None,
            dataset_name=dataset_name,
            fps_xyz=fps_xyz,
        )

    def visualize_pred_on_scannet(
        self, input_per_image, processed_result,
        gt_targets, index, scannet_idxs=None,
        fps_xyz=None
    ):
        pc = input_per_image['scannet_coords'].cpu().numpy()
        if scannet_idxs is not None:
            pc = pc[scannet_idxs.cpu().numpy()]

        color = (input_per_image['scannet_color'] / 255.0).cpu().numpy()
        color = np.clip(color, 0, 1)
        if scannet_idxs is not None:
            color = color[scannet_idxs.cpu().numpy()]

        scene_name = input_per_image['file_name'].split('/')[-3]
        pred_scores = processed_result["instances_3d"]['pred_scores']
        pred_masks = processed_result["instances_3d"]['pred_masks']
        pred_labels = processed_result["instances_3d"]['pred_classes']

        # sort by scores in ascending order
        sort_idx = torch.argsort(pred_scores)
        pred_masks = pred_masks.permute(1, 0)[sort_idx].cpu().numpy()
        pred_labels = pred_labels[sort_idx].cpu().numpy()

        # threshold by scores > 0.5
        pred_scores = pred_scores[sort_idx].cpu().numpy()
        conf = pred_scores > 0.05
        pred_masks = pred_masks[conf]
        pred_labels = pred_labels[conf]

        # whiteboard = pred_labels == 44
        # pred_masks = pred_masks[whiteboard]
        # pred_labels = pred_labels[whiteboard]

        if fps_xyz is not None:
            fps_xyz = fps_xyz[0][sort_idx][conf].cpu().numpy()

        gt_masks = gt_targets[index]['masks'].cpu().numpy()
        if "max_valid_points" in gt_targets[index]:
            max_valid_point = gt_targets[index]["max_valid_points"]
            gt_masks = gt_masks[:, :max_valid_point]

        gt_labels = gt_targets[index]['labels'].cpu().numpy()

        valids = np.ones_like(pc[:, 0]).astype(bool)

        dataset_name = input_per_image['dataset_name']

        vis_utils.plot_3d_offline(
            pc, color, masks=pred_masks, valids=valids,
            labels=pred_labels,
            gt_masks=gt_masks, gt_labels=gt_labels, scene_name=scene_name,
            data_dir=self.cfg.VISUALIZE_LOG_DIR,
            mask_classes=self.cfg.SKIP_CLASSES, dataset_name=dataset_name,
            fps_xyz=fps_xyz
        )


    def prepare_2d(
        self,
        pred_masks,
        img_size,
        labels_per_image,
        scores_per_image,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
        decoder_3d=False,
        output_img_size=None,
    ):
        pred_masks = self.upsample_pred_masks(
            pred_masks,
            batched_inputs,
            multiview_data,
            shape,
            downsample=False,
            interp="trilinear" if decoder_3d else "bilinear",
        )

        context_img_id = pred_masks.shape[1] // 2
        pred_masks = pred_masks[:, context_img_id]

        pred_masks = pred_masks[:, : img_size[0], : img_size[1]]

        if output_img_size is not None:
            pred_masks = F.interpolate(
                pred_masks[None], size=output_img_size, mode="bilinear"
            )[0]

        masks = pred_masks > 0.0
        image_size = masks.shape[-2:]

        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        result_2d = Instances(image_size)
        result_2d.pred_masks = masks
        result_2d.pred_boxes = Boxes(torch.zeros(masks.size(0), 4))
        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * result_2d.pred_masks.flatten(1)
        ).sum(1) / (result_2d.pred_masks.flatten(1).sum(1) + 1e-6)
        result_2d.scores = scores_per_image * mask_scores_per_image
        result_2d.pred_classes = labels_per_image
        return result_2d
    
    def prepare_3d(
        self,
        pred_masks,
        output_height,
        output_width,
        labels_per_image,
        scores_per_image,
        valids=None,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
    ):
        pred_masks = self.upsample_pred_masks(
            pred_masks,
            batched_inputs,
            multiview_data,
            shape,
            downsample=self.cfg.HIGH_RES_INPUT,
            interp="trilinear",
        )

        if valids is not None:
            # downsample valids
            if self.size_divisibility > 1:
                h, w = output_height, output_width
                pad_h = int(
                    np.ceil(h / self.size_divisibility) * self.size_divisibility - h
                )
                pad_w = int(
                    np.ceil(w / self.size_divisibility) * self.size_divisibility - w
                )
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H, W = pred_masks.shape[-2:]
            valids = (
                F.interpolate(valids.float().unsqueeze(0), size=(H, W), mode="nearest")
                .squeeze(0)
                .bool()
            )

        if valids is not None:
            pred_masks = pred_masks[:, valids]

        masks = pred_masks > 0.0
        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        # add +1 to labels as mask3d evals from 1-18
        result_3d = {
            "pred_classes": labels_per_image + 1,
            "pred_masks": masks.flatten(1).permute(1, 0),
            "pred_scores": scores_per_image * mask_scores_per_image,
        }
        return result_3d

    def inference_video(
        self,
        pred_cls,
        pred_masks,
        output_height,
        output_width,
        valids=None,
        decoder_3d=False,
        num_classes=None,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
        output_img_size=None,
        actual_decoder_3d=False,
    ):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        test_topk_per_image = pred_masks.shape[0] 
        
        if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.NON_PARAM_SOFTMAX:
            scores = pred_cls[:, :-1]
        else:
            if self.cfg.NON_PARAM_SOFTMAX:
                scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            elif self.cfg.OPEN_VOCAB_SIGMOID:
                scores = pred_cls[..., :-1].sigmoid()

        skip_classes = self.cfg.SKIP_CLASSES if decoder_3d else self.cfg.SKIP_CLASSES_2D

        if skip_classes is not None:
            skip_classes = torch.tensor(skip_classes, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(num_classes, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]
            num_classes -= len(skip_classes)

        
        num_queries = (
            self.num_queries * 2
            if self.cfg.SEPERATE_2D_3D_QUERIES
            else self.num_queries
        )

        labels = (
            torch.arange(num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]

        results = {}
        if not decoder_3d or not actual_decoder_3d:
            results["2d"] = self.prepare_2d(
                pred_masks,
                (output_height, output_width),
                labels_per_image,
                scores_per_image,
                batched_inputs,
                multiview_data,
                shape,
                decoder_3d,
                output_img_size,
            )

        if decoder_3d:
            results["3d"] = self.prepare_3d(
                pred_masks,
                output_height,
                output_width,
                labels_per_image,
                scores_per_image,
                valids,
                batched_inputs,
                multiview_data,
                shape,
            )
        return results

    def inference_video_semantic(
        self,
        mask_cls,
        mask_pred,
        image_size=None,
        valids=None,
        batched_inputs=None,
        multiview_data=None,
        shape=None,
    ):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        mask_pred = self.upsample_pred_masks(
            mask_pred,
            batched_inputs,
            multiview_data,
            shape,
            downsample=self.cfg.HIGH_RES_INPUT,
            interp="trilinear",
        )

        if self.cfg.MODEL.OPEN_VOCAB and not self.cfg.NON_PARAM_SOFTMAX:
            mask_cls = mask_cls[..., :-1]
        else:
            if self.cfg.NON_PARAM_SOFTMAX:
                mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            elif self.cfg.OPEN_VOCAB_SIGMOID:
                mask_cls = mask_cls[..., :-1].sigmoid()

        mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qvhw->cvhw", mask_cls, mask_pred).max(0)[1]
        if valids is not None:
            if self.size_divisibility > 1:
                h, w = image_size[0], image_size[1]
                pad_h = int(
                    np.ceil(h / self.size_divisibility) * self.size_divisibility - h
                )
                pad_w = int(
                    np.ceil(w / self.size_divisibility) * self.size_divisibility - w
                )
                valids = F.pad(valids, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H, W = mask_pred.shape[-2:]
            valids = (
                F.interpolate(valids.float().unsqueeze(0), size=(H, W), mode="nearest")
                .squeeze(0)
                .bool()
            )
            semseg = semseg[valids]
        return semseg.reshape(-1)

    def inference_scannet_ghost(self, pred_masks, pred_cls, num_classes):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        test_topk_per_image = pred_masks.shape[
            0
        ]  # 100 #self.cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        
        if self.cfg.MODEL.OPEN_VOCAB:
            scores = pred_cls[:, :-1]
        else:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]

        # num_classes = self.sem_seg_head.num_classes
        if num_classes == 20 and self.cfg.SKIP_CLASSES is not None:
            # because we skip floor and wall for evaluation

            # -1 to 0 index it
            skip_classes = torch.tensor(self.cfg.SKIP_CLASSES, device=self.device) - 1

            # +1 for background class
            keep_class_mask = torch.ones(num_classes, device=self.device)
            keep_class_mask[skip_classes] = 0
            scores = scores[:, keep_class_mask.bool()]
            num_classes = 18

        num_queries = pred_masks.shape[0]
        labels = (
            torch.arange(num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            test_topk_per_image, sorted=False
        )

        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        pred_masks = pred_masks[topk_indices]

        masks = pred_masks > 0.0
        mask_scores_per_image = (
            pred_masks.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        result_3d = {
            "pred_classes": labels_per_image + 1,
            "pred_masks": masks.flatten(1).permute(1, 0),
            "pred_scores": scores_per_image * mask_scores_per_image,
        }
        return result_3d

    def inference_scannet_ghost_semantic(self, mask_cls, mask_pred):
        """
        pred_cls: 100 X 19
        pred_masks: 100 X 5 X 480 X 640
        """
        if self.cfg.MODEL.OPEN_VOCAB:
            mask_cls = mask_cls[..., :-1]
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[:, :-1]

        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qn->cn", mask_cls, mask_pred).max(0)[1]
        return semseg.reshape(-1)


def save_pointcloud_to_ply(pointcloud, filepath="pointcloud.ply"):
    """
    Save a point cloud to PLY format
    
    Args:
        pointcloud: Tensor of shape [N, 3] containing 3D points
        filepath: Output filepath for the PLY file
    """
    import os
    import numpy as np
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert to numpy if it's a tensor
    if isinstance(pointcloud, torch.Tensor):
        points = pointcloud.detach().cpu().numpy()
    else:
        points = np.array(pointcloud)
    
    # Write PLY file header
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        # Write point coordinates
        for i in range(len(points)):
            f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]}\n")
    
    print(f"Saved point cloud with {len(points)} points to {filepath}")


def check_for_nans_in_dict(outputs_dict, prefix=""):
    """Check if any tensor in the outputs dictionary has NaN values."""
    import pdb
    
    found_nan = False
    
    if isinstance(outputs_dict, dict):
        for k, v in outputs_dict.items():
            current_key = f"{prefix}.{k}" if prefix else k
            
            # Recursively check nested dictionaries
            if isinstance(v, dict):
                nested_nan = check_for_nans(v, current_key)
                found_nan = found_nan or nested_nan
            
            # Check tensors for NaN values
            elif isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    print(f"NaN found in tensor: {current_key}, shape: {v.shape}")
                    found_nan = True
            
            # Check lists of tensors
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, torch.Tensor) and torch.isnan(item).any():
                        print(f"NaN found in tensor list: {current_key}[{i}], shape: {item.shape}")
                        found_nan = True
    
    # Enter debugger if NaNs were found
    if found_nan and prefix == "":  # Only trigger at the top level
        print("NaN values detected. Entering debugger.")
        pdb.set_trace()
    
    return found_nan


def process_point_features(point_features):
    # Handle underflow - replace very small values with zeros
    underflow_mask = torch.abs(point_features) < 1e-7  # Threshold for what's considered underflow
    point_features = torch.where(underflow_mask, torch.zeros_like(point_features), point_features)
    
    # Convert to bfloat16 for better dynamic range
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        point_features = point_features.to(torch.bfloat16)
    
    return point_features
