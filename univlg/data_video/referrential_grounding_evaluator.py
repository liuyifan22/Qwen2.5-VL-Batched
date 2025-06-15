# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import copy
import itertools
import logging
import numpy as np
import torch
from prettytable import PrettyTable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pytorch3d.ops import knn_points
import pyviz3d.visualizer as viz
from pathlib import Path
import detectron2.utils.comm as comm
from detectron2.evaluation import DatasetEvaluator
from collections import Counter
import json

from univlg.utils.bbox_utils import _set_axis_align_bbox, _get_ious, get_3d_box_scanrefer
from univlg.utils.misc import all_gather, is_dist_avail_and_initialized

import ipdb
st = ipdb.set_trace

def box_xyzxyz_to_cxcyczwhd(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    x_c = 0.5 * (x0 + x1)
    y_c = 0.5 * (y0 + y1)
    z_c = 0.5 * (z0 + z1)
    w = x1 - x0
    h = y1 - y0
    d = z1 - z0
    return torch.stack([x_c, y_c, z_c, w, h, d], dim=-1)

def visualize_pc_masks_and_bbox(
    pc, color, gt_pcs,
    pred_pcs, gt_bbox, pred_bbox,
    data_dir=None, sample_name=None, inputs=None,
    gt_anchor_pcs=None, gt_anchor_bboxs=None, sr3d_data=None,
    anchor_pcs=None, anchor_bboxs=None
):
    """
    Input
        pc: N X 3
        color: N X 3 (0-255)
        gt_masks: M X N
        pred_masks: M_pred X N
        gt_bbox: M X 6
        pred_bbox: M_pred X 6 (min-max box) 
    """
    point_size = 25
    v = viz.Visualizer()
    v.add_points("RGB", pc,
                 colors=color,
                 alpha=0.8,
                 visible=True,
                 point_size=25)

     # add gt masks
    masks_colors = [np.tile(np.array([0, 255, 0])[None], (pc.shape[0], 1)) for pc in gt_pcs]
    v.add_points(
        "Instances (GT)", gt_pcs[0],
        colors=masks_colors[0],
        alpha=0.8,
        visible=False,
        point_size=point_size
    )

    # add pred masks
    dists = knn_points(torch.from_numpy(pc[None]).cuda(), torch.from_numpy(pc[None]).cuda(), K=8)[0][0, :, 1:].mean(1)
    threshold = dists.mean() + 2 * dists.std()
    pc = pc[(dists < threshold).cpu().numpy()]
    masks_colors = [np.tile(np.array([255, 0, 0])[None], (pc.shape[0], 1)) for pc in pred_pcs]
    v.add_points(
        "Instances (PRED)", pred_pcs[0],
        colors=masks_colors[0],
        alpha=0.8,
        visible=False,
        point_size=point_size)

    # add gt boxes
    gt_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(gt_bbox)).numpy()
    v.add_bounding_box(
        'Boxes (GT)',
        position=gt_bbox[..., :3][0],
        size=gt_bbox[..., 3:][0],
        color=np.array([0, 255, 0]),
        alpha=0.8,
        edge_width=0.03
    )

    # add pred boxes
    pred_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(pred_bbox)).numpy()
    v.add_bounding_box(
        'Boxes (Pred)',
        position=pred_bbox[..., :3][0],
        size=pred_bbox[..., 3:][0],
        color=np.array([255, 0, 0]),
        alpha=0.8,
        visible=True,
        edge_width=0.03)

    if gt_anchor_pcs is not None:
        anchor_colors = get_color(len(gt_anchor_pcs))
        for i in range(len(gt_anchor_pcs)):
            _anchor_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(gt_anchor_bboxs[i])).numpy()
            _anchor_name = sr3d_data['anchors_names'][i]
            v.add_points(
                f"A PC {i},{_anchor_name[:5]}", gt_anchor_pcs[i],
                colors=np.array(anchor_colors[i])[None].repeat(gt_anchor_pcs[i].shape[0], axis=0),
                alpha=0.8, visible=False, point_size=point_size)
            v.add_bounding_box(
                f'A BB {i},{_anchor_name[:5]}',
                position=_anchor_bbox[..., :3][0],
                size=_anchor_bbox[..., 3:][0],
                color=np.array([0, 255, 0]),
                alpha=0.8,
                edge_width=0.03,
                visible=False
            )

        v.add_labels(
            'Labels',
            [sr3d_data['text_caption'], sr3d_data['target_name'], sr3d_data['anchors_names']],
            [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])],
            [np.array([255.0, 0.0, 0.0]), np.array([0.0, 255.0, 0.0]), np.array([0.0, 0.0, 255.0])],
            visible=True
        )

    if anchor_pcs is not None:
        anchor_colors = get_color(len(anchor_pcs))
        for i in range(0, min(len(gt_anchor_pcs), len(anchor_pcs))):
            _anchor_bbox = box_xyzxyz_to_cxcyczwhd(torch.from_numpy(anchor_bboxs[i])).numpy()
            v.add_points(
                f"PA PC {i}", anchor_pcs[i],
                colors=np.array(anchor_colors[i])[None].repeat(anchor_pcs[i].shape[0], axis=0),
                alpha=0.8, visible=False, point_size=point_size)
            v.add_bounding_box(
                f'PA BB {i}',
                position=_anchor_bbox[..., :3][0],
                size=_anchor_bbox[..., 3:][0],
                color=np.array(anchor_colors[i]),
                alpha=0.8,
                edge_width=0.03,
                visible=False
            )

    if gt_anchor_pcs is not None and anchor_pcs is not None:
        print(f"Found {len(gt_anchor_pcs)} GT Anchors and {len(anchor_pcs)} Pred Anchors")

    if data_dir is None:
        data_dir = os.environ['OUTPUT_DIR_PREFIX'] + '/debug/bdetr2_visualizations'

    data_dir = Path(f"{data_dir}/{inputs[0]['dataset_name']}/{sample_name.replace(' ', '_')[:100]}")
    if not data_dir.exists():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saved to {data_dir}")
    v.save(str(data_dir))

def get_color(max_value: int, colormap='spring'):
    colormap = plt.get_cmap('spring')  # Pink is 0, Yellow is 1
    colors = [mcolors.to_rgb(colormap(i / max_value)) for i in range(max_value)]  # Generate colors
    return (np.array(colors) * 255).astype(int).tolist()

def convert_bbox_to_corners_with_colors(bboxes):
    """
    Convert bounding boxes to a format with 8 corners and deterministically generate a color for each box.

    Args:
    - bboxes (np.array): An array of shape [N, 6] containing bounding boxes.

    Returns:
    - np.array: An array of dictionaries, each with 'corners' and 'color' keys.
    """
    converted = []
    colors = get_color(len(bboxes))
    for idx, bbox in enumerate(bboxes):
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        corners = np.array([
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
        ])
        converted.append({"corners": corners.tolist(), "color": colors[idx]})
    return np.array(converted)

def crop_pcd_to_combined_bbox(pcd: torch.Tensor, bboxes: torch.Tensor, extra_extent: float) -> torch.Tensor:
    """
    Crop point cloud data (PCD) to match the combined extent of all bounding boxes.

    Parameters:
    - pcd: torch.Tensor of shape [M, 6] representing the point cloud data with [xyz rgb].
    - bboxes: torch.Tensor of shape [N, 6] representing the bounding boxes with [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
    - torch.Tensor: Cropped PCD matching the combined extent of all bounding boxes.
    """
    # Calculate the min/max extent of all bounding boxes
    combined_min = torch.min(bboxes[:, :3], dim=0).values - extra_extent
    combined_max = torch.max(bboxes[:, 3:], dim=0).values + extra_extent

    # Check if points are within the combined bbox extents
    in_combined_bbox = (pcd[:, 0] >= combined_min[0]) & (pcd[:, 0] <= combined_max[0]) & \
                        (pcd[:, 1] >= combined_min[1]) & (pcd[:, 1] <= combined_max[1]) & \
                        (pcd[:, 2] >= combined_min[2]) & (pcd[:, 2] <= combined_max[2])

    return pcd[in_combined_bbox]

def sample_k_rows(tensor, K):
    indices = torch.randperm(tensor.size(0))[:K]
    return tensor[indices]

class ReferrentialGroundingEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        thresholds,
        topks,
        cfg=None,
        checkpoint_dir=None  # Add checkpoint directory parameter
    ):
        self._logger = logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.thresholds = thresholds
        self.topks = topks
        self.cfg = cfg
        self._cpu_device = torch.device("cpu")
        self.num_viz = 0
        
        # Setup checkpointing
        self.checkpoint_dir = "./eval_milestones"
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f"{dataset_name}_eval_checkpoint_prompt_engineering.pkl")
        if not is_dist_avail_and_initialized() or comm.is_main_process():
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.processed_count = 0
        self.checkpoint_frequency = 10  # Save every 100 samples
        self.use_checkpoint = False

    def reset(self):
        # Try to load existing checkpoint
        if os.path.exists(self.checkpoint_file) and self.use_checkpoint:
            if not is_dist_avail_and_initialized() or comm.is_main_process():
                try:
                    with open(self.checkpoint_file, 'rb') as f:
                        checkpoint = pickle.load(f)
                    self.detection_results = checkpoint['detection_results']
                    self.mask_detection_results = checkpoint['mask_detection_results']
                    self.processed_count = checkpoint['processed_count']
                    self._logger.info(f"Resumed evaluation from checkpoint with {self.processed_count} processed samples")
                except Exception as e:
                    self._logger.warning(f"Failed to load checkpoint: {e}. Starting fresh evaluation.")
                    self.detection_results = []
                    self.mask_detection_results = []
                    self.processed_count = 0
            
            # Synchronize checkpoint data across all processes in distributed training
            if is_dist_avail_and_initialized():
                # Broadcast processed_count
                count_tensor = torch.tensor([self.processed_count if comm.is_main_process() else 0], 
                                        dtype=torch.long, device="cuda")
                torch.distributed.broadcast(count_tensor, 0)
                self.processed_count = count_tensor.item()
                
                # If we loaded a checkpoint, broadcast the results to all processes
                if self.processed_count > 0:
                    # On main process, prepare data for broadcast
                    if comm.is_main_process():
                        # Pack results into a serializable format
                        packed_results = {"detection": self.detection_results, "mask": self.mask_detection_results}
                        buffer = pickle.dumps(packed_results)
                        size = torch.tensor([len(buffer)], dtype=torch.long, device="cuda")
                        # Convert buffer to tensor on CPU first
                        buffer_tensor = torch.ByteTensor(list(buffer)).to("cuda")
                    else:
                        size = torch.tensor([0], dtype=torch.long, device="cuda")
                    
                    # Broadcast size first
                    torch.distributed.broadcast(size, 0)
                    
                    # Then prepare and broadcast the actual data
                    if not comm.is_main_process():
                        buffer_tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")
                    
                    torch.distributed.broadcast(buffer_tensor, 0)
                    
                    # Unpack on non-main processes
                    if not comm.is_main_process():
                        buffer = buffer_tensor.cpu().numpy().tobytes()
                        packed_results = pickle.loads(buffer)
                        self.detection_results = packed_results["detection"]
                        self.mask_detection_results = packed_results["mask"]
                        
        else:
            self.detection_results = []
            self.mask_detection_results = []
            self.processed_count = 0
            
        self.num_viz = 0
        # Initialize error tracking list
        self.error_codes = []
        
        # Define error type descriptions for reporting
        self.error_types = {
            1000: "Invalid view number",
            2000: "Steps timeout in exploration",
            3000: "Invalid bounding box format",
            4000: "Wrong image (GT not visible)",
            5000: "Other errors"
        }

    def save_checkpoint(self):
        """Save current evaluation progress to checkpoint file"""
        if not is_dist_avail_and_initialized() or comm.is_main_process():
            checkpoint = {
                'detection_results': self.detection_results,
                'mask_detection_results': self.mask_detection_results,
                'processed_count': self.processed_count,
            }
            # Atomic write to prevent corruption if interrupted
            tmp_file = f"{self.checkpoint_file}.tmp"
            with open(tmp_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            os.rename(tmp_file, self.checkpoint_file)
            self._logger.info(f"Saved evaluation checkpoint after {self.processed_count} samples")

    # def reset(self):
    #     self.detection_results = []
    #     self.mask_detection_results = []
    #     self.num_viz = 0

    def process(self, inputs, outputs):
        # if type(outputs[0]) != dict:
        #     outputs = outputs[1]
        # assert len(inputs) == 1
        # for j in range(len(outputs)):
        #     inputs_ = copy.copy(inputs[0])
        #     inputs_['sr3d_data'] = [inputs_['sr3d_data'][j]]
        #     if self.cfg.USE_GT_MASKS: # yifan: we all use gt masks, FIX this
        #         self.process_single([inputs_], [outputs[j]])
        #     else:
        #         self.process_single([inputs_], [outputs[j]])
        self.process_single(inputs, outputs)
                
    def process_single_gt(self, inputs, outputs):
        """
        Process a single input-output pair for 3D referential grounding evaluation.

        This function handles one sample (batch size = 1) from a 3D scene and its corresponding model
        predictions. It extracts predicted instance scores and masks, computes axis-aligned bounding boxes
        (AABBs) for the top-K predictions, and evaluates these predictions against the ground truth target
        (specified in the input). Depending on configuration flags, it may also perform subsampling of the
        point cloud or visualize the results.

        Parameters
        ----------
        inputs : list[dict]
            A list containing one dictionary with the following required keys:
            - "scannet_coords": torch.Tensor of shape (P, 3)
                    3D coordinates of the scene's point cloud (e.g., P ≈ 2.5e5 points).
            - "scannet_color": torch.Tensor of shape (P, 3)
                    Color information for each point (RGB values in 0-255).
            - "scannet_labels": torch.Tensor of shape (P, 2)
                    Per-point labels; the second column (index 1) is used to match the ground truth target.
            - "image_id": str
                    A unique identifier for the scene.
            - "file_name": str
                    File path string used to extract the scene name for visualization.
            - 'dataset_name': str
                    Name of the dataset (used for visualization path).
            - "sr3d_data": list of dict
                    A list with one dictionary that must include:
                    - "target_id": int
                        The ground truth object identifier to detect.
                    - "annotation_id": int or None
                        The annotation id for the target.
                    - "anchor_ids": list of int
                        A list of additional anchor object identifiers (used if visualization is enabled).
                    - "text_caption": str
                        A caption or description for the scene/object.
                    - 'anchors_names': list[str]
                        Names of anchor objects (used for extra viz).
                    - 'target_name': str
                        Name of the target object (used for viz).
                    (Other entries in sr3d_data may be present but are not used in this function.)

        outputs : list[dict]
            A list containing one dictionary with a key "instances_3d", which is a dict that must include:
            - "pred_scores": torch.Tensor of shape (N, C)
                    Prediction confidence scores for N candidate instances. The first column (index 0) is used
                    as the main score (e.g., shape (100, 3)).
            - "pred_masks": torch.Tensor of shape (N, M)
                    Raw prediction masks (pre-thresholding) for each instance over M points (e.g., M ≈ 2.5e5).
            - "scannet_idxs": torch.Tensor or None (optional)
                    Indices used for subsampling the point cloud when FORCE_SUBSAMPLE is enabled.
            - 'reduced_scene_id_to_original_id': torch.Tensor | None
            - (Other keys like "scannet_gt_masks", etc. may be present but are not used.)

        Processing Details
        ------------------
        - The function extracts the predicted scores and applies a sigmoid to the predicted masks,
        thresholding them at 0.5 to yield binary masks.
        - It selects the top-K predictions (with K equal to max(self.topks)) based on the confidence scores.
        - For each of the top predictions, it uses the binary mask to index into "scannet_coords" (a tensor of shape (P, 3))
        to extract the corresponding 3D points, and computes an axis-aligned bounding box via _set_axis_align_bbox.
        If no points are selected, a box with -∞ values is used.
        - If the FORCE_SUBSAMPLE flag is enabled, the function subsamples "scannet_coords", "scannet_labels", and
        "scannet_color" using indices from "scannet_idxs".
        - The ground truth points for the target object (specified by "target_id" in sr3d_data) are
        extracted from "scannet_coords" using "scannet_labels". An axis-aligned bounding box for the ground truth is
        computed, and Intersection-over-Union (IoU) is calculated between each predicted box and the ground truth box.
        - Similarly, mask IoUs are computed between the predicted binary masks and a full ground truth mask derived
        from "scannet_labels".
        - The detection and mask detection results (each a boolean array indicating success at various thresholds)
        are appended to the evaluator's internal lists for later aggregation.

        Returns
        -------
        None
        """

        assert len(inputs) == len(outputs) == 1
        
        # Root word is the first object
        scores = outputs[0]['instances_3d']['pred_scores'][:, 0]
        
        # Average confidence of each prediction mask.
        top_k_weighted_scores = scores
        max_k = max(self.topks)
        
        top_id = torch.argsort(top_k_weighted_scores, descending=True)[:max_k]
        pred_ids = (outputs[0]['instances_3d']['pred_masks'])[top_id, :].argmax(-1)
        reduced_scene_id_to_original_id = outputs[0]['instances_3d']['reduced_scene_id_to_original_id']
        pred_ids = reduced_scene_id_to_original_id[pred_ids]
        target_id = inputs[0]['sr3d_data'][0]['target_id']
        detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
        correct = (pred_ids == target_id).cpu().numpy()
        for i in range(len(self.topks)):
            detected[i, :] = np.any(correct[:self.topks[i], None], axis=0)
        self.detection_results.append(detected)
        self.mask_detection_results.append(detected)

    def process_single(self, inputs, outputs):
        """
        Process a single prediction with the new format: 
        {'ids': [id1, id2, ...], 'ious': [iou1, iou2, ...]}
        
        For evaluation, we simply check if predicted IDs match the ground truth target ID.
        If they match, we consider it as a perfect match (IoU=1.0).
        If they don't match, we consider it a complete miss (IoU=0.0).
        """
        if isinstance(outputs[0], int) and outputs[0] >= 1000:
            error_code = outputs[0]
            self.error_codes.append(error_code)
            
            # Print the error for immediate feedback
            if hasattr(self, 'error_types') and error_code in self.error_types:
                print(f"Error detected: {self.error_types[error_code]} (code {error_code})")
            
            # Create empty detection result (counts as a failure)
            detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
            self.detection_results.append(detected)
            self.mask_detection_results.append(detected)
            self.processed_count += 1
            
            # Save checkpoint if needed
            if self.processed_count % self.checkpoint_frequency == 0 and self.use_checkpoint:
                self.save_checkpoint()
            
            return
        
        # Standard processing for valid outputs
        detected = np.zeros((len(self.topks), len(self.thresholds)), dtype=bool)
        
        # Get ground truth target ID
        target_id = inputs[0]['sr3d_data'][0]['target_id']
        
        # Get predictions from the output
        try:
            pred_ids = outputs[0].get('ids', [])
            # We don't need the original IoUs
            _ = outputs[0].get('ious', [])
        except:
            # If we can't extract predictions, count it as "Other error"
            self.error_codes.append(5000)
            self.detection_results.append(detected)
            self.mask_detection_results.append(detected)
            self.processed_count += 1
            return
        
        if len(pred_ids) == 0:
            self.detection_results.append(detected)
            self.mask_detection_results.append(detected)
            self.processed_count += 1
            return
            
        # Convert to numpy arrays for easier processing
        pred_ids = np.array(pred_ids)
        
        # Create binary mask of correct predictions (1.0 for match, 0.0 for no match)
        binary_correctness = np.zeros_like(pred_ids, dtype=float)
        binary_correctness[pred_ids == target_id] = 1.0
        
        # Number of predictions might be less than max_k
        max_k = max(self.topks)
        n_preds = len(pred_ids)
        
        # For each top-k setting
        for i, k in enumerate(self.topks):
            actual_k = min(k, n_preds)  # Handle case when fewer than k predictions are available
            
            # For each threshold
            for j, threshold in enumerate(self.thresholds):
                # If threshold > 0, we need a correct prediction to pass
                # If threshold = 0, any prediction passes regardless of correctness
                if threshold > 0:
                    detected[i, j] = np.any(binary_correctness[:actual_k] > 0)
                else:
                    # Special case: threshold=0 means any prediction passes
                    detected[i, j] = actual_k > 0
        
        # Store detection results (use same values for both bbox and mask metrics)
        self.detection_results.append(detected)
        self.mask_detection_results.append(detected)
        self.processed_count += 1
        
        # Save checkpoint periodically (only on main process if distributed)
        if self.processed_count % self.checkpoint_frequency == 0 and self.use_checkpoint:
            self.save_checkpoint()
        
        # Print accumulated success accuracy after each iteration
        if not is_dist_avail_and_initialized() or comm.is_main_process():
            current_results = np.array(self.detection_results).astype(np.float32)
            current_acc = current_results.mean(axis=0)
            
            print(f"\n--- Accumulated success after {len(self.detection_results)} samples ---")
            for i, k in enumerate(self.topks):
                metrics = [f"{current_acc[i, j]:.3f}" for j in range(len(self.thresholds))]
                print(f"Top-{k}: " + " | ".join([f"thresh={t}:{m}" for t, m in zip(self.thresholds, metrics)]))

    def evaluate(self):
        # First save a final checkpoint before gathering results
        if self.processed_count > 0 and self.use_checkpoint:
            self.save_checkpoint()
            
        # Wait for all processes to finish saving their checkpoints
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()
            
        if is_dist_avail_and_initialized():
            detection_results = all_gather(self.detection_results)
            detection_results = list(itertools.chain(*detection_results))
            mask_detection_results = all_gather(self.mask_detection_results)
            mask_detection_results = list(itertools.chain(*mask_detection_results))
            
            # Gather error codes as well
            error_codes = all_gather(self.error_codes)
            error_codes = list(itertools.chain(*error_codes))
            
            if not comm.is_main_process():
                return {}
        else:
            detection_results = self.detection_results
            mask_detection_results = self.mask_detection_results
            error_codes = self.error_codes

        if self.cfg.TEST_DATASET_INFERENCE:
            try:
                Path(self.cfg.TEST_RESULT_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
                print(f'exporting test results to {self.cfg.TEST_RESULT_EXPORT_PATH}/{self.dataset_name}_test_results.json')
                with open(f'{self.cfg.TEST_RESULT_EXPORT_PATH}/{self.dataset_name}_test_results.json', 'w') as json_file:
                    json.dump(detection_results, json_file, indent=4)
            except Exception as e:
                print(f"Error exporting test results: {e}")
                st()
            return None

        self.detection_results = []
        self.mask_detection_results = []
        
        total_samples = len(detection_results)
        detection_results = np.array(detection_results).astype(np.float32).mean(axis=0)
        mask_detection_results = np.array(mask_detection_results).astype(np.float32).mean(axis=0)
            
        table = PrettyTable()
        table.field_names = ['Dataset', ''] + ['thresold = ' + str(thres) for thres in self.thresholds]
        for i in range(len(self.topks)):
            row = [self.dataset_name, f'top {self.topks[i]} scores (BOX)'] + ['{:.3f}'.format(detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        for i in range(len(self.topks)):
            row = [self.dataset_name, f'top {self.topks[i]} scores (MASK)'] + ['{:.3f}'.format(mask_detection_results[i, j]) for j in range(len(self.thresholds))]
            table.add_row(row)
        print(table)

        res = {}
        
        for i in range(len(self.topks)):
            for j in range(len(self.thresholds)):
                res[f"top_{self.topks[i]}_threshold_{self.thresholds[j]}"] = detection_results[i, j]

        for i in range(len(self.topks)):
            for j in range(len(self.thresholds)):
                res[f"mask_top_{self.topks[i]}_threshold_{self.thresholds[j]}"] = mask_detection_results[i, j]
        
        from collections import Counter
        error_counts = Counter(error_codes)
        
        # Report errors
        if error_counts:
            error_table = PrettyTable()
            error_table.field_names = ['Error Type', 'Count', 'Percentage']
            
            
            
            for code, label in self.error_types.items():
                count = error_counts.get(code, 0)
                if count > 0:
                    percent = (count / total_samples) * 100
                    error_table.add_row([label, count, f'{percent:.2f}%'])
            
            print("\nError Analysis:")
            print(error_table)
            
            # Add error statistics to results
            for code, label in self.error_types.items():
                count = error_counts.get(code, 0)
                percent = (count / total_samples) * 100
                res[f"error_{code}_{label.replace(' ', '_').lower()}"] = percent
        
        return res
        
