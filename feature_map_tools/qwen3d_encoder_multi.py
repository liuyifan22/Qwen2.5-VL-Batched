import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import open3d as o3d
from torch_scatter import scatter_mean
from PIL import Image

class Qwen3DEncoder(nn.Module):
    """
    A 3D encoder that uses a pre-initialized Qwen vision-language model to extract features 
    from multiple images and projects them into 3D space using camera parameters and depth information.
    """
    
    def __init__(
        self, 
        model, 
        processor,
        voxel_size: float = 0.05,
        feature_dim: int = 2048,
        min_points_per_voxel: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize the Qwen3DEncoder with pre-initialized model and processor.
        
        Args:
            model: Pre-initialized Qwen VL model (e.g., Qwen2.5-VL-3B-Instruct)
            processor: Pre-initialized Qwen processor
            voxel_size: Size of voxels for point cloud discretization
            feature_dim: Dimension of the feature vectors
            min_points_per_voxel: Minimum number of points required in a voxel
            device: Device to run the model on (if None, uses model's device)
        """
        super().__init__()
        
        # run visual encoder in fp16
        self.processor = processor
        self.device = device or next(model.parameters()).device
        self.voxel_size = voxel_size
        self.feature_dim = feature_dim
        self.min_points_per_voxel = min_points_per_voxel
        
        # Access the vision encoder component
        self.visual = model.to(device)
        
        # Ensure the model is in eval mode for feature extraction
        self.visual.eval()
        
    def extract_features_parallel(
        self, 
        images: Union[torch.Tensor, List[Image.Image]],
        text: Optional[str] = None,
        stream: Optional[torch.cuda.Stream] = None
    ) -> torch.Tensor:
        """
        Extract features from images using the Qwen visual encoder with CUDA stream support.
        
        Args:
            images: Batch of images [B, C, H, W] or list of PIL images
            text: Optional text prompt to condition the feature extraction
            stream: Optional CUDA stream for parallelization
            
        Returns:
            features: Extracted features [B, N, C]
        """
        # Use the provided stream or default stream
        with torch.cuda.stream(stream if stream else torch.cuda.current_stream()):
            with torch.no_grad(), torch.cuda.amp.autocast():
                # Process images
                image_inputs = self.processor.image_processor(
                    images=images,
                    do_rescale=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get the processor-calculated grid_thw
                processed_images = image_inputs['pixel_values']
                processor_grid_thw = image_inputs['image_grid_thw']
                
                # Use the model's visual component to extract features
                vision_outputs = self.visual(
                    processed_images, 
                    grid_thw=processor_grid_thw
                )
                    
        return vision_outputs, processor_grid_thw
    
    def project_to_3d(
        self, 
        features: torch.Tensor,
        depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        feature_grid_shape: Tuple[int, int, int],
        return_valid_mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Project image features to 3D space using depth and camera parameters.
        
        Args:
            features: Image features [B, N, C]
            depths: Batch of depth maps [B, H, W]
            intrinsics: Camera intrinsic matrices [B, 3, 3]
            extrinsics: Camera extrinsic matrices [B, 4, 4]
            feature_grid_shape: Shape of the feature grid B* (T, H, W)
            return_valid_mask: Whether to return valid mask
            
        Returns:
            points_3d: 3D coordinates [B, N, 3]
            features_3d: Features at each point [B, N, C]
            valid_mask: Optional mask of valid points [B, N]
        """
        batch_size = depths.shape[0]
        
        if len(features.shape) == 2:
            # Calculate features per image
            features_per_image = features.shape[0] // batch_size
            # Reshape to [B, N, C]
            features = features.view(batch_size, features_per_image, -1)
        
        assert batch_size == features.shape[0]
        
        # Early return for empty batches
        if batch_size == 0:
            empty_points = torch.zeros(0, 3, device=self.device)
            empty_features = torch.zeros(0, features.shape[-1] if features.shape[-1] > 0 else self.feature_dim, 
                                        device=self.device)
            empty_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            
            if return_valid_mask:
                return empty_points, empty_features, empty_mask
            else:
                return empty_points, empty_features
        
        # --- Vectorized projection to 3D coordinates ---
        # Unpack grid shape and image dims
        T, H, W = feature_grid_shape[0]
        img_h, img_w = depths.shape[1], depths.shape[2]

        # Build feature‐grid [H*W,2]
        grid_h = torch.linspace(0, H-1, H, device=self.device)
        grid_w = torch.linspace(0, W-1, W, device=self.device)
        yy, xx = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [HW,2]

        # Scale to pixel coords & center, then normalize to [-1,1]
        scale = torch.tensor([img_w / W, img_h / H], device=self.device)
        grid_pix = grid * scale + (scale / 2)
        norm_grid = torch.stack([
            2.0 * grid_pix[:,0] / img_w - 1.0,
            2.0 * grid_pix[:,1] / img_h - 1.0
        ], dim=-1)  # [HW,2]

        # Prepare batched grid for sampling: [B, HW, 1, 2]
        grid_batched = norm_grid.view(1, H*W, 1, 2).expand(batch_size, -1, -1, -1)

        # Sample depths in one go: depths [B,H,W]→[B,1,H,W]
        sampled = F.grid_sample(
            depths.unsqueeze(1),      # [B,1,H,W]
            grid_batched,             # [B,HW,1,2]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [B,1,HW,1]
        z = sampled.view(batch_size, H*W)              # [B,HW]

        # Compute camera coords in parallel
        fx = intrinsics[:,0,0].unsqueeze(1)  # [B,1]
        fy = intrinsics[:,1,1].unsqueeze(1)
        cx = intrinsics[:,0,2].unsqueeze(1)
        cy = intrinsics[:,1,2].unsqueeze(1)

        x = (grid_pix[:,0].unsqueeze(0) - cx) * z / fx  # [B,HW]
        y = (grid_pix[:,1].unsqueeze(0) - cy) * z / fy
        pts_cam = torch.stack([x, y, z], dim=-1)        # [B,HW,3]

        # Valid mask
        valid_mask = z > 0.01                            # [B,HW]

        # To world coords: add homogeneous 1, then batch‐matmul with extrinsics
        ones = torch.ones(batch_size, H*W, 1, device=self.device)
        hom = torch.cat([pts_cam, ones], dim=-1)        # [B,HW,4]
        world = hom @ extrinsics.transpose(1,2)         # [B,HW,4]
        points_3d = world[...,:3]                       # [B,HW,3]

        # Features already [B,HW,C] or reshape if needed
        features_3d = features if features.shape[1]==H*W else features.view(batch_size, H*W, -1)
        
        return (points_3d, features_3d, valid_mask) if return_valid_mask else (points_3d, features_3d)
    
    def voxelize_pointcloud(
        self, 
        points_3d: torch.Tensor, 
        features_3d: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voxelize the point cloud and average features within each voxel.
        
        Args:
            points_3d: 3D coordinates [B, N, 3]
            features_3d: Features at each point [B, N, C]
            valid_mask: Mask of valid points [B, N]
            
        Returns:
            voxelized_points: Voxelized point cloud [M, 3]
            voxelized_features: Features for each voxel [M, C]
        """
        # Vectorized across the batch:
        # 1) Flatten and mask valid points/features
        B, N, _ = points_3d.shape
        C = features_3d.shape[-1]
        pts_flat = points_3d.reshape(B * N, 3)
        feats_flat = features_3d.reshape(B * N, C)
        mask_flat = valid_mask.reshape(B * N)

        pts_valid = pts_flat[mask_flat]
        feats_valid = feats_flat[mask_flat]
        if pts_valid.numel() == 0:
            return (torch.zeros(0, 3, device=self.device),
                    torch.zeros(0, C, device=self.device))

        # 2) Compute voxel indices and hash
        vox_idx = torch.floor(pts_valid / self.voxel_size).long()
        h1, h2, h3 = vox_idx[:,0].to(torch.int64), vox_idx[:,1].to(torch.int64), vox_idx[:,2].to(torch.int64)
        voxel_hash = h1 * 100_000_000 + h2 * 10_000 + h3

        # 3) Unique + inverse + counts
        uniq, inv, cnt = torch.unique(voxel_hash, return_inverse=True, return_counts=True)
        keep = cnt >= self.min_points_per_voxel
        if keep.sum() == 0:
            return (torch.zeros(0, 3, device=self.device),
                    torch.zeros(0, C, device=self.device))

        # 4) Scatter-mean features and centers
        vf = scatter_mean(feats_valid, inv, dim=0)            # [#vox, C]
        centers = (vox_idx.to(pts_valid.dtype) + 0.5) * self.voxel_size
        vp = scatter_mean(centers, inv, dim=0)                # [#vox, 3]

        # 5) Select kept voxels
        final_points   = vp[keep]
        final_features = vf[keep]
        return final_points, final_features
    
    def forward(
        self, 
        images: torch.Tensor,
        depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        text: Optional[str] = None,
        stream: Optional[torch.cuda.Stream] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of images with associated depth maps and camera parameters.
        
        Args:
            images: Batch of images [B, C, H, W]
            depths: Batch of depth maps [B, H, W]
            intrinsics: Camera intrinsic matrices [B, 3, 3]
            extrinsics: Camera extrinsic matrices [B, 4, 4]
            text: Optional text prompt
            stream: Optional CUDA stream for parallelization
            
        Returns:
            points_3d: Voxelized point cloud [N, 3]
            features_3d: Features for each point [N, C]
        """
        # Extract features from images
        import time
        start_time = time.time()
        features, grid_thw = self.extract_features_parallel(images, text, stream)
        print(f">>> Feature extraction time: {time.time() - start_time:.2f} seconds")
        
        # Project features to 3D
        points_3d, features_3d, valid_mask = self.project_to_3d(
            features, 
            depths, 
            intrinsics, 
            extrinsics,
            feature_grid_shape=grid_thw,
            return_valid_mask=True
        )
        
        # Voxelize point cloud
        voxelized_points, voxelized_features = self.voxelize_pointcloud(
            points_3d, features_3d, valid_mask
        )
        
        return voxelized_points, voxelized_features
    
    def process_scene(
        self,
        images: List[torch.Tensor],
        depths: List[torch.Tensor],
        intrinsics: List[torch.Tensor],
        extrinsics: List[torch.Tensor],
        text: Optional[str] = None,
        batch_size: int = 4,
        num_streams: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a scene with multiple images using in-GPU parallelization with CUDA streams.
        
        Args:
            images: List of images [N, C, H, W]
            depths: List of depth maps [N, H, W]
            intrinsics: List of camera intrinsic matrices [N, 3, 3]
            extrinsics: List of camera extrinsic matrices [N, 4, 4]
            text: Optional text prompt
            batch_size: Batch size for processing
            num_streams: Number of CUDA streams to use for parallelization
            
        Returns:
            points_3d: Voxelized point cloud [M, 3]
            features_3d: Features for each point [M, C]
        """
        all_points = []
        all_features = []
        
        # Create CUDA streams for parallelization
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # Process in batches
        num_images = len(images)
        import time
        start_time = time.time()
        
        # Pre-allocate memory for results
        batch_results = []
        
        # Process batches in parallel using CUDA streams
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            stream_idx = (i // batch_size) % num_streams
            
            # Use the appropriate CUDA stream
            with torch.cuda.stream(streams[stream_idx]):
                # Prepare batch data
                batch_images = torch.stack(images[i:end_idx]).to(self.device)
                batch_depths = torch.stack(depths[i:end_idx]).to(self.device)
                batch_intrinsics = torch.stack(intrinsics[i:end_idx]).to(self.device)
                batch_extrinsics = torch.stack(extrinsics[i:end_idx]).to(self.device)
                
                # Process batch in this stream
                points, features = self.forward(
                    batch_images, 
                    batch_depths, 
                    batch_intrinsics, 
                    batch_extrinsics,
                    text
                )
                
                # Store results
                if points.shape[0] > 0:
                    batch_results.append((points.clone(), features.clone(), stream_idx))
        
        # Synchronize all streams before continuing
        for stream in streams:
            stream.synchronize()
        
        # Collect results in order
        for points, features, _ in batch_results:
            all_points.append(points)
            all_features.append(features)
        
        print(f"Processing all images time: {time.time() - start_time:.2f} seconds")
        
        # Continue with the existing voxelization code...
        # Combine results
        if not all_points:
            return torch.zeros(0, 3, device=self.device), torch.zeros(0, self.feature_dim, device=self.device)
            
        combined_points = torch.cat(all_points, dim=0)
        combined_features = torch.cat(all_features, dim=0)
        
        # --- VECTORIZED VOXELIZATION via scatter_mean ---
        # 1) compute integer voxel coords
        vox_idx = torch.floor(combined_points / self.voxel_size).long()
        # 2) unique hash
        h1 = vox_idx[:,0].to(torch.int64)
        h2 = vox_idx[:,1].to(torch.int64)
        h3 = vox_idx[:,2].to(torch.int64)
        voxel_hash = h1 * 100_000_000 + h2 * 10_000 + h3

        # 3) unique + inverse + counts
        uniq, inv, cnt = torch.unique(
            voxel_hash, return_inverse=True, return_counts=True
        )
        # 4) mask out small voxels
        keep = cnt >= self.min_points_per_voxel
        if keep.sum() == 0:
            return torch.zeros(0,3,device=self.device), torch.zeros(0,self.feature_dim,device=self.device)

        # 5) scatter_mean features into voxels
        vf = scatter_mean(combined_features, inv, dim=0)  # [#vox, C]

        # 6) compute voxel centers: (idx + 0.5)*vs
        centers = (vox_idx.to(combined_points.dtype) + 0.5) * self.voxel_size
        vp = scatter_mean(centers, inv, dim=0)  # [#vox, 3]

        # 7) select only kept voxels
        final_points = vp[keep]
        final_features = vf[keep]
        return final_points, final_features