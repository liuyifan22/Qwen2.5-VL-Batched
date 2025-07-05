import torch
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, BatchFeature


class Qwen2_5_VLProcessorBatched(Qwen2_5_VLProcessor):

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self._mean = torch.tensor(self.image_processor.image_mean)
        self._std = torch.tensor(self.image_processor.image_std)
        self._patch_size = self.image_processor.patch_size
        self._merge_size= self.image_processor.merge_size
        self._temporal_patch_size= self.image_processor.temporal_patch_size

    def process_images(self, images):
        # Images must have shape (B, 3, h, w) and be in [0, 1]
        # Normalize first
        images = images - self._mean.to(images.device)[None, :, None, None]
        images = images / self._std.to(images.device)[None, :, None, None]

        # Unsqueeze: simulate a temporal dimension of 1
        images = images[:, None]  # (B, 1, 3, h, w)
        # Patchify: pad if necessary
        if images.shape[1] % self._temporal_patch_size != 0:
            pad_len = (
                self._temporal_patch_size
                - (images.shape[1] % self._temporal_patch_size)
            )
            repeats = images[:, -1:].repeat(1, pad_len, 1, 1, 1)
            images = torch.cat([images, repeats], 1)  # (B, t, 3, h, w)

        # Reshape contiguously
        b, t, channel, h, w = images.shape
        grid_h = h // self._patch_size
        grid_w = w // self._patch_size
        grid_t = t // self._temporal_patch_size
        images = images.reshape(
            b,
            grid_t,
            self._temporal_patch_size,
            channel,
            grid_h // self._merge_size,
            self._merge_size,
            self._patch_size,
            grid_w // self._merge_size,
            self._merge_size,
            self._patch_size
        ).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

        # Reshape in a weird way
        flattened = images.reshape(
            b * grid_t * grid_h * grid_w,
            channel * self._temporal_patch_size * self._patch_size * self._patch_size
        )

        image_grid_thw = torch.tensor([
            (grid_t, grid_h, grid_w)
            for _ in range(b)
        ])
        return {"pixel_values": flattened, "image_grid_thw": image_grid_thw}

    def __call__(
        self,
        images=None,
        text=None,
        videos=None,
        **kwargs,
    ):
        output_kwargs = {
            'text_kwargs': {'padding': True, 'return_tensors': 'pt'},
            'images_kwargs': {'return_tensors': 'pt'},
            'audio_kwargs': {'padding': True, 'return_tensors': 'pt'},
            'videos_kwargs': {'fps': 2.0, 'return_tensors': 'pt'},
            'common_kwargs': {'return_tensors': 'pt'}
        }

        image_inputs = self.process_images(images=images)
        image_grid_thw = image_inputs["image_grid_thw"]

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})


if __name__ == "__main__":
    self = Qwen2_5_VLProcessorBatched.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    images = torch.arange(224**2).reshape(1, 1, 224, 224)
    # Unsqueeze: simulate a temporal dimension of 1
    images = images[:, None]  # (B, 1, 3, h, w)
    # Patchify: pad if necessary
    if images.shape[1] % self._temporal_patch_size != 0:
        pad_len = (
            self._temporal_patch_size
            - (images.shape[1] % self._temporal_patch_size)
        )
        repeats = images[:, -1:].repeat(1, pad_len, 1, 1, 1)
        images = torch.cat([images, repeats], 1)  # (B, t, 3, h, w)

    # Reshape contiguously
    b, t, channel, h, w = images.shape
    grid_h = h // self._patch_size
    grid_w = w // self._patch_size
    grid_t = t // self._temporal_patch_size
    images1 = images.reshape(
        b,
        grid_t,
        self._temporal_patch_size,
        channel,
        grid_h // self._merge_size,
        self._merge_size,
        self._patch_size,
        grid_w // self._merge_size,
        self._merge_size,
        self._patch_size
    ).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)

    # Reshape in a weird way
    flattened = images1.reshape(
        b * grid_t * grid_h * grid_w,
        channel * self._temporal_patch_size * self._patch_size * self._patch_size
    )

    images2 = images.reshape(
        b,
        grid_t,
        self._temporal_patch_size,
        channel,
        grid_h,
        self._patch_size,
        grid_w,
        self._patch_size
    ).permute(0, 1, 4, 6, 3, 2, 5, 7)

    # Reshape in a non-weird way
    flattened2 = images2.reshape(
        b * grid_t * grid_h * grid_w,
        channel * self._temporal_patch_size * self._patch_size * self._patch_size
    )

    # Compare sets of patch elements
    # flattened = flattened.reshape(8, 2, 8, 2, 392).permute(0, 2, 1, 3, 4)
    flattened = flattened.reshape(8*8, 2*2*392)
    flattened2 = flattened2.reshape(8, 2, 8, 2, 392).permute(0, 2, 1, 3, 4)
    flattened2 = flattened2.reshape(8*8, 2*2*392)
    for fl1, fl2 in zip(flattened, flattened2):
        # import ipdb; ipdb.set_trace()
        assert set(fl1.cpu().numpy().tolist()) == set(fl2.cpu().numpy().tolist())

    import ipdb; ipdb.set_trace()
