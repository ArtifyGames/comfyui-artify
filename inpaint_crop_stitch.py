import math

import comfy.model_management
import nodes
from PIL import Image
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as VF


class InpaintOps:
    def rescale_i(self, samples, width, height, algorithm):
        # samples shape: [B, H, W, C]
        samples_chw = samples.movedim(-1, 1)
        algorithm_enum = getattr(Image, algorithm.upper())
        result = []
        for i in range(samples_chw.shape[0]):
            sample_pil = VF.to_pil_image(samples_chw[i].cpu()).resize((width, height), algorithm_enum)
            result.append(VF.to_tensor(sample_pil))
        return torch.stack(result, dim=0).to(samples.device).movedim(1, -1)

    def rescale_m(self, samples, width, height, algorithm):
        # samples shape: [B, H, W]
        algorithm_enum = getattr(Image, algorithm.upper())
        result = []
        for i in range(samples.shape[0]):
            sample_pil = VF.to_pil_image(samples[i].cpu()).resize((width, height), algorithm_enum)
            result.append(VF.to_tensor(sample_pil).squeeze(0))
        return torch.stack(result, dim=0).to(samples.device)

    def fillholes_iterative_hipass_fill_m(self, samples):
        # Flood-fill "outside" in inverted mask; everything else is hole or mask.
        bsz, h, w = samples.shape
        device = samples.device

        inv_mask = 1.0 - (samples > 0.5).float()
        padded_inv = torch.zeros((bsz, h + 2, w + 2), device=device, dtype=samples.dtype)
        padded_inv[:, 1:-1, 1:-1] = inv_mask

        outside = torch.zeros((bsz, h + 2, w + 2), device=device, dtype=samples.dtype)
        outside[:, 0, :] = 1
        outside[:, -1, :] = 1
        outside[:, :, 0] = 1
        outside[:, :, -1] = 1

        current = outside
        for _ in range(max(h, w)):
            next_outside = TF.max_pool2d(current.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            next_outside = next_outside * padded_inv
            next_outside[:, 0, :] = 1
            next_outside[:, -1, :] = 1
            next_outside[:, :, 0] = 1
            next_outside[:, :, -1] = 1
            if torch.equal(next_outside, current):
                break
            current = next_outside

        filled = 1.0 - current[:, 1:-1, 1:-1]
        return torch.max(samples, filled)

    def hipassfilter_m(self, samples, threshold):
        out = samples.clone()
        out[out < threshold] = 0
        return out

    def expand_m(self, mask, pixels):
        sigma = pixels / 4.0
        kernel_size = math.ceil(sigma * 1.5 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        return TF.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=padding).squeeze(1)

    def invert_m(self, samples):
        return 1.0 - samples

    def blur_m(self, samples, pixels):
        sigma = pixels / 4.0
        if sigma <= 0:
            return samples

        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        x = torch.arange(kernel_size, device=samples.device, dtype=samples.dtype) - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)).view(1, 1, kernel_size, kernel_size)

        blurred = TF.conv2d(samples.unsqueeze(1), kernel_2d, padding=kernel_size // 2, groups=1).squeeze(1)
        return blurred.clamp(0.0, 1.0)

    def batched_findcontextarea_m(self, mask):
        # mask shape: [B, H, W]
        bsz, h, w = mask.shape
        device = mask.device
        any_y = mask.max(dim=2).values > 0.5  # [B, H]
        any_x = mask.max(dim=1).values > 0.5  # [B, W]

        def get_min_max(any_dim, size):
            idx = torch.arange(size, device=device).unsqueeze(0).expand(bsz, -1)
            min_idx = torch.where(any_dim, idx, torch.tensor(size, device=device))
            max_idx = torch.where(any_dim, idx, torch.tensor(-1, device=device))
            lo = torch.min(min_idx, dim=1).values
            hi = torch.max(max_idx, dim=1).values
            empty = ~any_dim.any(dim=1)
            lo[empty] = -1
            hi[empty] = -1
            return lo, hi

        y_min, y_max = get_min_max(any_y, h)
        x_min, x_max = get_min_max(any_x, w)
        width = torch.where(x_min >= 0, x_max - x_min + 1, torch.tensor(-1, device=device))
        height = torch.where(y_min >= 0, y_max - y_min + 1, torch.tensor(-1, device=device))
        return x_min, y_min, width, height

    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        img_h, img_w = mask.shape[1], mask.shape[2]
        grow_x = (w.float() * (extend_factor - 1.0) / 2.0).round().long()
        grow_y = (h.float() * (extend_factor - 1.0) / 2.0).round().long()

        new_x = torch.clamp(x - grow_x, min=0)
        new_y = torch.clamp(y - grow_y, min=0)
        new_x2 = torch.clamp(x + w + grow_x, max=img_w)
        new_y2 = torch.clamp(y + h + grow_y, max=img_h)
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y

        empty = (w == -1)
        new_x[empty] = 0
        new_y[empty] = 0
        new_w[empty] = img_w
        new_h[empty] = img_h
        return new_x, new_y, new_w, new_h

    def batched_combinecontextmask_m(self, x, y, w, h, optional_context_mask):
        ox, oy, ow, oh = self.batched_findcontextarea_m(optional_context_mask)

        mask_empty = (x == -1)
        x_1 = torch.where(mask_empty, ox, x)
        y_1 = torch.where(mask_empty, oy, y)
        w_1 = torch.where(mask_empty, ow, w)
        h_1 = torch.where(mask_empty, oh, h)

        opt_empty = (ox == -1)
        ox_2 = torch.where(opt_empty, x_1, ox)
        oy_2 = torch.where(opt_empty, y_1, oy)
        ow_2 = torch.where(opt_empty, w_1, ow)
        oh_2 = torch.where(opt_empty, h_1, oh)

        new_x = torch.min(x_1, ox_2)
        new_y = torch.min(y_1, oy_2)
        new_x_max = torch.max(x_1 + w_1, ox_2 + ow_2)
        new_y_max = torch.max(y_1 + h_1, oy_2 + oh_2)
        new_w = new_x_max - new_x
        new_h = new_y_max - new_y

        both_empty = (x_1 == -1)
        new_x[both_empty] = -1
        new_y[both_empty] = -1
        new_w[both_empty] = -1
        new_h[both_empty] = -1
        return new_x, new_y, new_w, new_h

    def pad_to_multiple(self, value, multiple):
        return int(math.ceil(value / multiple) * multiple)

    def crop_magic_im(self, image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm, resize_output=True):
        image = image.clone()
        mask = mask.clone()

        if target_w <= 0 or target_h <= 0 or w <= 0 or h <= 0:
            return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

        if padding != 0:
            target_w = self.pad_to_multiple(target_w, padding)
            target_h = self.pad_to_multiple(target_h, padding)

        target_aspect = target_w / target_h
        _, image_h, image_w, _ = image.shape
        context_aspect = w / h

        if context_aspect < target_aspect:
            new_w = int(h * target_aspect)
            new_h = h
            new_x = x - (new_w - w) // 2
            new_y = y

            if new_x < 0:
                shift = -new_x
                if new_x + new_w + shift <= image_w:
                    new_x += shift
                else:
                    new_x = -((new_w - image_w) // 2)
            elif new_x + new_w > image_w:
                overflow = new_x + new_w - image_w
                if new_x - overflow >= 0:
                    new_x -= overflow
                else:
                    new_x = -((new_w - image_w) // 2)
        else:
            new_w = w
            new_h = int(w / target_aspect)
            new_x = x
            new_y = y - (new_h - h) // 2

            if new_y < 0:
                shift = -new_y
                if new_y + new_h + shift <= image_h:
                    new_y += shift
                else:
                    new_y = -((new_h - image_h) // 2)
            elif new_y + new_h > image_h:
                overflow = new_y + new_h - image_h
                if new_y - overflow >= 0:
                    new_y -= overflow
                else:
                    new_y = -((new_h - image_h) // 2)

        if not resize_output:
            if new_w < target_w:
                grow_w = target_w - new_w
                new_x -= grow_w // 2
                new_w = target_w
                if new_x < 0:
                    shift = -new_x
                    if new_x + new_w + shift <= image_w:
                        new_x += shift
                    else:
                        new_x = -((new_w - image_w) // 2)
                elif new_x + new_w > image_w:
                    overflow = new_x + new_w - image_w
                    if new_x - overflow >= 0:
                        new_x -= overflow
                    else:
                        new_x = -((new_w - image_w) // 2)
            if new_h < target_h:
                grow_h = target_h - new_h
                new_y -= grow_h // 2
                new_h = target_h
                if new_y < 0:
                    shift = -new_y
                    if new_y + new_h + shift <= image_h:
                        new_y += shift
                    else:
                        new_y = -((new_h - image_h) // 2)
                elif new_y + new_h > image_h:
                    overflow = new_y + new_h - image_h
                    if new_y - overflow >= 0:
                        new_y -= overflow
                    else:
                        new_y = -((new_h - image_h) // 2)

        up_padding = max(0, -new_y)
        left_padding = max(0, -new_x)
        down_padding = max(0, (new_y + new_h) - image_h)
        right_padding = max(0, (new_x + new_w) - image_w)

        expanded_h = image_h + up_padding + down_padding
        expanded_w = image_w + left_padding + right_padding

        expanded_image = torch.zeros((image.shape[0], expanded_h, expanded_w, image.shape[3]), device=image.device, dtype=image.dtype)
        expanded_mask = torch.ones((mask.shape[0], expanded_h, expanded_w), device=mask.device, dtype=mask.dtype)

        image_chw = image.permute(0, 3, 1, 2)
        expanded_chw = expanded_image.permute(0, 3, 1, 2)
        expanded_chw[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image_chw

        if up_padding > 0:
            expanded_chw[:, :, :up_padding, left_padding:left_padding + image_w] = expanded_chw[:, :, up_padding:up_padding + 1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
        if down_padding > 0:
            expanded_chw[:, :, -down_padding:, left_padding:left_padding + image_w] = expanded_chw[:, :, up_padding + image_h - 1:up_padding + image_h, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
        if left_padding > 0:
            expanded_chw[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_chw[:, :, up_padding:up_padding + image_h, left_padding:left_padding + 1].repeat(1, 1, 1, left_padding)
        if right_padding > 0:
            expanded_chw[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_chw[:, :, up_padding:up_padding + image_h, -right_padding - 1:-right_padding].repeat(1, 1, 1, right_padding)

        expanded_image = expanded_chw.permute(0, 2, 3, 1)
        expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

        cto_x = left_padding
        cto_y = up_padding
        cto_w = image_w
        cto_h = image_h
        canvas_image = expanded_image
        canvas_mask = expanded_mask

        ctc_x = new_x + left_padding
        ctc_y = new_y + up_padding
        ctc_w = new_w
        ctc_h = new_h

        cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        if resize_output:
            if target_w > ctc_w or target_h > ctc_h:
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
                cropped_mask = self.rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
            else:
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
                cropped_mask = self.rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

        return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h

    def stitch_magic_im(self, canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
        canvas_image = canvas_image.clone()
        inpainted_image = inpainted_image.clone()
        mask = mask.clone()

        _, h, w, _ = inpainted_image.shape
        if ctc_w > w or ctc_h > h:
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
            resized_mask = self.rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
        else:
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
            resized_mask = self.rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

        resized_mask = resized_mask.clamp(0.0, 1.0).unsqueeze(-1)
        canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
        canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended
        return canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]


class ArtifyInpaintCrop:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "image": "Input image to crop for inpainting.",
            "downscale_algorithm": "Algorithm used when stitched content must be downscaled to fit the crop window.",
            "upscale_algorithm": "Algorithm used when stitched content must be upscaled to fit the crop window.",
            "mask_fill_holes": "Fill enclosed holes in the mask before computing context.",
            "mask_expand_pixels": "Expand mask edges by N pixels before context detection.",
            "mask_invert": "Invert mask so unmasked regions become the inpaint target.",
            "mask_blend_pixels": "Blend radius (in pixels) used for stitching transitions.",
            "mask_hipass_filter": "Ignore mask values below this threshold.",
            "context_from_mask_extend_factor": "Grow mask-derived context area by this factor.",
            "min_context_megapixels": "Minimum context area in megapixels. Grows context while preserving aspect ratio.",
            "device_mode": "Run crop math on CPU (safe) or GPU (faster).",
            "mask": "Primary inpainting mask.",
            "optional_context_mask": "Extra mask area to force-include as context.",
        }
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear", "tooltip": tooltips["downscale_algorithm"]}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic", "tooltip": tooltips["upscale_algorithm"]}),
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": tooltips["mask_fill_holes"]}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": tooltips["mask_expand_pixels"]}),
                "mask_invert": ("BOOLEAN", {"default": False, "tooltip": tooltips["mask_invert"]}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1, "tooltip": tooltips["mask_blend_pixels"]}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tooltips["mask_hipass_filter"]}),
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": tooltips["context_from_mask_extend_factor"]}),
                "min_context_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01, "tooltip": tooltips["min_context_megapixels"]}),
                "device_mode": (["cpu (compatible)", "gpu (much faster)"], {"default": "gpu (much faster)", "tooltip": tooltips["device_mode"]}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
                "optional_context_mask": ("MASK", {"tooltip": tooltips["optional_context_mask"]}),
            },
        }

    CATEGORY = "Artify/Inpaint"
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
    FUNCTION = "inpaint_crop"

    def _enforce_min_context_megapixels(self, x, y, w, h, min_megapixels, image_w, image_h):
        if min_megapixels <= 0.0 or w <= 0 or h <= 0:
            return x, y, w, h

        current_area = float(w * h)
        target_area = float(min_megapixels) * 1024.0 * 1024.0
        if current_area >= target_area:
            return x, y, w, h

        scale_needed = math.sqrt(target_area / current_area)
        scale_max = min(float(image_w) / float(w), float(image_h) / float(h))
        scale = min(scale_needed, scale_max)
        if scale <= 1.0:
            return x, y, w, h

        target_w = min(image_w, max(1, int(round(w * scale))))
        target_h = min(image_h, max(1, int(round(h * scale))))

        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)

        new_x = int(round(center_x - target_w / 2.0))
        new_y = int(round(center_y - target_h / 2.0))

        max_x = max(0, image_w - target_w)
        max_y = max(0, image_h - target_h)
        new_x = min(max(new_x, 0), max_x)
        new_y = min(max(new_y, 0), max_y)
        return new_x, new_y, int(target_w), int(target_h)

    def inpaint_crop(
        self,
        image,
        downscale_algorithm,
        upscale_algorithm,
        mask_fill_holes,
        mask_expand_pixels,
        mask_invert,
        mask_blend_pixels,
        mask_hipass_filter,
        context_from_mask_extend_factor,
        min_context_megapixels,
        device_mode,
        mask=None,
        optional_context_mask=None,
    ):
        image = image.clone()
        if mask is not None:
            mask = mask.clone()
        if optional_context_mask is not None:
            optional_context_mask = optional_context_mask.clone()

        if device_mode == "gpu (much faster)":
            device = comfy.model_management.get_torch_device()
            image = image.to(device)
            if mask is not None:
                mask = mask.to(device)
            if optional_context_mask is not None:
                optional_context_mask = optional_context_mask.to(device)

        ops = InpaintOps()

        if mask is not None and (image.shape[0] == 1 or mask.shape[0] in (1, image.shape[0])):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] in (1, image.shape[0])):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros(
                        (optional_context_mask.shape[0], image.shape[1], image.shape[2]),
                        device=image.device,
                        dtype=image.dtype,
                    )

        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0])
        if mask.shape[0] > 1 and image.shape[0] == 1:
            image = image.expand(mask.shape[0], -1, -1, -1).clone()
        if image.shape[0] > 1 and mask.shape[0] == 1:
            mask = mask.expand(image.shape[0], -1, -1).clone()

        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0])
        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()

        assert image.ndimension() == 4
        assert mask.ndimension() == 3
        assert optional_context_mask.ndimension() == 3
        assert mask.shape[1:] == image.shape[1:3]
        assert optional_context_mask.shape[1:] == image.shape[1:3]
        assert mask.shape[0] == image.shape[0]
        assert optional_context_mask.shape[0] == image.shape[0]

        stitcher = {
            "downscale_algorithm": downscale_algorithm,
            "upscale_algorithm": upscale_algorithm,
            "blend_pixels": mask_blend_pixels,
            "canvas_to_orig_x": [],
            "canvas_to_orig_y": [],
            "canvas_to_orig_w": [],
            "canvas_to_orig_h": [],
            "canvas_image": [],
            "cropped_to_canvas_x": [],
            "cropped_to_canvas_y": [],
            "cropped_to_canvas_w": [],
            "cropped_to_canvas_h": [],
            "cropped_mask_for_blend": [],
            "device_mode": device_mode,
        }
        result_image = []
        result_mask = []

        for i in range(image.shape[0]):
            sub_image = image[i:i + 1]
            sub_mask = mask[i:i + 1]
            sub_opt_mask = optional_context_mask[i:i + 1]

            if mask_fill_holes:
                sub_mask = ops.fillholes_iterative_hipass_fill_m(sub_mask)
            if mask_expand_pixels > 0:
                sub_mask = ops.expand_m(sub_mask, mask_expand_pixels)
            if mask_invert:
                sub_mask = ops.invert_m(sub_mask)
            if mask_blend_pixels > 0:
                sub_mask = ops.expand_m(sub_mask, mask_blend_pixels)
                sub_mask = ops.blur_m(sub_mask, mask_blend_pixels * 0.5)
            if mask_hipass_filter >= 0.01:
                sub_mask = ops.hipassfilter_m(sub_mask, mask_hipass_filter)
                sub_opt_mask = ops.hipassfilter_m(sub_opt_mask, mask_hipass_filter)

            bx, by, bw, bh = ops.batched_findcontextarea_m(sub_mask)
            if bx[0] == -1:
                bx[0], by[0], bw[0], bh[0] = 0, 0, sub_image.shape[2], sub_image.shape[1]

            if context_from_mask_extend_factor >= 1.01:
                bx, by, bw, bh = ops.batched_growcontextarea_m(sub_mask, bx, by, bw, bh, context_from_mask_extend_factor)

            bx, by, bw, bh = ops.batched_combinecontextmask_m(bx, by, bw, bh, sub_opt_mask)
            if bx[0] == -1:
                bx[0], by[0], bw[0], bh[0] = 0, 0, sub_image.shape[2], sub_image.shape[1]

            cur_x, cur_y, cur_w, cur_h = bx[0].item(), by[0].item(), bw[0].item(), bh[0].item()
            cur_x, cur_y, cur_w, cur_h = self._enforce_min_context_megapixels(
                cur_x,
                cur_y,
                cur_w,
                cur_h,
                min_context_megapixels,
                sub_image.shape[2],
                sub_image.shape[1],
            )

            # No forced target resizing in Artify version.
            cto_x, cto_y = 0, 0
            cto_w, cto_h = sub_image.shape[2], sub_image.shape[1]
            canvas_image = sub_image
            ctc_x, ctc_y, ctc_w, ctc_h = cur_x, cur_y, cur_w, cur_h
            cropped_image = sub_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
            cropped_mask = sub_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

            mask_for_blend = cropped_mask
            if mask_blend_pixels > 0:
                mask_for_blend = ops.blur_m(mask_for_blend, mask_blend_pixels * 0.5)

            stitcher["canvas_to_orig_x"].append(cto_x)
            stitcher["canvas_to_orig_y"].append(cto_y)
            stitcher["canvas_to_orig_w"].append(cto_w)
            stitcher["canvas_to_orig_h"].append(cto_h)
            stitcher["canvas_image"].append(canvas_image.cpu())
            stitcher["cropped_to_canvas_x"].append(ctc_x)
            stitcher["cropped_to_canvas_y"].append(ctc_y)
            stitcher["cropped_to_canvas_w"].append(ctc_w)
            stitcher["cropped_to_canvas_h"].append(ctc_h)
            stitcher["cropped_mask_for_blend"].append(mask_for_blend.cpu())

            result_image.append(cropped_image.squeeze(0).cpu())
            result_mask.append(cropped_mask.squeeze(0).cpu())

        first_h, first_w = result_image[0].shape[0], result_image[0].shape[1]
        for idx, img in enumerate(result_image[1:], start=1):
            if img.shape[0] != first_h or img.shape[1] != first_w:
                raise ValueError(
                    "ArtifyInpaintCrop batch outputs have different sizes after crop "
                    f"(item 0: {first_w}x{first_h}, item {idx}: {img.shape[1]}x{img.shape[0]}). "
                    "Process items individually or ensure masks produce same crop size."
                )

        return stitcher, torch.stack(result_image, dim=0), torch.stack(result_mask, dim=0)


class ArtifyInpaintStitch:
    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "stitcher": "Metadata output from Inpaint Crop (Artify).",
            "inpainted_image": "Inpaint result to stitch back into the original image.",
        }
        return {
            "required": {
                "stitcher": ("STITCHER", {"tooltip": tooltips["stitcher"]}),
                "inpainted_image": ("IMAGE", {"tooltip": tooltips["inpainted_image"]}),
            }
        }

    CATEGORY = "Artify/Inpaint"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint_stitch"

    def inpaint_stitch(self, stitcher, inpainted_image):
        inpainted_image = inpainted_image.clone()
        ops = InpaintOps()
        results = []

        device_mode = stitcher.get("device_mode", "cpu (compatible)")
        if device_mode == "gpu (much faster)":
            device = comfy.model_management.get_torch_device()
            inpainted_image = inpainted_image.to(device)
        else:
            device = torch.device("cpu")

        for key in ("canvas_image", "cropped_mask_for_blend"):
            if key in stitcher:
                stitcher[key] = [t.to(device) if torch.is_tensor(t) else t for t in stitcher[key]]

        batch_size = inpainted_image.shape[0]
        assert len(stitcher["cropped_to_canvas_x"]) in (1, batch_size), "Stitch batch size does not match image batch size."
        override = len(stitcher["cropped_to_canvas_x"]) == 1 and batch_size > 1

        for i in range(batch_size):
            one_image = inpainted_image[i:i + 1]
            one_stitcher = {}
            for key in ("downscale_algorithm", "upscale_algorithm", "blend_pixels"):
                one_stitcher[key] = stitcher[key]
            for key in (
                "canvas_to_orig_x",
                "canvas_to_orig_y",
                "canvas_to_orig_w",
                "canvas_to_orig_h",
                "canvas_image",
                "cropped_to_canvas_x",
                "cropped_to_canvas_y",
                "cropped_to_canvas_w",
                "cropped_to_canvas_h",
                "cropped_mask_for_blend",
            ):
                one_stitcher[key] = stitcher[key][0] if override else stitcher[key][i]

            out = ops.stitch_magic_im(
                one_stitcher["canvas_image"],
                one_image,
                one_stitcher["cropped_mask_for_blend"],
                one_stitcher["cropped_to_canvas_x"],
                one_stitcher["cropped_to_canvas_y"],
                one_stitcher["cropped_to_canvas_w"],
                one_stitcher["cropped_to_canvas_h"],
                one_stitcher["canvas_to_orig_x"],
                one_stitcher["canvas_to_orig_y"],
                one_stitcher["canvas_to_orig_w"],
                one_stitcher["canvas_to_orig_h"],
                one_stitcher["downscale_algorithm"],
                one_stitcher["upscale_algorithm"],
            )
            results.append(out.squeeze(0))

        return (torch.stack(results, dim=0).cpu(),)


NODE_CLASS_MAPPINGS = {
    "ArtifyInpaintCrop": ArtifyInpaintCrop,
    "ArtifyInpaintStitch": ArtifyInpaintStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtifyInpaintCrop": "Inpaint Crop (Artify)",
    "ArtifyInpaintStitch": "Inpaint Stitch (Artify)",
}
