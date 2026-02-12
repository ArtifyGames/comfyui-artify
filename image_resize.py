import math
import re

import torch
import torch.nn.functional as F
from comfy import model_management
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION
from torchvision.transforms.functional import gaussian_blur


class ArtifyImageResize:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        tooltips = {
            "image": "Input image batch (BHWC).",
            "custom_width": "Target width. If only one side is set, the other preserves aspect ratio.",
            "custom_height": "Target height. If only one side is set, the other preserves aspect ratio.",
            "megapixels": "If > 0, computes size from megapixels and aspect ratio (highest priority).",
            "scale_by": "Scale factor applied exactly once (ignored when megapixels > 0).",
            "resize_mode": "How resize_value is interpreted: longest_side or shortest_side.",
            "resize_value": "Target side value used with resize_mode.",
            "upscale_method": "Interpolation/upscale kernel.",
            "device": "CPU is safest; GPU can be faster (Lanczos requires CPU).",
            "divisible_by": "Snap final width/height to a multiple of this value.",
            "divisible_mode": "Rounding mode for divisible snapping.",
            "output_mode": "Final fit behavior: stretch, crop, or padding variants.",
            "crop_position": "Anchor for crop and asymmetric padding placement.",
            "pad_color": "Padding color for pad mode (#RRGGBB or #RGB).",
            "mask": "Optional mask resized with the same geometry as image.",
        }
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": tooltips["image"]}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": tooltips["custom_width"]}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": tooltips["custom_height"]}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.01, "tooltip": tooltips["megapixels"]}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": tooltips["scale_by"]}),
                "resize_mode": (["longest_side", "shortest_side"], {"default": "longest_side", "tooltip": tooltips["resize_mode"]}),
                "resize_value": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": tooltips["resize_value"]}),
                "upscale_method": (cls.upscale_methods, {"default": "lanczos", "tooltip": tooltips["upscale_method"]}),
                "device": (["cpu", "gpu"], {"default": "cpu", "tooltip": tooltips["device"]}),
                "divisible_by": ("INT", {"default": 2, "min": 1, "max": 512, "step": 1, "tooltip": tooltips["divisible_by"]}),
                "divisible_mode": (["floor", "ceil", "nearest"], {"default": "floor", "tooltip": tooltips["divisible_mode"]}),
                "output_mode": (["stretch", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"], {"default": "stretch", "tooltip": tooltips["output_mode"]}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center", "tooltip": tooltips["crop_position"]}),
                "pad_color": ("STRING", {"default": "#FFFFFF", "tooltip": tooltips["pad_color"]}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": tooltips["mask"]}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "WIDTH", "HEIGHT")
    FUNCTION = "resize"
    CATEGORY = "Artify/Image"

    def _normalize_hex_color(self, value):
        color = value.strip()
        if not color.startswith("#"):
            color = f"#{color}"
        if re.fullmatch(r"#([0-9A-Fa-f]{3})", color):
            color = "#" + "".join(ch * 2 for ch in color[1:])
        if not re.fullmatch(r"#([0-9A-Fa-f]{6})", color):
            raise ValueError(f"Invalid color: {value}. Use #RRGGBB or #RGB.")
        return color.upper()

    def _hex_to_rgb_tensor(self, hex_color, dtype, device):
        color = self._normalize_hex_color(hex_color).lstrip("#")
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
        return torch.tensor([r, g, b], dtype=dtype, device=device)

    def _resize_image(self, image, width, height, method):
        return common_upscale(image.movedim(-1, 1), width, height, method, crop="disabled").movedim(1, -1)

    def _resize_mask(self, mask, width, height):
        if mask is None:
            return None
        return F.interpolate(mask.unsqueeze(1), size=(height, width), mode="bicubic", align_corners=False).squeeze(1)

    def _compute_target_dimensions(
        self,
        orig_width,
        orig_height,
        custom_width,
        custom_height,
        megapixels,
        scale_by,
        resize_mode,
        resize_value,
    ):
        width = orig_width
        height = orig_height

        if megapixels > 0:
            aspect_ratio = orig_width / orig_height
            target_pixels = int(megapixels * 1024 * 1024)
            height = max(1, int(math.sqrt(target_pixels / aspect_ratio)))
            width = max(1, int(round(aspect_ratio * height)))
        elif resize_value > 0:
            if resize_mode == "longest_side":
                ratio = resize_value / max(orig_width, orig_height)
            else:
                ratio = resize_value / min(orig_width, orig_height)
            width = max(1, int(round(orig_width * ratio)))
            height = max(1, int(round(orig_height * ratio)))
        elif custom_width > 0 or custom_height > 0:
            if custom_width > 0 and custom_height == 0:
                width = custom_width
                height = max(1, int(round(orig_height * (width / orig_width))))
            elif custom_height > 0 and custom_width == 0:
                height = custom_height
                width = max(1, int(round(orig_width * (height / orig_height))))
            else:
                width = custom_width
                height = custom_height

        if megapixels <= 0 and scale_by != 1.0:
            width = max(1, int(round(width * scale_by)))
            height = max(1, int(round(height * scale_by)))

        if width > MAX_RESOLUTION or height > MAX_RESOLUTION:
            raise ValueError(
                f"Target size {width}x{height} exceeds MAX_RESOLUTION={MAX_RESOLUTION}. "
                "Lower scale or target size."
            )

        return width, height

    def _apply_divisible(self, value, divisor, mode):
        if divisor <= 1:
            return value

        lower = value - (value % divisor)
        if lower <= 0:
            lower = value
        upper = lower if lower == value else lower + divisor

        if mode == "ceil":
            return upper
        if mode == "nearest":
            if abs(value - lower) <= abs(upper - value):
                return lower
            return upper
        return lower

    def _fit_inside(self, src_width, src_height, dst_width, dst_height):
        ratio = min(dst_width / src_width, dst_height / src_height)
        fit_width = max(1, min(dst_width, int(round(src_width * ratio))))
        fit_height = max(1, min(dst_height, int(round(src_height * ratio))))
        return fit_width, fit_height

    def _compute_crop_box(self, src_width, src_height, dst_width, dst_height, crop_position):
        src_aspect = src_width / src_height
        dst_aspect = dst_width / dst_height

        if src_aspect > dst_aspect:
            crop_width = max(1, int(round(src_height * dst_aspect)))
            crop_height = src_height
        else:
            crop_width = src_width
            crop_height = max(1, int(round(src_width / dst_aspect)))

        x = (src_width - crop_width) // 2
        y = (src_height - crop_height) // 2

        if crop_position == "top":
            y = 0
        elif crop_position == "bottom":
            y = src_height - crop_height
        elif crop_position == "left":
            x = 0
        elif crop_position == "right":
            x = src_width - crop_width

        return x, y, crop_width, crop_height

    def _compute_padding(self, width, height, dst_width, dst_height, crop_position):
        remaining_w = dst_width - width
        remaining_h = dst_height - height

        if crop_position == "left":
            pad_left = 0
            pad_right = remaining_w
            pad_top = remaining_h // 2
            pad_bottom = remaining_h - pad_top
        elif crop_position == "right":
            pad_left = remaining_w
            pad_right = 0
            pad_top = remaining_h // 2
            pad_bottom = remaining_h - pad_top
        elif crop_position == "top":
            pad_left = remaining_w // 2
            pad_right = remaining_w - pad_left
            pad_top = 0
            pad_bottom = remaining_h
        elif crop_position == "bottom":
            pad_left = remaining_w // 2
            pad_right = remaining_w - pad_left
            pad_top = remaining_h
            pad_bottom = 0
        else:
            pad_left = remaining_w // 2
            pad_right = remaining_w - pad_left
            pad_top = remaining_h // 2
            pad_bottom = remaining_h - pad_top

        return pad_left, pad_right, pad_top, pad_bottom

    def _make_pillarbox_background(self, source_image, dst_width, dst_height):
        _, src_height, src_width, _ = source_image.shape
        fill_scale = max(dst_width / float(src_width), dst_height / float(src_height))
        fill_w = max(1, int(round(src_width * fill_scale)))
        fill_h = max(1, int(round(src_height * fill_scale)))

        bg = self._resize_image(source_image, fill_w, fill_h, "bilinear")
        x0 = max(0, (fill_w - dst_width) // 2)
        y0 = max(0, (fill_h - dst_height) // 2)
        bg = bg[:, y0:y0 + dst_height, x0:x0 + dst_width, :]

        if bg.shape[1] != dst_height or bg.shape[2] != dst_width:
            pad_h = dst_height - bg.shape[1]
            pad_w = dst_width - bg.shape[2]
            pad_top = max(0, pad_h // 2)
            pad_bottom = max(0, pad_h - pad_top)
            pad_left = max(0, pad_w // 2)
            pad_right = max(0, pad_w - pad_left)
            bg = F.pad(bg.movedim(-1, 1), (pad_left, pad_right, pad_top, pad_bottom), mode="replicate").movedim(1, -1)

        sigma = max(1.0, 0.006 * float(min(dst_height, dst_width)))
        kernel_size = int(round(sigma * 2)) * 2 + 1
        bg = gaussian_blur(bg.movedim(-1, 1), kernel_size=[kernel_size, kernel_size], sigma=sigma).movedim(1, -1)
        return torch.clamp(bg * 0.35, 0.0, 1.0)

    def _pad_image(self, image, source_image, dst_width, dst_height, pad_left, pad_right, pad_top, pad_bottom, output_mode, pad_color):
        if output_mode == "pad_edge":
            return F.pad(image.movedim(-1, 1), (pad_left, pad_right, pad_top, pad_bottom), mode="replicate").movedim(1, -1)

        if output_mode == "pad_edge_pixel":
            b, h, w, c = image.shape
            out = torch.zeros((b, dst_height, dst_width, c), dtype=image.dtype, device=image.device)
            out[:, pad_top:pad_top + h, pad_left:pad_left + w, :] = image
            for idx in range(b):
                out[idx, :pad_top, pad_left:pad_left + w, :] = image[idx, 0:1, :, :].repeat(pad_top, 1, 1)
                out[idx, pad_top + h:, pad_left:pad_left + w, :] = image[idx, h - 1:h, :, :].repeat(pad_bottom, 1, 1)
                out[idx, pad_top:pad_top + h, :pad_left, :] = image[idx, :, 0:1, :].repeat(1, pad_left, 1)
                out[idx, pad_top:pad_top + h, pad_left + w:, :] = image[idx, :, w - 1:w, :].repeat(1, pad_right, 1)
                out[idx, :pad_top, :pad_left, :] = image[idx, 0, 0, :]
                out[idx, :pad_top, pad_left + w:, :] = image[idx, 0, w - 1, :]
                out[idx, pad_top + h:, :pad_left, :] = image[idx, h - 1, 0, :]
                out[idx, pad_top + h:, pad_left + w:, :] = image[idx, h - 1, w - 1, :]
            return out

        if output_mode == "pillarbox_blur":
            out = self._make_pillarbox_background(source_image, dst_width, dst_height)
            out[:, pad_top:pad_top + image.shape[1], pad_left:pad_left + image.shape[2], :] = image
            return out

        bg_color = self._hex_to_rgb_tensor(pad_color, image.dtype, image.device)
        out = torch.zeros((image.shape[0], dst_height, dst_width, image.shape[3]), dtype=image.dtype, device=image.device)
        out[:] = bg_color
        out[:, pad_top:pad_top + image.shape[1], pad_left:pad_left + image.shape[2], :] = image
        return out

    def _pad_mask(self, mask, dst_width, dst_height, pad_left, pad_right, pad_top, pad_bottom, output_mode):
        if mask is None:
            return None
        if output_mode in {"pad_edge", "pad_edge_pixel"}:
            return F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        out = torch.zeros((mask.shape[0], dst_height, dst_width), dtype=mask.dtype, device=mask.device)
        out[:, pad_top:pad_top + mask.shape[1], pad_left:pad_left + mask.shape[2]] = mask
        return out

    def resize(
        self,
        image,
        custom_width,
        custom_height,
        megapixels,
        scale_by,
        resize_mode,
        resize_value,
        upscale_method,
        device,
        divisible_by,
        divisible_mode,
        output_mode,
        crop_position,
        pad_color,
        mask=None,
    ):
        _, orig_height, orig_width, _ = image.shape

        target_device = torch.device("cpu")
        if device == "gpu":
            if upscale_method == "lanczos":
                raise ValueError("Lanczos is not supported on GPU mode. Choose another upscaler or CPU.")
            target_device = model_management.get_torch_device()

        work_image = image.to(target_device)
        work_mask = mask.to(target_device) if mask is not None else None
        if work_mask is not None and work_mask.ndim == 2:
            work_mask = work_mask.unsqueeze(0)

        target_width, target_height = self._compute_target_dimensions(
            orig_width,
            orig_height,
            custom_width,
            custom_height,
            megapixels,
            scale_by,
            resize_mode,
            resize_value,
        )

        if divisible_by > 1:
            target_width = self._apply_divisible(target_width, divisible_by, divisible_mode)
            target_height = self._apply_divisible(target_height, divisible_by, divisible_mode)

        target_width = max(1, target_width)
        target_height = max(1, target_height)
        if target_width > MAX_RESOLUTION or target_height > MAX_RESOLUTION:
            raise ValueError(
                f"Final size {target_width}x{target_height} exceeds MAX_RESOLUTION={MAX_RESOLUTION}. "
                "Lower scale, divisible settings, or base target size."
            )

        if output_mode == "crop":
            x, y, crop_w, crop_h = self._compute_crop_box(orig_width, orig_height, target_width, target_height, crop_position)
            work_image = work_image[:, y:y + crop_h, x:x + crop_w, :]
            if work_mask is not None:
                work_mask = work_mask[:, y:y + crop_h, x:x + crop_w]
            out_image = self._resize_image(work_image, target_width, target_height, upscale_method)
            out_mask = self._resize_mask(work_mask, target_width, target_height)
        elif output_mode == "stretch":
            out_image = self._resize_image(work_image, target_width, target_height, upscale_method)
            out_mask = self._resize_mask(work_mask, target_width, target_height)
        else:
            fit_w, fit_h = self._fit_inside(orig_width, orig_height, target_width, target_height)
            resized_image = self._resize_image(work_image, fit_w, fit_h, upscale_method)
            resized_mask = self._resize_mask(work_mask, fit_w, fit_h)
            pad_left, pad_right, pad_top, pad_bottom = self._compute_padding(fit_w, fit_h, target_width, target_height, crop_position)
            out_image = self._pad_image(
                resized_image,
                work_image,
                target_width,
                target_height,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                output_mode,
                pad_color,
            )
            out_mask = self._pad_mask(
                resized_mask,
                target_width,
                target_height,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                output_mode,
            )

        if out_mask is None:
            out_mask = torch.zeros((out_image.shape[0], target_height, target_width), dtype=out_image.dtype, device=out_image.device)
        else:
            out_mask = torch.clamp(out_mask, 0.0, 1.0)

        return out_image.cpu(), out_mask.cpu(), target_width, target_height


NODE_CLASS_MAPPINGS = {
    "ArtifyImageResize": ArtifyImageResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArtifyImageResize": "Image Resize (Artify)",
}
