# Changelog

## 0.2.5 - 2026-02-12

- Fixed stitch drift introduced after removing user-facing output padding/resize controls.
- Added internal crop-size grid alignment to reduce latent-size rounding mismatches.
- Added safer stitch handling for small output-size mismatches (top-left anchored pad path).

## 0.2.4 - 2026-02-12

- Fixed potential 1-pixel crop/stitch drift by switching to integer-centered context growth math.
- Avoided unnecessary resize in stitch when inpainted image already matches crop size.

## 0.2.3 - 2026-02-12

- Fixed `min_context_megapixels` behavior for values above `1.0` when image dimensions allow larger contexts.
- Changed minimum-context megapixel growth to preserve input image aspect ratio (instead of variable context ratio).

## 0.2.2 - 2026-02-12

- Replaced `min_context_width` / `min_context_height` with `min_context_megapixels`.
- Context minimum is now enforced by area growth while preserving the current aspect ratio.

## 0.2.1 - 2026-02-12

- Removed `output_resize_to_target_size` from `Inpaint Crop (Artify)`.
- Removed dependent target-size controls from `Inpaint Crop (Artify)`.
- Added tooltips to all Artify node input settings.

## 0.2.0 - 2026-02-12

- Added `Inpaint Crop (Artify)` node (`ArtifyInpaintCrop`).
- Added `Inpaint Stitch (Artify)` node (`ArtifyInpaintStitch`).
- Added minimum context window controls to crop:
  - `min_context_width`
  - `min_context_height`
- Removed pre-resize and outpainting controls from the Artify crop workflow.

## 0.1.0 - 2026-02-12

- Initial release.
- Added `Image Resize (Artify)` node (`ArtifyImageResize`).
- Implemented single-pass `scale_by` behavior (no squared scaling).
- Added configurable `divisible_mode` (`floor`, `ceil`, `nearest`).
- Added support for `stretch`, `crop`, `pad`, `pad_edge`, `pad_edge_pixel`, and `pillarbox_blur` output modes.
