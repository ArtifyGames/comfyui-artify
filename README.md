# ComfyUI Artify

ComfyUI Artify is a lightweight custom node pack focused on image utilities.

## Included nodes

- `Image Resize (Artify)`
  - Node id: `ArtifyImageResize`
  - Category: `Artify/Image`

## Why this node exists

This resize node is designed to avoid the common `scale_by` pitfall where scale
is effectively applied twice. Here, `scale_by` is applied exactly once.

## Resize behavior

Target dimension priority:

1. `megapixels` (if `> 0`)
2. `resize_value` + `resize_mode` (`longest_side` or `shortest_side`)
3. `custom_width` / `custom_height`
4. original image size

Then:

- `scale_by` is applied once (skipped when `megapixels > 0`)
- `divisible_by` snaps final dimensions using `divisible_mode`:
  - `floor`
  - `ceil`
  - `nearest`

Output modes:

- `stretch`
- `crop`
- `pad`
- `pad_edge`
- `pad_edge_pixel`
- `pillarbox_blur`

Outputs:

- `IMAGE`
- `MASK` (if no input mask is provided, a zero mask is returned)
- `WIDTH`
- `HEIGHT`

## Install

### Method 1: ComfyUI custom_nodes folder

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/comfyui-artify.git
```

If you add optional dependencies later:

```bash
python -m pip install -r requirements.txt
```

Restart ComfyUI and refresh the browser page.

## Quick test

1. Add `Image Resize (Artify)` to a workflow.
2. Feed a `1328x1328` image.
3. Set:
   - `custom_width=0`
   - `custom_height=0`
   - `megapixels=0`
   - `resize_value=0`
   - `scale_by=1.25`
   - `divisible_by=1`
4. Expected output: `1660x1660`.

## License

See `LICENSE`.
