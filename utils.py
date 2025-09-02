import os
import cv2
import numpy as np
import subprocess
import sys
from glob import glob
from PIL import Image
import torch
from scipy.ndimage import grey_dilation, grey_erosion
from networks.isnet import ISNetDIS
from networks.transforms import trimap_transform, normalise_image
from networks.models import build_model
from demo import pred
from config import *
from skimage import io
import torch.nn.functional as F
from torchvision.transforms.functional import normalize








def load_isnet_model(model_path):
    """
    Load MODNet from a checkpoint. Handles plain state_dict or wrapped dict with 'model_state_dict'.
    Strips DDP 'module.' prefixes if present.
    """
    net = ISNetDIS()

    map_loc = None if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=map_loc)

    # Unwrap checkpoint format
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip DDP 'module.' prefix if present
    def strip_module_prefix(sd):
        if not any(k.startswith("module.") for k in sd.keys()):
            return sd
        return {k[len("module."):]: v for k, v in sd.items()}

    state_dict = strip_module_prefix(state_dict)

    # Load with strict=False to be resilient to minor mismatches
    load_result = net.load_state_dict(state_dict, strict=False)
    missing = getattr(load_result, 'missing_keys', [])
    unexpected = getattr(load_result, 'unexpected_keys', [])
    if missing:
        print(f"[load_model] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_model] Unexpected keys: {len(unexpected)}")

    if torch.cuda.is_available():
        net = net.cuda()
        print(f"ISNet Model loaded on GPU: {model_path}")
    else:
        print(f"ISNet Model loaded on CPU: {model_path}")
    
    net.eval()
    return net



def preprocess_image(image, input_size):
    """
    Preprocess image for model inference
    """
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
    
    im_shp = image.shape[0:2]
    im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=input_size, mode="bilinear", align_corners=False).type(torch.uint8)
    image_normalized = torch.divide(im_tensor, 255.0)
    image_normalized = normalize(image_normalized, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    
    if torch.cuda.is_available():
        image_normalized = image_normalized.cuda()
    
    return image_normalized, im_shp

def _preprocess_with_size(image, size_hw):
    """Preprocess with an explicit size (H,W) override."""
    if len(image.shape) < 3:
        image = image[:, :, np.newaxis]
    im_shp = image.shape[0:2]
    im_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=size_hw, mode="bilinear", align_corners=False).type(torch.uint8)
    image_normalized = torch.divide(im_tensor, 255.0)
    image_normalized = normalize(image_normalized, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    if torch.cuda.is_available():
        image_normalized = image_normalized.cuda()
    return image_normalized, im_shp

def run_model_inference(net, image_tensor, original_shape):
    """
    Run model inference and return processed mask
    """
    with torch.no_grad():
        result = net(image_tensor)
        # Normalize output structure across models
        # MODNet: may return a single tensor or a tuple where the last item is the matte tensor
        # ISNetDIS: returns (preds_list, features_list); take the highest-res prediction preds_list[0]
        if isinstance(result, (list, tuple)):
            if len(result) == 2 and isinstance(result[0], (list, tuple)):
                # ISNet-like output
                preds = result[0]
                if not preds:
                    raise RuntimeError("Model returned empty preds list")
                result = preds[0]
            else:
                # Fallback: take the last element
                result = result[-1]
        # Ensure result is NCHW tensor
        if not torch.is_tensor(result):
            raise RuntimeError(f"Unexpected model output type: {type(result)}")
        # result may be (N,1,H,W); upsample back to original shape
        result = torch.squeeze(F.interpolate(result, size=original_shape, mode='bilinear', align_corners=False), 0)
        
        # Normalize result
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        
        # Convert to numpy
        result_np = (result * 255).cpu().data.numpy().astype(np.uint8)
        result_np = np.squeeze(result_np)
        
    return result_np

def _infer_scaled(net, image_np, scale: float, do_hflip: bool):
    """Run a single inference pass at a given scale with optional horizontal flip.
    Returns matte in original resolution (H,W) as float32 in [0,1]."""
    # Determine scaled size from INPUT_SIZE to keep model-friendly dimensions
    tgt_h = max(1, int(round(INPUT_SIZE[0] * scale)))
    tgt_w = max(1, int(round(INPUT_SIZE[1] * scale)))
    img_t, orig_shape = _preprocess_with_size(image_np, (tgt_h, tgt_w))
    if do_hflip:
        img_t = torch.flip(img_t, dims=[3])  # flip width dimension
    matte_np = run_model_inference(net, img_t, orig_shape)  # uint8 (H,W)
    if do_hflip:
        matte_np = np.fliplr(matte_np)
    matte_f = matte_np.astype(np.float32) / 255.0
    return matte_f

def ensemble_matte(net, image_np):
    """Compute ensemble matte using multi-scale and TTA settings.
    Falls back to single pass when disabled."""
    use_ms = bool(MULTI_SCALE_ENABLED)
    use_flip = bool(TTA_HFLIP_ENABLED)
    if not use_ms and not use_flip:
        # Single-pass using configured INPUT_SIZE
        img_t, orig_shape = preprocess_image(image_np, INPUT_SIZE)
        matte_np = run_model_inference(net, img_t, orig_shape)
        return matte_np.astype(np.float32) / 255.0

    scales = MULTI_SCALE_SCALES if use_ms else [1.0]
    parts = []
    for s in scales:
        parts.append(_infer_scaled(net, image_np, s, False))
        if use_flip:
            parts.append(_infer_scaled(net, image_np, s, True))
    stack = np.stack(parts, axis=0)  # (K,H,W)
    matte = np.mean(stack, axis=0)
    matte = np.clip(matte, 0.0, 1.0)
    return matte


def process_single_image(isnet_model, fba_model, image_path, output_folder, threshold=150):
    """
    Process a single image through ISNet + FBA
    """
    # Load image
    try:
        original_image = io.imread(image_path)  # skimage → RGB or RGBA
        if original_image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        # Convert RGBA → RGB if needed
        if original_image.ndim == 3 and original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        elif original_image.ndim == 2:  # grayscale → RGB
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return False

    # Run ISNet inference
    try:
        matte01 = ensemble_matte(isnet_model, original_image).astype(np.float32)  # (H,W), [0,1]
        mask = (matte01 * 255.0).astype(np.uint8)
        print(f"ISNet inference completed for {image_path}")
    except Exception as e:
        print(f"Error running inference on {image_path}: {e}")
        return False

    # Create binary mask + trimap
    binary_mask = create_binary_mask(mask, threshold)
    trimap = binary_to_trimap(binary_mask, fg_width_px=5, bg_width_px=70)
    print(trimap.shape)
    trimap_np = trimap.detach().cpu().numpy()
    # Ensure shape is (H, W, 2)
    if trimap_np.ndim == 4:  # (N,C,H,W)
        trimap_np = np.transpose(trimap_np[0], (1, 2, 0))  # → (H,W,C)
    elif trimap_np.ndim == 3 and trimap_np.shape[0] == 2:  # (C,H,W)
        trimap_np = np.transpose(trimap_np, (1, 2, 0))     # → (H,W,C)

    # ---- FIX: ensure shape is (H,W,2) ----
    if trimap_np.ndim == 3 and trimap_np.shape[-1] == 1:
        bg = (trimap_np[:, :, 0] == 0).astype(np.uint8)
        fg = (trimap_np[:, :, 0] == 1).astype(np.uint8)
        trimap_np = np.stack([bg, fg], axis=-1)

    print("DEBUG fixed trimap_np:", trimap_np.shape)  # should be (H,W,2)
    
    print(trimap_np.shape)


    # Run FBA inference
    try:
        fg, bg, alpha = pred(original_image.astype(np.float32) / 255.0, trimap_np, fba_model)
        print(f"FBA inference completed for {image_path}")
    except Exception as e:
        print(f"Error running inference on {image_path}: {e}")
        return False

    # Generate outputs
    black_bg_image = create_output_with_black_background(original_image, fg, bg, alpha)
    white_bg_image = create_output_with_white_background(original_image, fg, bg, alpha)

    # Save
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    try:
        saved = save_outputs(output_folder, image_name, original_image, binary_mask,
                             black_bg_image, white_bg_image, trimap)
        print(f"  Saved: {list(saved.values())}")
        return True
    except Exception as e:
        print(f"Error saving outputs for {image_path}: {e}")
        return False



def find_all_images(root_path, extensions):
    """
    Recursively find all image files in the root path and all subdirectories
    """
    all_images = []
    for root, dirs, files in os.walk(root_path):
        for ext in extensions:
            # Remove the * from extension for endswith check
            ext_clean = ext.replace("*", "").lower()
            for file in files:
                if file.lower().endswith(ext_clean):
                    all_images.append(os.path.join(root, file))
    return all_images

def check_output_exists(output_folder, image_name):
    """
    Check if output files already exist for given image
    """
    mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
    black_path = os.path.join(output_folder, f"{image_name}_black.png")
    white_path = os.path.join(output_folder, f"{image_name}_white.png")
    original_path = os.path.join(output_folder, f"{image_name}_original.png")
    return all(os.path.exists(p) for p in [mask_path, black_path, white_path, original_path])

def create_binary_mask(mask, threshold=150):
    """
    Create binary mask with static threshold
    """
    binary_mask = (mask >= threshold).astype(np.uint8) * 255
    return binary_mask


def create_output_with_black_background(original_image, fg, bg, alpha):
    """
    Composite original image over a black background using alpha.
    Foreground = original image * alpha
    Background = black
    """
    # Normalize alpha to [0,1]
    if alpha.max() > 1:
        alpha = alpha / 255.0
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, axis=-1)

    black_bg = np.zeros_like(original_image, dtype=np.uint8)

    # Composite: image * alpha + black * (1 - alpha)
    output = (original_image * alpha + black_bg * (1 - alpha)).astype(np.uint8)
    return output


def create_output_with_white_background(original_image, fg, bg, alpha):
    """
    Composite original image over a white background using alpha.
    Foreground = original image * alpha
    Background = white
    """
    # Normalize alpha to [0,1]
    if alpha.max() > 1:
        alpha = alpha / 255.0
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, axis=-1)

    white_bg = np.ones_like(original_image, dtype=np.uint8) * 255

    # Composite: image * alpha + white * (1 - alpha)
    output = (original_image * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    return output



def _ensure_bgr(img_rgb):
    """Ensure image is BGR before saving with cv2 (expects RGB or grayscale)."""
    if img_rgb is None:
        return None
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_rgb


def _to_numpy_uint8(img):
    """
    Ensure input (tensor/float array) is converted to a numpy uint8 image.
    - If torch.Tensor -> convert to cpu numpy
    - If float in [0,1] -> scale to [0,255]
    - Ensure final dtype = uint8
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img

def _to_trimap_uint8(trimap_tensor):
    """
    Convert torch/numpy trimap into grayscale uint8 image:
    background=0 (black), unknown=128 (gray), foreground=255 (white).
    """
    if isinstance(trimap_tensor, torch.Tensor):
        tri = trimap_tensor.detach().cpu().numpy()
    else:
        tri = trimap_tensor

    # Remove batch/channel dims → (H,W)
    tri = np.squeeze(tri)

    # Normalize if needed
    if tri.max() <= 1.0:
        tri = (tri * 255).astype(np.uint8)

    # Ensure clean mapping to {0,128,255}
    tri = np.where(tri < 85, 0, np.where(tri < 170, 128, 255)).astype(np.uint8)

    return tri


def save_outputs(output_folder, image_name, original_image_rgb, binary_mask, black_bg_image_rgb, white_bg_image_rgb, trimap):
    """
    Save original, mask, black/white background, and trimap as PNG files.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Save original (RGB → BGR for OpenCV)
    original_path = os.path.join(output_folder, f"{image_name}_original.png")
    # cv2.imwrite(original_path, _ensure_bgr(original_image_rgb))

    # Save mask
    mask_path = os.path.join(output_folder, f"{image_name}_mask.png")
    # cv2.imwrite(mask_path, binary_mask.astype(np.uint8))

    # Save black/white background images
    black_path = os.path.join(output_folder, f"{image_name}_black.png")
    white_path = os.path.join(output_folder, f"{image_name}_white.png")
    cv2.imwrite(black_path, _ensure_bgr(black_bg_image_rgb))
    cv2.imwrite(white_path, _ensure_bgr(white_bg_image_rgb))

    # Save trimap (converted to grayscale image with 0,128,255)
    tri = _to_trimap_uint8(trimap)
    trimap_path = os.path.join(output_folder, f"{image_name}_trimap.png")
    # cv2.imwrite(trimap_path, tri)

    return {
        "original": original_path,
        "mask": mask_path,
        "black": black_path,
        "white": white_path,
        "trimap": trimap_path,
    }


def check_gsutil_installed():
    """
    Check if gsutil is installed and accessible
    """
    try:
        result = subprocess.run(['gsutil', 'version'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def sync_to_gcp_bucket(local_folder, gs_bucket_path, rsync_options=None):
    """
    Sync local folder to GCP bucket using gsutil rsync
    """
    if not check_gsutil_installed():
        print("ERROR: gsutil is not installed or not accessible.")
        print("Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        return False
    
    if not os.path.exists(local_folder):
        print(f"ERROR: Local folder does not exist: {local_folder}")
        return False
    
    # Ensure bucket path ends with /
    if not gs_bucket_path.endswith('/'):
        gs_bucket_path += '/'
    
    # Build gsutil rsync command with correct ordering
    # Syntax: gsutil [-m] rsync [-r] [-d] [-x <regex>] SRC DST
    cmd = ['gsutil']
    rsync_opts = rsync_options[:] if rsync_options else []
    # Extract global gsutil options (currently only -m is common)
    global_opts = []
    if '-m' in rsync_opts:
        global_opts.append('-m')
        rsync_opts = [opt for opt in rsync_opts if opt != '-m']
    cmd.extend(global_opts)
    cmd.append('rsync')
    # Append rsync-specific options next
    cmd.extend(rsync_opts)
    # Finally, add source and destination
    cmd.extend([local_folder, gs_bucket_path])
    
    print(f"Syncing {local_folder} to {gs_bucket_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the gsutil command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ Successfully synced to {gs_bucket_path}")
            if result.stdout:
                print("STDOUT:", result.stdout)
            return True
        else:
            print(f"✗ Failed to sync to {gs_bucket_path}")
            print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout while syncing to {gs_bucket_path}")
        return False
    except Exception as e:
        print(f"✗ Error during sync: {e}")
        return False

def sync_all_outputs_to_gcp(model_configs, gs_bucket_base_path, rsync_options=None):
    """
    Sync all output folders to GCP bucket
    """
    print(f"\n{'='*60}")
    print("STARTING GCP SYNC")
    print(f"{'='*60}")
    
    success_count = 0
    total_count = len(model_configs)
    
    for output_folder in model_configs.keys():
        if os.path.exists(output_folder):
            # Create bucket path for this output folder
            bucket_path = gs_bucket_base_path + output_folder
            
            success = sync_to_gcp_bucket(output_folder, bucket_path, rsync_options)
            if success:
                success_count += 1
            print()  # Add blank line between syncs
        else:
            print(f"⚠ Skipping {output_folder} - folder does not exist")
            print()
    
    print(f"{'='*60}")
    print(f"GCP SYNC COMPLETED: {success_count}/{total_count} folders synced successfully")
    print(f"{'='*60}")
    
    return success_count == total_count

def sync_base_output_dir_to_gcp(local_base_folder, gs_bucket_base_path, rsync_options=None):
    """Sync the entire base output directory (containing epoch_* folders) to GCP."""
    return sync_to_gcp_bucket(local_base_folder, gs_bucket_base_path, rsync_options)


def binary_to_trimap(mask, fg_width_px: int = 15, bg_width_px: int = 15) -> torch.Tensor:
    """
    Convert binary mask (torch.Tensor or np.ndarray) into a trimap with values {0, 0.5, 1}.
    Unknown band is around the contour with widths defined by erosion/dilation.

    Args:
        mask: torch.Tensor or np.ndarray. Shape {N,1,H,W}, {1,H,W}, or {H,W}.
              Values should be in [0,1].
        fg_width_px: erosion size for sure foreground region.
        bg_width_px: dilation size for sure background region.

    Returns:
        trimap: torch.Tensor {N,1,H,W}, float32 with values in {0, 0.5, 1}.
    """
    # Convert numpy to tensor if needed
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    # Normalize to 4D
    if mask.ndim == 2:   # H,W
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3: # C,H,W or N,H,W
        mask = mask.unsqueeze(0) if mask.shape[0] != 1 else mask.unsqueeze(1)
    elif mask.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    n, c, h, w = mask.shape
    assert c == 1, f"Expected single channel mask, got {c} channels"

    # Convert to binary numpy array
    m_np = (mask[:, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    out = []

    for i in range(n):
        bin_m = m_np[i]

        # Background dilation → sure background
        dil = grey_dilation(bin_m, size=(bg_width_px, bg_width_px))
        # Foreground erosion → sure foreground
        ero = grey_erosion(bin_m, size=(fg_width_px, fg_width_px))

        tri = np.full_like(bin_m, 0.5, dtype=np.float32)  # start unknown
        tri[dil == 0] = 0.0  # sure background
        tri[ero == 1] = 1.0  # sure foreground

        out.append(tri)

    tri_np = np.stack(out, axis=0)  # N,H,W
    tri_t = torch.from_numpy(tri_np).unsqueeze(1)  # N,1,H,W
    return tri_t.to(mask.device, dtype=torch.float32)


def fba_pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    ''' Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    '''
    h, w = trimap_np.shape[:2]
    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(
            trimap_transform(trimap_scale_np), permute=False)
        image_transformed_torch = normalise_image(
            image_torch.clone())

        output = model(
            image_torch,
            trimap_torch,
            image_transformed_torch,
            trimap_transformed_torch)
        out = output[0].detach().cpu().numpy()  # (C,H,W)
        out = np.transpose(out, (1, 2, 0))      # (H,W,C)

        # Ensure it's float32 for OpenCV
        out = out.astype(np.float32)

        # Check dimensions
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid resize dimensions: {(w,h)}")

        out_resized = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)


    alpha = out_resized[:, :, 0]
    fg = out_resized[:, :, 1:4]
    bg = out_resized[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]

    return fg, bg, alpha