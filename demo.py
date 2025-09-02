# Our libs
from networks.transforms import trimap_transform, normalise_image
from networks.models import build_model
from dataloader import PredDataset

# System libs
import os
import argparse

# External libs
import cv2
import numpy as np
import torch
import time





def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    """Scales inputs to multiple of 8, works for 2D or multi-channel arrays."""
    print(x.shape)
    if x.ndim == 2:  # Grayscale (H,W)
        h, w = x.shape
    elif x.ndim == 3:  # Multi-channel (H,W,C)
        h, w, _ = x.shape
    else:
        raise ValueError(f"Unexpected input shape for scale_input: {x.shape}")

    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)

    # Resize → keep channel dimension intact
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)

    if x.ndim == 3 and x.shape[2] > 1 and x_scale.ndim == 2:
        # Sometimes cv2 flattens incorrectly, restore channel dim
        x_scale = x_scale.reshape(h1, w1, -1)

    return x_scale


def np_to_torch(x: np.ndarray, permute: bool = True) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If grayscale (H,W), add channel
    print(x.shape)
    if x.ndim == 2:
        x = x[:, :, None]  # (H,W,1)

    if permute:
        # (H,W,C) → (C,H,W)
        x = np.transpose(x, (2, 0, 1))

    # Always add batch dimension → (1,C,H,W)
    return torch.from_numpy(x).unsqueeze(0).float().to(device)


def predict_fba_folder(model, args):
    save_dir = args.output_dir

    dataset_test = PredDataset(args.image_dir, args.trimap_dir)

    gen = iter(dataset_test)
    for item_dict in gen:
        image_np = item_dict['image']
        trimap_np = item_dict['trimap']

        st = time.time()
        fg, bg, alpha = pred(image_np, trimap_np, model)
        print("Time taken for prediction: ", time.time() - st)
        cv2.imwrite(os.path.join(
            save_dir, item_dict['name'][:-4] + '_fg.png'), fg[:, :, ::-1] * 255)
        cv2.imwrite(os.path.join(
            save_dir, item_dict['name'][:-4] + '_bg.png'), bg[:, :, ::-1] * 255)
        cv2.imwrite(os.path.join(
            save_dir, item_dict['name'][:-4] + '_alpha.png'), alpha * 255)


def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict alpha, foreground, and background using FBA model.
    """
    h, w = trimap_np.shape[:2]
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid target size for resize: {(w, h)}, trimap shape={trimap_np.shape}")

    # Scale inputs
    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np), permute=False)
        image_transformed_torch = normalise_image(image_torch.clone())

        output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

        # Convert to numpy (C,H,W) → (H,W,C)
        out_np = output[0].cpu().numpy()
        out_np = np.transpose(out_np, (1, 2, 0)).astype(np.float32)

        # Debug info
        print("DEBUG:", 
              "model_out", output[0].shape, 
              "after transpose", out_np.shape, 
              "target", (w, h))

        # Resize to match trimap size
        out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Split channels
    alpha = out_np[:, :, 0]
    fg = out_np[:, :, 1:4]
    bg = out_np[:, :, 4:7]

    # Apply trimap constraints
    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]

    return fg, bg, alpha



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--weights', default='FBA.pth')
    parser.add_argument('--image_dir', default='./examples/images', help="")
    parser.add_argument('--trimap_dir', default='./examples/trimaps', help="")
    parser.add_argument(
        '--output_dir',
        default='./examples/predictions',
        help="")

    args = parser.parse_args()
    model = build_model(args.weights)
    model.eval().cuda()
    predict_fba_folder(model, args)
