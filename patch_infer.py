import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from patchify import patchify
import torch

from utils import get_transforms, device
from model_cnn import ConvolutionalNetwork

def scan_large_image(img_path, patch_hw=(60,60), step=20, model_path="CNNModel.pt", positive_class=1):
    # load model
    d = device()
    model = ConvolutionalNetwork().to(d)
    model.load_state_dict(torch.load(model_path, map_location=d))
    model.eval()

    # image and patchify
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img)
    H, W, C = arr.shape
    ph, pw = patch_hw
    patches = patchify(arr, (ph, pw, C), step=step)

    _, test_t = get_transforms()
    hits = []

    for iy in range(patches.shape[0]):
        for ix in range(patches.shape[1]):
            patch = Image.fromarray(patches[iy, ix, 0])
            tens = test_t(patch).unsqueeze(0).to(d)
            with torch.no_grad():
                pred = model(tens).argmax(dim=1).item()
            if pred == positive_class:
                x0 = ix * step
                y0 = iy * step
                hits.append((x0, y0, pw, ph))

    # visualize
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(img)
    for (x0,y0,w,h) in hits:
        ax.add_patch(Rectangle((x0, y0), w, h, edgecolor="red", facecolor="none", lw=1))
    ax.set_axis_off()
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--patch_h", type=int, default=60)
    ap.add_argument("--patch_w", type=int, default=60)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--model_path", default="CNNModel.pt")
    args = ap.parse_args()
    fig = scan_large_image(args.image, (args.patch_h, args.patch_w), args.step, args.model_path)
    fig.savefig("patch_infer_overlay.png", dpi=200)
    print("Saved patch_infer_overlay.png")
