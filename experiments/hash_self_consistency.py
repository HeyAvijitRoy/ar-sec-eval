import sys
import numpy as np
import torch
from PIL import Image

from models.neuralhash import NeuralHash
from utils.hashing import load_hash_matrix, compute_hash

def pil_to_tensor_rgb(pil_img: Image.Image, size=(96, 128)) -> torch.Tensor:
    """
    Convert PIL RGB image to a torch tensor shaped [1, 3, H, W], float32 in [0,1].
    NeuralHash seed is 128x96, so default resize is (H=96, W=128).
    """
    # PIL expects size as (W, H)
    w, h = size[1], size[0]
    img = pil_img.resize((w, h))
    arr = np.array(img, dtype=np.float32) / 255.0          # H,W,3 in [0,1]
    arr = np.transpose(arr, (2, 0, 1))                     # 3,H,W
    t = torch.from_numpy(arr).unsqueeze(0)                 # 1,3,H,W
    return t

def main(img_path: str):
    seed = load_hash_matrix()  # numpy [96,128]
    nh = NeuralHash()
    nh.eval()

    img = Image.open(img_path).convert("RGB")
    x = pil_to_tensor_rgb(img, size=(96, 128))  # 1,3,96,128

    with torch.no_grad():
        logits1 = nh(x)
        logits2 = nh(x)

    # Binary bit vectors for Hamming
    h1 = compute_hash(logits1, seed=seed, binary=True, as_string=False)
    h2 = compute_hash(logits2, seed=seed, binary=True, as_string=False)

    h1_np = h1.detach().cpu().numpy()
    h2_np = h2.detach().cpu().numpy()
    dist = int(np.sum(h1_np != h2_np))

    hex1 = compute_hash(logits1, seed=seed, binary=False)
    hex2 = compute_hash(logits2, seed=seed, binary=False)

    print("img:", img_path)
    print("hex1:", hex1)
    print("hex2:", hex2)
    print("hamming:", dist)

if __name__ == "__main__":
    main(sys.argv[1])