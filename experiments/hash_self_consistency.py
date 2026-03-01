# experiments/hash_self_consistency.py
# Author: Avijit Roy
#
# Purpose:
# - Verify NeuralHash determinism for a given image
# - Uses the *real* trained weights (models/model.pth)
# - Confirms stable hash across separate runs

# PYTHONPATH=. python experiments/hash_self_consistency.py ./inputs/inputs_50/n03000684_400.JPEG
# PYTHONPATH=. python experiments/hash_self_consistency.py ./inputs/inputs_50/n03000684_400.JPEG
# PYTHONPATH=. python experiments/hash_self_consistency.py ./inputs/inputs_50/n03000684_400.JPEG
# Expected: Same hex and hamming distance 0 across runs

import sys
import numpy as np
import torch
from PIL import Image

from models.neuralhash import NeuralHash
from utils.hashing import load_hash_matrix, compute_hash


def pil_to_tensor_rgb(pil_img: Image.Image, size=(96, 128)) -> torch.Tensor:
    # size is (H=96, W=128). PIL resize expects (W,H)
    w, h = size[1], size[0]
    img = pil_img.resize((w, h))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))          # 3,H,W
    return torch.from_numpy(arr).unsqueeze(0)   # 1,3,H,W


def main(img_path: str):
    # Hard-pin CPU for determinism check
    device = torch.device("cpu")
    torch.set_num_threads(1)

    seed = load_hash_matrix()
    nh = NeuralHash().to(device)
    nh.eval()

    # ? Load trained weights (THIS WAS MISSING)
    state = torch.load("./models/model.pth", map_location=device)
    nh.load_state_dict(state)

    img = Image.open(img_path).convert("RGB")
    x = pil_to_tensor_rgb(img).to(device)

    with torch.no_grad():
        logits1 = nh(x)
        logits2 = nh(x)

    # Binary bits (for Hamming)
    h1 = compute_hash(logits1, seed=seed, binary=True, as_string=False)
    h2 = compute_hash(logits2, seed=seed, binary=True, as_string=False)

    h1_np = h1.detach().cpu().numpy()
    h2_np = h2.detach().cpu().numpy()
    dist = int(np.sum(h1_np != h2_np))

    # Hex string
    hex1 = compute_hash(logits1, seed=seed, binary=False)
    hex2 = compute_hash(logits2, seed=seed, binary=False)

    print("device: cpu")
    print("img:", img_path)
    print("hex1:", hex1)
    print("hex2:", hex2)
    print("hamming:", dist)


if __name__ == "__main__":
    main(sys.argv[1])