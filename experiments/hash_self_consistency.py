import sys
import numpy as np
from PIL import Image

from models.neuralhash import NeuralHash
from utils.hashing import load_hash_matrix, compute_hash

def main(img_path: str):
    seed = load_hash_matrix()
    nh = NeuralHash()

    img = Image.open(img_path).convert("RGB")

    # NeuralHash() should output logits when called on an image (depends on your implementation).
    # If NeuralHash expects PIL image, this works. If it expects tensor, you'll need the model's preprocess.
    logits1 = nh(img)
    logits2 = nh(img)

    # Get binary bit-vectors (0/1) so we can compute Hamming distance
    h1 = compute_hash(logits1, seed=seed, binary=True, as_string=False)
    h2 = compute_hash(logits2, seed=seed, binary=True, as_string=False)

    # Convert to numpy for distance
    h1_np = h1.detach().cpu().numpy() if hasattr(h1, "detach") else np.array(h1)
    h2_np = h2.detach().cpu().numpy() if hasattr(h2, "detach") else np.array(h2)

    dist = int(np.sum(h1_np != h2_np))

    # Also print hex (compute_hash can do it directly)
    hex1 = compute_hash(logits1, seed=seed, binary=False)
    hex2 = compute_hash(logits2, seed=seed, binary=False)

    print("img:", img_path)
    print("hex1:", hex1)
    print("hex2:", hex2)
    print("hamming:", dist)

if __name__ == "__main__":
    main(sys.argv[1])