import sys
import numpy as np
from PIL import Image

from models.neuralhash import NeuralHash
from utils.hashing import load_hash_matrix, compute_hash, to_hex

def main(img_path: str):
    seed = load_hash_matrix()
    nh = NeuralHash()

    img = Image.open(img_path).convert("RGB")

    h1 = compute_hash(nh, img, seed)
    h2 = compute_hash(nh, img, seed)

    # Expect exact match
    dist = np.sum(h1 != h2)

    print("img:", img_path)
    print("hex1:", to_hex(h1))
    print("hex2:", to_hex(h2))
    print("hamming:", int(dist))

if __name__ == "__main__":
    main(sys.argv[1])