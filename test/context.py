import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import voxelproj

if __name__ == "__main__":
    print("Testing import")
    print(voxelproj.__name__)
