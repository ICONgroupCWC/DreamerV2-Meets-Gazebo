# import torch

# def patches_to_image(patches):
#     B = patches.shape[0]
#     patches = patches.view(B, 14, 14, 3, 6, 6)
#     patches = patches.permute(0, 3, 1, 4, 2, 5)
#     return patches.reshape(B, 3, 84, 84)

import torch

def patches_to_image(patches, patch_size=6, image_size=84):
    B, N, D = patches.shape
    H = W = image_size // patch_size

    patches = patches.view(
        B, H, W, 3, patch_size, patch_size
    )
    patches = patches.permute(0, 3, 1, 4, 2, 5)
    return patches.reshape(B, 3, image_size, image_size)
