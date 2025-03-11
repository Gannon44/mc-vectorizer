import numpy as np

def augment_structure(structure):
    """
    Given a 4D structure (H, W, D, C) where C is the channel dimension (32),
    apply a set of geometric augmentations: x-flips, y-flips, and rotations
    by 90° about the z axis (assumed to be the third spatial dimension).
    
    We define a fixed set of 13 unique transformations.
    
    Returns:
      A list of augmented numpy arrays.
    """
    aug_list = []
    
    # Helper: rotation about z means rotating the first two axes.
    def rot(struct, k):
        return np.rot90(struct, k=k, axes=(0,1))
    
    # Identity and pure rotations.
    transforms = [
        lambda s: s,            # Identity
        lambda s: rot(s, 1),      # 90°
        lambda s: rot(s, 2),      # 180°
        lambda s: rot(s, 3)       # 270°
    ]
    
    # Flips along x and y.
    no_flip = lambda s: s
    flip_x = lambda s: np.flip(s, axis=0)
    flip_y = lambda s: np.flip(s, axis=1)
    flip_xy = lambda s: flip_y(flip_x(s))  # Both flips
    
    # We'll include: x-flip only, y-flip only, and each combined with rotations.
    flip_transforms = [
        no_flip,
        flip_x,
        flip_y,
        flip_xy
    ]
    
    # Add the identity and rotations.
    for t in transforms:
        aug_list.append(t(structure))
    
    # Add flips combined with each rotation (including rotation=0).
    for flip in flip_transforms:
        for k in range(0, 4):
            aug = rot(flip(structure), k)
            # Avoid duplicates of versions (in cases where houses are symmetrical).
            if not any(np.array_equal(aug, a) for a in aug_list):
                aug_list.append(aug)
    
    return aug_list
