"""
Albumentations pipelines for the histopathology project.

The label depends on the centre 32x32 region of each 96x96 tile.
Therefore, these transforms deliberately avoid random crops, translations,
or other operations that can move the centre region out of frame.
"""

import albumentations as A


def get_train_transforms(
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    rotate90_p: float = 0.5,
    color_jitter_p: float = 0.3,
):
    """Return safe augmentations for training images."""
    return A.Compose(
        [
            A.HorizontalFlip(p=horizontal_flip_p),
            A.VerticalFlip(p=vertical_flip_p),
            A.RandomRotate90(p=rotate90_p),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=color_jitter_p,
            ),
        ]
    )


def get_valid_transforms():
    """Validation/test preprocessing: no random augmentation."""
    return A.Compose([])


# Convenient names for callers that prefer constants.
train_transforms = get_train_transforms()
valid_transforms = get_valid_transforms()
