from ai.image_quality import *

def apply_chat_edits(image, command):
    image = adjust_brightness_contrast(
        image,
        brightness=command.get("brightness", 0),
        contrast=command.get("contrast", 0)
    )

    image = apply_softness(image, command.get("softness", 0.0))
    image = apply_sharpness(image, command.get("sharpness", 0.0))
    image = apply_warmth(image, command.get("warmth", 0))

    return image
