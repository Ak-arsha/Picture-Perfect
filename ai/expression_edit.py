import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import cv2

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")

def edit_expression(face_img, target_expression):
    prompt = f"a realistic human face with a {target_expression} expression"
    
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)

    result = pipe(
        prompt=prompt,
        image=pil_img,
        strength=0.35,
        guidance_scale=7.5
    ).images[0]

    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
