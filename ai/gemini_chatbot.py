import os
import json
import google.generativeai as genai

genai.configure(api_key="AIzaSyCqyOa47KltbWUucr_BY2mR7EBVbDgWAnI")

model = genai.GenerativeModel("gemini-pro")

SYSTEM_PROMPT = """
You are an image editing assistant.

Your job:
- Convert user requests into image adjustment parameters.
- DO NOT edit faces, expressions, or gaze.
- Only handle image quality and background.

Allowed JSON keys:
brightness (-100 to 100)
contrast (-100 to 100)
softness (0.0 to 1.0)
sharpness (0.0 to 1.0)
warmth (-50 to 50)

Rules:
- Respond ONLY with valid JSON
- If user asks something unsupported, ignore it
"""

def parse_image_edit(user_text):
    response = model.generate_content(
        SYSTEM_PROMPT + "\nUser request: " + user_text
    )

    try:
        return json.loads(response.text)
    except:
        return {}
