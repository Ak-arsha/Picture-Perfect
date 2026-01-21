# Picture Perfect 📸

**Picture Perfect** is an advanced AI-powered image post-processing studio. It combines computer vision for facial corrections with generative AI (Gemini) for natural language image editing, all wrapped in a stunning 3D-inspired interface.

## 🚀 Features

### 🌟 Studio Mode
*   **Smile Correction**: Natural smile enhancement using MediaPipe Face Mesh.
*   **Gaze Correction**: Realigns eyes to look directly at the camera.
*   **Instant Enhancements**: Automated pipeline for group photos.

### 🤖 AI Editor (New!)
*   **Chat with your Photos**: Use natural language to edit your images.
    *   *"Make it warmer and softer"*
    *   *"Increase contrast and sharpness"*
*   **Powered by Gemini**: Uses Google's Gemini Pro to understand your creative intent.

### 📂 Cloud Gallery
*   **Auto-Sync**: Safely stores your masterpieces in Supabase.
*   **Dashboard**: View and download your history of edits from anywhere.

### 🎨 Visuals
*   **3D Glassmorphism**: Premium glass-effect cards with 3D hover interactions.
*   **Deep Aesthetics**: Animated deep-purple and neon gradients for an immersive experience.

## 🛠️ Tech Stack
*   **Frontend**: Streamlit (Advanced CSS)
*   **AI Core**: OpenCV, MediaPipe, Google Gemini Pro
*   **Backend**: Supabase (Auth, DB, Storage)

## 📦 Quick Start

1.  **Clone & Install**
    ```bash
    git clone ...
    pip install -r requirements.txt
    ```
2.  **Secrets**
    Add your Supabase and Gemini keys to `.streamlit/secrets.toml`.
3.  **Run**
    ```bash
    streamlit run app.py
    ```

## 📝 License
MIT License.

---
*Created with ❤️ & AI*
