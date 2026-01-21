# Picture Perfect 📸

**Picture Perfect** is an advanced AI-powered image post-processing tool designed to automatically enhance group photos. It leverages computer vision and deep learning techniques to correct common imperfections such as closed eyes, awkward smiles, and undirected gazes, ensuring everyone looks their best.

## 🚀 Recently Added Features

*   **🎨 Glassmorphic 3D UI**: A stunning, modern interface with animated gradients and glass-effect cards for a premium user experience.
*   **📂 User Dashboard**: A personalized dashboard to view your history of enhancements and download past edits.
*   **☁️ Cloud Sync**: Automatically saves your original and processed photos to the cloud (Supabase) for easy access anywhere.

## ✨ Core Capabilities

*   **😁 Smile Correction**: Automatically detects and adjusts smiles to look natural using MediaPipe Face Mesh.
*   **👀 Gaze Correction**: Realigns eye gaze to look directly at the camera, fixing "looking away" shots.
*   **✨ Facial Enhancement**: Subtle sharpening and lighting adjustments to bring focus to the faces.
*   **🤖 Automated Pipeline**: Process images with a single script.

## 🛠️ Technology Stack

*   **Frontend**: Streamlit (with Custom CSS/JS)
*   **Backend**: Supabase (Auth & Storage)
*   **AI/CV**: OpenCV, MediaPipe, NumPy

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/picture-perfect.git
    cd picture-perfect
    ```

2.  **Set up a virtual environment**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Secrets**
    Create `.streamlit/secrets.toml` with your Supabase credentials:
    ```toml
    [supabase]
    url = "your-project-url"
    key = "your-anon-key"
    ```

## 📖 Usage

1.  Run the application:
    ```bash
    streamlit run app.py
    ```
2.  **Login/Signup** to access cloud features.
3.  Upload a photo in the **"✨ Create"** tab and adjust sliders.
4.  View your past edits in the **"📂 Dashboard"** tab.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created with ❤️ by [Your Name]*
