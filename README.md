# Picture Perfect 📸

**Picture Perfect** is an advanced AI-powered image post-processing tool designed to automatically enhance group photos. It leverages computer vision and deep learning techniques to correct common imperfections such as closed eyes, awkward smiles, and undirected gazes, ensuring everyone looks their best.

## 🚀 Features

*   **😁 Smile Correction**: Automatically detects and adjusts smiles to look natural using MediaPipe Face Mesh.
*   **👀 Gaze Correction**: Realigns eye gaze to look directly at the camera, fixing "looking away" shots.
*   **✨ Facial Enhancement**: Subtle sharpening and lighting adjustments to bring focus to the faces.
*   **🤖 Automated Pipeline**: Process images with a single script.

## 🛠️ Technology Stack

*   **Python 3.8+**
*   **OpenCV**: For high-performance image processing and manipulation.
*   **MediaPipe**: For robust and real-time face landmark detection (468 points).
*   **NumPy**: For efficient matrix operations and geometric transformations.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/picture-perfect.git
    cd picture-perfect
    ```

2.  **Set up a virtual environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 📖 Usage

1.  Place your input images in the `test_images` folder (or verify the default `group.jpg` exists).
2.  Run the main processing script:
    ```bash
    python main.py
    ```
3.  The processed results will be saved in `test_images/` with prefixes like `result_`.

## 🧠 How It Works

### Gaze Correction
The algorithm extracts the iris region using facial landmarks, synthesizes a clean eye texture, and "warps" the iris to the center of the eye socket. It strictly adheres to the sclera boundaries to maintain realism and adds a synthetic catchlight for vitality.

### Smile Warp
Uses mesh deformation to lift the corners of the mouth and adjust the lip curvature, creating a pleasant and natural smile without distorting the surrounding skin.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created with ❤️ by [Your Name]*
