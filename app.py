import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from supabase import create_client, Client
from ai.face_mesh import get_face_landmarks
from ai.smile_warp import warp_smile
from ai.gaze_correction import correct_gaze

# --- Supabase Setup ---
# Initialize Supabase client
# Uses secrets from .streamlit/secrets.toml
try:
    url: str = st.secrets["supabase"]["url"]
    key: str = st.secrets["supabase"]["key"]
    supabase: Client = create_client(url, key)
    SUPABASE_AVAILABLE = True
except Exception as e:
    st.warning("Supabase credentials not found in `.streamlit/secrets.toml`. Database features will be disabled.")
    SUPABASE_AVAILABLE = False

# --- UI Setup ---
st.set_page_config(page_title="Picture Perfect", page_icon="📸", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #1E1E1E;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("📸 Picture Perfect AI")
st.subheader("Automated Group Photo Enhancer")

# --- Auth / User State ---
if 'user' not in st.session_state:
    st.session_state['user'] = None

def login_form():
    st.sidebar.title("Login / Signup")
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("Login"):
        if SUPABASE_AVAILABLE:
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state['user'] = res.user
                st.sidebar.success("Logged in!")
                st.experimental_rerun()
            except Exception as e:
                st.sidebar.error(f"Login failed: {e}")
        else:
             st.sidebar.error("Supabase not configured.")

    if col2.button("Signup"):
        if SUPABASE_AVAILABLE:
            try:
                res = supabase.auth.sign_up({"email": email, "password": password})
                st.sidebar.success("Check your email to confirm signup!")
            except Exception as e:
                st.sidebar.error(f"Signup failed: {e}")

# --- Main App Logic ---

def process_image(image_bytes, smile_intensity, gaze_intensity):
    # Convert bytes to opencv image
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Process
    faces = get_face_landmarks(image)
    st.write(f"Detected {len(faces)} face(s).")
    
    output_image = image.copy()
    
    for face in faces:
        # Gaze
        output_image = correct_gaze(output_image, face, intensity=gaze_intensity)
        # Smile
        output_image = warp_smile(output_image, face, intensity=smile_intensity)
    
    return image, output_image

# Display Auth or User Info
if st.session_state['user']:
    st.sidebar.write(f"Welcome, {st.session_state['user'].email}")
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.experimental_rerun()
else:
    login_form()

# Input
uploaded_file = st.file_uploader("Upload a group photo", type=['jpg', 'jpeg', 'png'])

# Controls
st.sidebar.header("Adjustments")
smile_strength = st.sidebar.slider("Smile Intensity", 0.0, 10.0, 6.0)
gaze_strength = st.sidebar.slider("Gaze Correction", 0.0, 2.0, 1.0, help="1.0 = Center")

if uploaded_file is not None:
    # Reset pointer
    uploaded_file.seek(0)
    
    col1, col2 = st.columns(2)
    
    # Process
    if st.button("Enhance Photo ✨"):
        with st.spinner("Processing..."):
            original, processed = process_image(uploaded_file, smile_strength, gaze_strength)
            
            # Save to temp for downloading
            # In real app with Supabase, upload here
            
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            with col1:
                st.image(original_rgb, caption="Original", use_column_width=True)
            with col2:
                st.image(processed_rgb, caption="Picture Perfect", use_column_width=True)
                
            # DB Operations
            if SUPABASE_AVAILABLE and st.session_state['user']:
                try:
                    user_id = st.session_state['user'].id
                    
                    # 1. Upload Original
                    file_ext = uploaded_file.name.split('.')[-1]
                    import time
                    timestamp = int(time.time())
                    
                    org_filename = f"{user_id}/{timestamp}_original.{file_ext}"
                    # Reset stream
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()
                    
                    # Upload to Storage
                    # We assume a bucket named 'photos' exists.
                    bucket_name = "photos" 
                    supabase.storage.from_(bucket_name).upload(org_filename, file_bytes, {"content-type": f"image/{file_ext}"})
                    
                    # Get Public URL
                    org_url = supabase.storage.from_(bucket_name).get_public_url(org_filename)
                    
                    # Insert into 'photos' table
                    # Assuming columns: user_id, url
                    photo_data = {"user_id": user_id, "url": org_url}
                    photo_res = supabase.table("photos").insert(photo_data).execute()
                    photo_id = photo_res.data[0]['id']
                    
                    # 2. Upload Processed
                    proc_filename = f"{user_id}/{timestamp}_edited.jpg"
                    _, proc_buffer = cv2.imencode(".jpg", processed)
                    supabase.storage.from_(bucket_name).upload(proc_filename, proc_buffer.tobytes(), {"content-type": "image/jpeg"})
                    proc_url = supabase.storage.from_(bucket_name).get_public_url(proc_filename)
                    
                    # 3. Insert into 'edits' table
                    # Assuming columns: photo_id, resulting_url, settings (json)
                    edit_data = {
                        "photo_id": photo_id, 
                        "resulting_url": proc_url,
                        "settings": {"smile": smile_strength, "gaze": gaze_strength}
                    }
                    supabase.table("edits").insert(edit_data).execute()
                    
                    st.success("Saved to Cloud Protocol! ☁️")
                    
                except Exception as e:
                    st.error(f"Cloud Save Failed: {e}")
                    st.caption("Check if bucket 'photos' exists and table schemas match.")
                
            # Download
            is_success, buffer = cv2.imencode(".jpg", processed)
            st.download_button(
                label="Download Result",
                data=buffer.tobytes(),
                file_name="picture_perfect_result.jpg",
                mime="image/jpeg"
            )
