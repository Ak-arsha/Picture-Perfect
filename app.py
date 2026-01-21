import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from supabase import create_client, Client
from ai.face_mesh import get_face_landmarks
from ai.smile_warp import warp_smile
from ai.gaze_correction import correct_gaze

# --- Setup & Config ---
st.set_page_config(page_title="Picture Perfect", page_icon="📸", layout="wide")

# --- Custom CSS (Glassmorphism & 3D Feel) ---
st.markdown("""
<style>
    /* Background Animation */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }

    /* Titles */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    p, label, .stMarkdown {
        color: white !important;
        font-weight: 500;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid white;
        border-radius: 10px;
        backdrop-filter: blur(4px);
        transition: all 0.3s ease;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: white;
        color: #e73c7e;
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(255,255,255,0.7);
    }
    
     /* Inputs */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    .stTextInput>div>div>input:focus {
        border-color: white;
        box-shadow: 0 0 10px rgba(255,255,255,0.3);
    }

    /* Image containers */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

</style>
""", unsafe_allow_html=True)

# --- Supabase Initialization ---
try:
    url: str = st.secrets["supabase"]["url"]
    key: str = st.secrets["supabase"]["key"]
    supabase: Client = create_client(url, key)
    SUPABASE_AVAILABLE = True
except Exception as e:
    st.warning("Supabase credentials not found. Database features disabled.")
    SUPABASE_AVAILABLE = False

# --- Authentication Logic ---
if 'user' not in st.session_state:
    st.session_state['user'] = None

def login_form():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.title("🔐 Login")
        email = st.text_input("Email", placeholder="name@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        
        c1, c2 = st.columns(2)
        if c1.button("Login"):
            if SUPABASE_AVAILABLE:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = res.user
                    st.success("Welcome back!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
            else:
                 st.error("Supabase not configured.")
                 
        if c2.button("Signup"):
            if SUPABASE_AVAILABLE:
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Check email to confirm!")
                except Exception as e:
                    st.error(f"Signup failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Main Processing Logic ---
def process_image(image_bytes, smile_intensity, gaze_intensity):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    faces = get_face_landmarks(image)
    st.write(f"Found {len(faces)} face(s).")
    
    output_image = image.copy()
    for face in faces:
        output_image = correct_gaze(output_image, face, intensity=gaze_intensity)
        output_image = warp_smile(output_image, face, intensity=smile_intensity)
    
    return image, output_image

# --- Check Login ---
if not st.session_state['user']:
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 1rem;'>Picture Perfect 📸</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.5rem; margin-bottom: 3rem;'>AI-Powered Group Photo Enhancer</p>", unsafe_allow_html=True)
    login_form()

else:
    # --- Authenticated App UI ---
    st.sidebar.markdown(f"### 👤 {st.session_state['user'].email}")
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.experimental_rerun()

    # --- Tabs ---
    tab_create, tab_dashboard = st.tabs(["✨ Create", "📂 Dashboard"])
    
    # --- Tab 1: Create ---
    with tab_create:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.header("New Enhancement")
        
        col_up, col_set = st.columns([1, 1])
        
        with col_up:
            uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
            
        with col_set:
            smile_strength = st.slider("Smile Intensity", 0.0, 10.0, 6.0)
            gaze_strength = st.slider("Gaze Correction", 0.0, 2.0, 1.0, help="1.0 = Center")
            
        if uploaded_file is not None:
            uploaded_file.seek(0)
            if st.button("Magic Fix ✨", use_container_width=True):
                with st.spinner("Applying AI Magic..."):
                    original, processed = process_image(uploaded_file, smile_strength, gaze_strength)
                    
                    # Store results in session for display
                    st.session_state['last_original'] = original
                    st.session_state['last_processed'] = processed
                    
                    # --- Save to Cloud ---
                    if SUPABASE_AVAILABLE:
                        try:
                            user_id = st.session_state['user'].id
                            timestamp = int(time.time())
                            file_ext = uploaded_file.name.split('.')[-1]
                            
                            # Upload Original
                            uploaded_file.seek(0)
                            org_path = f"{user_id}/{timestamp}_org.{file_ext}"
                            supabase.storage.from_("photos").upload(org_path, uploaded_file.read(), {"content-type": f"image/{file_ext}"})
                            org_url = supabase.storage.from_("photos").get_public_url(org_path)
                            
                            # DB Insert Photo
                            photo_res = supabase.table("photos").insert({"user_id": user_id, "url": org_url}).execute()
                            photo_id = photo_res.data[0]['id']
                            
                            # Upload Processed
                            _, proc_buf = cv2.imencode(".jpg", processed)
                            proc_path = f"{user_id}/{timestamp}_edit.jpg"
                            supabase.storage.from_("photos").upload(proc_path, proc_buf.tobytes(), {"content-type": "image/jpeg"})
                            proc_url = supabase.storage.from_("photos").get_public_url(proc_path)
                            
                            # DB Insert Edit
                            supabase.table("edits").insert({
                                "photo_id": photo_id,
                                "resulting_url": proc_url,
                                "settings": {"smile": smile_strength, "gaze": gaze_strength}
                            }).execute()
                            
                            st.success("Saved to your Dashboard!")
                        except Exception as e:
                            st.error(f"Cloud Save Failed: {e}")

        # Display Result
        if 'last_processed' in st.session_state:
            c1, c2 = st.columns(2)
            orig_rgb = cv2.cvtColor(st.session_state['last_original'], cv2.COLOR_BGR2RGB)
            proc_rgb = cv2.cvtColor(st.session_state['last_processed'], cv2.COLOR_BGR2RGB)
            
            with c1:
                st.image(orig_rgb, caption="Original")
            with c2:
                st.image(proc_rgb, caption="Enhanced")
                
            _, dl_buf = cv2.imencode(".jpg", st.session_state['last_processed'])
            st.download_button("Download Result", dl_buf.tobytes(), "enhanced.jpg", "image/jpeg")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Tab 2: Dashboard ---
    with tab_dashboard:
        st.header("Your Collection")
        if SUPABASE_AVAILABLE:
            try:
                user_id = st.session_state['user'].id
                
                # Fetch User Photos
                photos = supabase.table("photos").select("id, created_at").eq("user_id", user_id).order("created_at", desc=True).execute()
                
                if not photos.data:
                    st.info("No photos found. Go create some!")
                else:
                    for photo in photos.data:
                        # Fetch Edits for this photo
                        edits = supabase.table("edits").select("*").eq("photo_id", photo['id']).execute()
                        
                        if edits.data:
                            # Display card for each
                            edit = edits.data[0] # Just show latest edit for now
                            
                            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
                            dc1, dc2 = st.columns([1, 3])
                            with dc1:
                                st.image(edit['resulting_url'], caption=f"Edit from {photo['created_at'][:10]}")
                            with dc2:
                                st.subheader("Settings Used")
                                st.code(edit['settings'], language='json')
                                st.markdown(f"[View Full Image]({edit['resulting_url']})")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
            except Exception as e:
                st.error(f"Error loading dashboard: {e}")
