import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from supabase import create_client, Client
from ai.face_mesh import get_face_landmarks
from ai.smile_warp import warp_smile
from ai.gaze_correction import correct_gaze
from ai.gemini_chatbot import parse_image_edit
from ai.chat_image_pipeline import apply_chat_edits

# --- Setup & Config ---
st.set_page_config(page_title="Picture Perfect", page_icon="📸", layout="wide")

# --- Custom CSS (3D & Entrance Animations) ---
st.markdown("""
<style>
    /* Global Font & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Animated Deep Gradient Background */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #1a0b2e, #431259, #2d1b4e, #0f0c29);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: #ffffff;
    }

    /* Entrance Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 40px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    
    .animate-enter {
        animation: fadeInUp 0.8s ease-out forwards;
    }

    /* 3D Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        margin-bottom: 2rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-style: preserve-3d;
    }
    
    .glass-card:hover {
        transform: translateY(-10px) rotateX(2deg);
        box-shadow: 0 20px 50px rgba(0,0,0,0.5), 0 0 20px rgba(255, 105, 180, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Typography */
    h1 {
        font-size: 4rem !important;
        background: linear-gradient(to right, #ff00cc, #333399); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }
    h2, h3 { color: #fff !important; }

    /* Inputs & Buttons */
    .stTextInput>div>div>input {
        background: rgba(0, 0, 0, 0.3);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 10px;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #ff00cc, #333399);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 0, 204, 0.4);
    }

    /* Image Styling */
    img {
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: transform 0.3s;
    }
    img:hover {
        transform: scale(1.02);
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

# --- Authentication ---
if 'user' not in st.session_state:
    st.session_state['user'] = None

def login_page():
    # Centered Layout with Animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="animate-enter glass-card" style="margin-top: 100px;">', unsafe_allow_html=True)
        st.title("Picture Perfect")
        st.markdown("### 🔐 Access Your Studio")
        
        email = st.text_input("Email", placeholder="user@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        
        b1, b2 = st.columns(2)
        if b1.button("Login", use_container_width=True):
            if SUPABASE_AVAILABLE:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = res.user
                    st.success("Welcome back!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Login failed: {e}")
            else: st.error("No Database Connection")
            
        if b2.button("Signup", use_container_width=True):
            if SUPABASE_AVAILABLE:
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Check your inbox!")
                except Exception as e:
                    st.error(f"Signup failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Core Logic ---
def process_initial(image_bytes, smile_intensity, gaze_intensity):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    faces = get_face_landmarks(image)
    if not faces: st.warning("No faces detected!")
    
    output_image = image.copy()
    for face in faces:
        output_image = correct_gaze(output_image, face, intensity=gaze_intensity)
        output_image = warp_smile(output_image, face, intensity=smile_intensity)
    
    return image, output_image

# --- App Render ---
if not st.session_state['user']:
    login_page()
else:
    # Sidebar
    st.sidebar.markdown(f"### 👤 {st.session_state['user'].email}")
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.experimental_rerun()

    # Main Tabs
    t1, t2, t3 = st.tabs(["✨ Studio", "🤖 AI Assistant", "📂 Gallery"])
    
    # === TAB 1: STUDIO ===
    with t1:
        st.markdown('<div class="animate-enter glass-card">', unsafe_allow_html=True)
        st.header("Create Magic ✨")
        
        c_up, c_par = st.columns([1,1])
        with c_up:
            uploaded_file = st.file_uploader("Drop your photo here", type=['jpg', 'png'])
        with c_par:
            smile_val = st.slider("Smile Factor", 0.0, 10.0, 6.0)
            gaze_val = st.slider("Gaze Fix", 0.0, 2.0, 1.0)
            
        if uploaded_file:
            uploaded_file.seek(0)
            if st.button("Enhance Photo 🚀", use_container_width=True):
                with st.spinner("Processing pixels..."):
                    orig, proc = process_initial(uploaded_file, smile_val, gaze_val)
                    st.session_state['current_orig'] = orig
                    st.session_state['current_proc'] = proc
                    st.session_state['current_name'] = uploaded_file.name
                    st.success("Enhanced!")
        
        # Display Current Work
        if 'current_proc' in st.session_state:
            ic1, ic2 = st.columns(2)
            with ic1:
                st.image(cv2.cvtColor(st.session_state['current_orig'], cv2.COLOR_BGR2RGB), caption="Original")
            with ic2:
                st.image(cv2.cvtColor(st.session_state['current_proc'], cv2.COLOR_BGR2RGB), caption="Enhanced Result")
                
            # Quick Save (auto-saves to DB for simplicity or via button)
            if st.button("Save to Gallery 💾"):
                if SUPABASE_AVAILABLE:
                    try:
                        uid = st.session_state['user'].id
                        ts = int(time.time())
                        
                        # Upload Logic (Simplified)
                        _, buf = cv2.imencode(".jpg", st.session_state['current_proc'])
                        path = f"{uid}/{ts}_save.jpg"
                        supabase.storage.from_("photos").upload(path, buf.tobytes(), {"content-type": "image/jpeg"})
                        url = supabase.storage.from_("photos").get_public_url(path)
                        
                        # Save metadata (Assuming simplified schema for now)
                        res = supabase.table("photos").insert({"user_id": uid, "url": url}).execute()
                        st.balloons()
                        st.success("Saved to Cloud!")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # === TAB 2: AI ASSISTANT ===
    with t2:
        st.markdown('<div class="animate-enter glass-card">', unsafe_allow_html=True)
        st.header("🤖 AI Editor")
        st.markdown("Talk to your photo. Example: *'Make it warmer and softer'*")
        
        if 'current_proc' in st.session_state:
            # Show current state
            st.image(cv2.cvtColor(st.session_state['current_proc'], cv2.COLOR_BGR2RGB), width=500, caption="Current State")
            
            user_prompt = st.text_input("What should we change?", placeholder="e.g., Increase brightness, make it sharper...")
            
            if st.button("Apply AI Edit 🪄"):
                if user_prompt:
                    with st.spinner("AI is thinking..."):
                        try:
                            # 1. Parse intent
                            cmds = parse_image_edit(user_prompt)
                            st.info(f"AI Detected: {cmds}")
                            
                            # 2. Apply edits to the CURRENT processed image
                            new_proc = apply_chat_edits(st.session_state['current_proc'], cmds)
                            
                            # 3. Update session
                            st.session_state['current_proc'] = new_proc
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"AI-Oops: {e}")
        else:
            st.info("Please enhance a photo in the Studio tab first!")
            
        st.markdown('</div>', unsafe_allow_html=True)

    # === TAB 3: GALLERY ===
    with t3:
        st.header("Your Masterpieces 📂")
        if SUPABASE_AVAILABLE:
            try:
                uid = st.session_state['user'].id
                # Fetch only photos table for simplicity based on previous setup
                data = supabase.table("photos").select("*").eq("user_id", uid).order("created_at", desc=True).execute()
                
                if data.data:
                    cols = st.columns(3)
                    for idx, item in enumerate(data.data):
                        with cols[idx % 3]:
                            st.markdown(f'<div class="glass-card" style="padding:1rem;">', unsafe_allow_html=True)
                            st.image(item['url'], use_column_width=True)
                            st.caption(f"Created: {item['created_at'][:10]}")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Gallery is empty.")
            except Exception as e:
                st.error(f"Could not load gallery: {e}")

