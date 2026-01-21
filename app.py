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

# --- Custom CSS (Rose Gold & Soft) ---
st.markdown("""
<style>
    /* Global Font & Theme */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #5d4037; /* Warm Brown Text */
    }

    /* Soft Rose Gold Background */
    .stApp {
        background: linear-gradient(120deg, #fff0f5 0%, #ead1dc 100%);
        background-attachment: fixed;
    }

    /* Typewriter Text - Rose Gold */
    .typewriter-text {
        font-family: 'Courier New', Courier, monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #b76e79; /* Rose Gold */
        border-right: 3px solid #b76e79;
        white-space: nowrap;
        overflow: hidden;
        margin: 0 auto;
        width: 0;
        animation: 
            typing 4s steps(30, end) infinite,
            blink-caret .75s step-end infinite;
    }
    
    @keyframes typing {
        0% { width: 0 }
        40% { width: 16ch }
        60% { width: 16ch }
        100% { width: 0 }
    }
    @keyframes blink-caret { 50% { border-color: transparent } }

    /* Inputs */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.8);
        color: #5d4037;
        border: 1px solid #dcc0cd;
        border-radius: 15px;
        padding: 12px;
        transition: all 0.3s;
    }
    .stTextInput>div>div>input:focus {
        border-color: #b76e79;
        box-shadow: 0 0 0 3px rgba(183, 110, 121, 0.2);
        background: #fff;
    }
    
    /* Buttons - Rose Gold Gradient */
    .stButton>button {
        background: linear-gradient(135deg, #b76e79 0%, #a15d68 100%);
        color: white;
        border-radius: 20px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(183, 110, 121, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(183, 110, 121, 0.4);
        background: linear-gradient(135deg, #c47f8a 0%, #b06974 100%);
    }
    
    /* Google Button */
    .google-btn-container {
        display: flex; 
        justify-content: center;
        margin-top: 20px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 15px;
        color: #5d4037;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.5);
        color: #b76e79;
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
    # Make it compact (Square-ish)
    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    
    with c2:
        # Typewriter
        st.markdown("""
        <div style="display:flex; justify-content:center; margin-bottom:1.5rem;">
            <div class="typewriter-text">Picture Perfect</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p style='color: #666; margin-bottom: 2rem; text-align: center;'>AI enhancements for your memories.</p>", unsafe_allow_html=True)
        
        email = st.text_input("Email", placeholder="hello@example.com", label_visibility="collapsed")
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
        
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        b1, b2 = st.columns(2)
        if b1.button("Login", use_container_width=True):
            if SUPABASE_AVAILABLE:
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = res.user
                    st.success("Welcome!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else: st.error("No DB")
            
        if b2.button("Join", use_container_width=True):
            if SUPABASE_AVAILABLE:
                try:
                    res = supabase.auth.sign_up({"email": email, "password": password})
                    st.success("Check email!")
                except Exception as e:
                    st.error(f"Error: {e}")

        # Divider
        st.markdown("""
        <div style="margin: 20px 0; border-top: 1px solid #eee;"></div>
        """, unsafe_allow_html=True)

        if st.button("Continue with Google", use_container_width=True):
            if SUPABASE_AVAILABLE:
                try:
                    # Get the OAuth URL
                    res = supabase.auth.sign_in_with_oauth({
                        "provider": "google",
                        "options": {
                            "query_params": {"access_type": "offline", "prompt": "consent"}
                        }
                    })
                    
                    if res.url:
                        # Use JS for reliable redirect + Fallback link
                        st.markdown(f'''
                            <a href="{res.url}" target="_self" style="
                                display: block; 
                                text-align: center; 
                                color: #b76e79; 
                                font-weight: bold; 
                                margin-top: 10px;
                                text-decoration: none;
                            ">Click here if not redirected automatically...</a>
                            <script>
                                setTimeout(function() {{
                                    window.top.location.href = "{res.url}";
                                }}, 1000);
                            </script>
                        ''', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"OAuth Error: {e}")

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
    # Authenticated UI - Cleaner, no glass cards
    st.sidebar.markdown(f"**{st.session_state['user'].email}**")
    if st.sidebar.button("Logout"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()

    t1, t2 = st.tabs(["Studio", "AI Assistant"])
    
    # === TAB 1: STUDIO ===
    with t1:
        st.header("Studio")
        
        c_up, c_par = st.columns([1,1])
        with c_up:
            uploaded_file = st.file_uploader("Upload", type=['jpg', 'png'], label_visibility="collapsed")
        with c_par:
            smile_val = st.slider("Smile", 0.0, 10.0, 6.0)
            gaze_val = st.slider("Gaze", 0.0, 2.0, 1.0)
            
        if uploaded_file:
            uploaded_file.seek(0)
            if st.button("Enhance Photo 🚀", use_container_width=True):
                with st.spinner("Processing pixels..."):
                    try:
                        orig, proc = process_initial(uploaded_file, smile_val, gaze_val)
                        st.session_state['current_orig'] = orig
                        st.session_state['current_proc'] = proc
                        st.session_state['current_name'] = uploaded_file.name
                        st.session_state['show_ai_result'] = False
                        st.success("Enhanced!")
                    except Exception as e:
                        st.error(f"Processing Error: {e}")
                        st.warning("AI engine encountered an issue. Please try another photo.")
        
        if 'current_proc' in st.session_state:
            st.markdown("---")
            ic1, ic2 = st.columns(2)
            with ic1: st.image(cv2.cvtColor(st.session_state['current_orig'], cv2.COLOR_BGR2RGB), caption="Original")
            with ic2: st.image(cv2.cvtColor(st.session_state['current_proc'], cv2.COLOR_BGR2RGB), caption="Result")
            
            # Download Button for Studio
            _, buf = cv2.imencode(".jpg", st.session_state['current_proc'])
            st.download_button(
                label="Download Result 📥",
                data=buf.tobytes(),
                file_name=f"enhanced_{st.session_state['current_name']}",
                mime="image/jpeg"
            )
                


    # === TAB 2: AI ===
    with t2:
        st.header("AI Assistant")
        if 'current_proc' in st.session_state:
            user_prompt = st.text_input("Edit Instruction", placeholder="Make it brighter...")
            if st.button("Apply"):
                if user_prompt:
                    cmds = parse_image_edit(user_prompt)
                    st.session_state['current_proc'] = apply_chat_edits(st.session_state['current_proc'], cmds)
                    st.session_state['show_ai_result'] = True
                    st.rerun()
            
            if st.session_state.get('show_ai_result', False):
                st.image(cv2.cvtColor(st.session_state['current_proc'], cv2.COLOR_BGR2RGB), caption="Result", channels="RGB")
                
                # Download Button for AI Assistant
                _, buf = cv2.imencode(".jpg", st.session_state['current_proc'])
                st.download_button(
                    label="Download AI Result 📥",
                    data=buf.tobytes(),
                    file_name=f"ai_edit_{int(time.time())}.jpg",
                    mime="image/jpeg"
                )

        else:
            st.info("Upload an image in Studio first.")



