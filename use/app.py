import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import os
import glob
import time
import base64


# Configure the page with fun theme
st.set_page_config(
    page_title="ദാനം കിട്ടിയ പശുവിൻ്റെ പല്ല് എണ്ണണോ", 
    page_icon="🦷", 
    layout="wide"
)


def get_gif_base64(gif_path):
    """Convert local GIF to base64 for embedding"""
    try:
        with open(gif_path, "rb") as gif_file:
            gif_bytes = gif_file.read()
            gif_base64 = base64.b64encode(gif_bytes).decode()
            return gif_base64
    except:
        return None


def show_loading_screen():
    """Display loading screen with blur effect and custom GIF"""
    gif_path = r"C:\Users\rohit\Desktop\use\images\tenor.gif"
    gif_base64 = get_gif_base64(gif_path)
    
    # CSS for blur effect and centered GIF
    st.markdown("""
    <style>
    .loading-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .loading-gif {
        width: 300px;
        height: 300px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .loading-text {
        margin-top: 20px;
        font-size: 24px;
        color: #333;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Loading screen HTML
    if gif_base64:
        loading_html = f"""
        <div class="loading-container">
            <img src="data:image/gif;base64,{gif_base64}" class="loading-gif" alt="Loading...">
            <div class="loading-text">🐄 Preparing the Experience... 🦷</div>
        </div>
        """
    else:
        loading_html = """
        <div class="loading-container">
            <div style="font-size: 48px; margin-bottom: 20px;">🐄</div>
            <div class="loading-text">🐄 Preparing the Experience... 🦷</div>
        </div>
        """
    
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
    
    # Show loading for 3 seconds (between 2-5 as requested)
    time.sleep(3)
    
    # Clear the loading screen
    loading_placeholder.empty()


@st.cache_resource
def load_model():
    """Load the YOLO model with some moo-gic!"""
    # Search for models without all the debugging noise
    train_runs = glob.glob('runs/detect/*/weights/best.pt')
    simple_paths = ['best.pt', 'runs/detect/train/weights/best.pt']
    
    all_paths = simple_paths + train_runs
    
    for model_path in all_paths:
        if os.path.exists(model_path):
            return YOLO(model_path)
    
    # Fallback to base model
    return YOLO('yolov8n.pt')


def main():
    # Initialize session state for loading
    if 'loading_done' not in st.session_state:
        st.session_state.loading_done = False
    
    # Show loading screen only once
    if not st.session_state.loading_done:
        show_loading_screen()
        st.session_state.loading_done = True
        st.rerun()  # ✅ FIXED: Changed from st.experimental_rerun()
    
    # Main title with Malayalam text
    st.markdown("# 🐄 ദാനം കിട്ടിയ പശുവിൻ്റെ പല്ല് എണ്ണണോ 🦷")
    st.markdown("### *Where every moo meets its perfect smile!* 😄")
    
    # Add some fun cow facts in sidebar
    with st.sidebar:
        st.header("🐮 Fun Cow Facts!")
        cow_facts = [
            "🦷 Cows have 32 teeth total!",
            "🌱 They chew 40-50 times per minute!",
            "😴 Cows sleep only 4 hours a day!",
            "🥛 One cow produces 6-7 gallons of milk daily!",
            "👥 Cows have best friends!"
        ]
        for fact in cow_facts:
            st.write(fact)
        
        st.markdown("---")
        confidence_threshold = st.slider(
            "🎯 Detection Sensitivity", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.25, 
            step=0.05,
            help="Lower = finds more teeth (but might see things that aren't there!)"
        )
    
    model = load_model()
    
    # Fun file uploader
    st.markdown("### 📸 Upload a Cow Photo and Let's Count Some Teeth! 🦷")
    uploaded_file = st.file_uploader(
        "🎯 Choose your cow image...", 
        type=["jpg", "jpeg", "png"],
        help="Show me those pearly whites! 🦷✨"
    )
    
    if uploaded_file is not None:
        # Process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("🚫 Oops! This image is as confusing as a cow trying to climb a tree!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Your Beautiful Cow!")
            st.image(image, channels="BGR", use_column_width=True)
        
        with col2:
            st.subheader("🔍 The Tooth Detective Results!")
            
            # Run the magic
            with st.spinner("🔄 Counting teeth like a dental hygienist..."):
                results = model(image, conf=confidence_threshold)
            
            boxes = results[0].boxes
            count = len(boxes) if boxes is not None else 0
            
            # Fun results display
            if count > 0:
                st.metric(label="🦷 Teeth Found!", value=count)
                
                # Fun messages based on count
                if count == 1:
                    st.success("🦷 Found 1 lonely tooth! It needs some friends!")
                elif count < 5:
                    st.success(f"🦷 Found {count} teeth! This cow is ready for a snack!")
                elif count < 10:
                    st.success(f"🦷 Wow! {count} teeth detected! This cow means business!")
                else:
                    st.success(f"🦷 AMAZING! {count} teeth! This cow could win a smile contest!")
                
                # Show confidence with fun emojis
                if boxes is not None:
                    confidences = boxes.conf.cpu().numpy()
                    avg_confidence = np.mean(confidences)
                    
                    confidence_emoji = "🎯" if avg_confidence > 0.7 else "🤔" if avg_confidence > 0.4 else "🙈"
                    st.write(f"{confidence_emoji} **Confidence Level:** {avg_confidence:.1%}")
                    
            else:
                st.warning("🤷‍♂️ No teeth detected! Maybe this cow is camera shy?")
                st.write("💡 Try adjusting the sensitivity slider or use a clearer photo!")
        
        # Show the annotated image with fun caption
        if count > 0:
            st.subheader("🎨 Tooth Detection Art Gallery!")
            annotated_img = results[0].plot()
            
            fun_captions = [
                f"🎯 {count} teeth spotted and boxed for your viewing pleasure!",
                f"📦 {count} teeth wrapped up with digital ribbons!",
                f"🖼️ Behold! {count} teeth in their rectangular homes!",
                f"🎪 Ladies and gentlemen, presenting {count} magnificent teeth!"
            ]
            
            st.image(annotated_img, channels="BGR", use_column_width=True, 
                    caption=fun_captions[min(count-1, 3)])
    
    else:
        # Fun waiting message
        st.markdown("### 🎭 Waiting for a Star...")
        st.write("👆 Upload a cow photo above and watch the tooth-counting magic happen!")
        st.write("🎪 *Drumroll please...* 🥁")
    
    # Fun footer
    st.markdown("---")
    st.markdown("### 🎉 Made with ❤️ and a lot of MOO-tivation! 🐄")
    st.markdown("*Remember: A cow's smile is worth a thousand moos!* 😊")


if __name__ == "__main__":
    main()
