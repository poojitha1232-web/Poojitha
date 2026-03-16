import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Page configuration
st.set_page_config(layout="wide")

# Custom CSS
st.markdown("""
<style>

/* Page background */
.stApp{
    background-color:#f2f6ff;
}

/* Main Heading Style */
.main-title{
    font-size:65px;
    font-weight:800;
    text-align:center;
    margin-bottom:40px;
    font-family:'Trebuchet MS', sans-serif;
    color:#2c3e50;
}

/* Image box center */
.image-box{
    text-align:center;
}

/* Image style */
img{
    border-radius:15px;
}

/* Caption text */
.desc-text{
    font-size:28px;
    font-weight:600;
    text-align:center;
    margin-top:12px;
    font-family:Arial, sans-serif;
    color:#34495e;
}

</style>
""", unsafe_allow_html=True)

# Main heading
st.markdown('<div class="main-title">AI Image Description</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

# Load AI model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

if uploaded_file:

    image = Image.open(uploaded_file)

    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Center layout
    col1, col2, col3 = st.columns([2,3,2])

    with col2:

        st.markdown('<div class="image-box">', unsafe_allow_html=True)

        st.image(image, width=500)

        # Description below image
        st.markdown(f'<div class="desc-text">{caption}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)