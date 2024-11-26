import streamlit as st
from PIL import Image
import numpy as np
from colab_sofa_fabric import apply_fabric_to_sofa
import io

st.set_page_config(page_title="Sofa Fabric Changer", layout="wide")

st.title("Sofa Fabric Texture Changer")
st.write("Upload a sofa image and a fabric texture to change your sofa's appearance!")

# Create two columns for input images
col1, col2 = st.columns(2)

# Sofa image upload
with col1:
    st.subheader("Upload Sofa Image")
    sofa_file = st.file_uploader("Choose a sofa image...", type=['png', 'jpg', 'jpeg'])
    if sofa_file is not None:
        sofa_image = Image.open(sofa_file).convert('RGB')
        st.image(sofa_image, caption='Original Sofa', use_column_width=True)

# Fabric image upload
with col2:
    st.subheader("Upload Fabric Texture")
    fabric_file = st.file_uploader("Choose a fabric texture...", type=['png', 'jpg', 'jpeg'])
    if fabric_file is not None:
        fabric_image = Image.open(fabric_file).convert('RGB')
        st.image(fabric_image, caption='Fabric Texture', use_column_width=True)

# Parameters
st.sidebar.header("Adjust Parameters")
fabric_scale = st.sidebar.slider("Fabric Pattern Scale (%)", 0, 100, 50)
target_dpi = st.sidebar.slider("Image Resolution (DPI)", 150, 600, 300)
brightness = st.sidebar.slider("Brightness Adjustment", 0.8, 1.5, 1.2)

# Process button
if st.button("Apply Fabric Texture") and sofa_file is not None and fabric_file is not None:
    with st.spinner('Processing... Please wait...'):
        try:
            # Process the images
            result = apply_fabric_to_sofa(sofa_image, fabric_image, fabric_scale, target_dpi)
            
            # Display result
            st.subheader("Result")
            st.image(result, caption='Processed Sofa', use_column_width=True)
            
            # Save button
            buf = io.BytesIO()
            result.save(buf, format='PNG', dpi=(target_dpi, target_dpi), quality=100)
            btn = st.download_button(
                label="Download Result",
                data=buf.getvalue(),
                file_name="retextured_sofa.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try different parameters or images.")

# Add instructions and tips
with st.expander("Instructions and Tips"):
    st.markdown("""
    ### How to use:
    1. Upload a clear image of your sofa
    2. Upload a fabric texture image
    3. Adjust the parameters:
        - **Fabric Pattern Scale**: Controls the size of the fabric pattern
        - **Image Resolution**: Higher DPI for better quality (but slower processing)
        - **Brightness**: Adjust the brightness of the result
    4. Click 'Apply Fabric Texture'
    5. Download the result
    
    ### Tips for best results:
    - Use high-quality images
    - Ensure the sofa is well-lit and clearly visible
    - Choose fabric textures with clear patterns
    - Adjust the pattern scale to match your sofa's size
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Junaid using Streamlit")
