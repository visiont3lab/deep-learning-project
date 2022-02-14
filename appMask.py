import streamlit as st 
import pandas as pd
from PIL import Image
import numpy as np
from mylib import FaceDetector, MaskDetector

# Load models
#fc = FaceDetector("models/haarcascade_frontalface_alt.xml")
fc = MaskDetector("models/haarcascade_frontalface_alt.xml", "models/mask_model.pkl")

st.title("My first app")

uploaded_file = st.file_uploader("Choose a file", 
                        type=['png','jpeg','jpg','bmp'])
if uploaded_file is not None:
    st.write("I have the image")
    
    im_pil = Image.open(uploaded_file) # RGB

    st.image(im_pil, caption = "Uploaded Image", use_column_width=True)

    # image as array
    im = np.array ( im_pil ) # np.asarray

    imDisplay = fc.run(im)

    st.image(imDisplay, caption = "Uploaded Image", use_column_width=True)


