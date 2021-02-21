import cv2
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import dlib
import os
import pickle

from colorthief import ColorThief
from classes import WBsRGB as wb_srgb
from PIL import Image


def main():
    '''Main function that will run the whole app
    
    '''
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.title('Foundation Shades Finder')
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Introduction", "Run the app", "Dataset"])
    if app_mode == "Introduction":
        st.sidebar.success('To continue select "Run the app".')
        intro()
    elif app_mode == "Dataset":
        load_data()
    elif app_mode == "Run the app":
        run_the_app()

def intro():
    '''Section for introduction and dataset discovery
    
    '''
    st.title('About Project')

def run_the_app():
    '''Execute the app to predict sentiment base on input comment
    
    '''
    model = load_knn_model()
    uploaded_file = st.file_uploader("Upload Selfie", type='jpg')
    if uploaded_file:
        inImg = Image.open(uploaded_file)
        inImg = np.array(inImg)
        inImg = cv2.cvtColor(inImg, cv2.COLOR_RGB2BGR)
        inImg, outImg, path = white_balancing(inImg, imshow=0, imwrite=1, imcrop=1)
        col1, col2 = st.beta_columns(2)
        # To read file as bytes:
        with col1:
            st.write("Original Image")
            st.image(inImg, use_column_width=True, channels='BGR')

        with col2:
            st.write("White-Balanced Image")
            st.image(outImg, use_column_width=True, channels='BGR')

        dominant_color = get_dominant_color(path)
        dominant_palette = np.zeros((150,150,3), np.uint8)
        dominant_palette[:] = dominant_color
        st.write("Foundation Shade Color")
        st.image(dominant_palette, use_column_width=True)

        st.write("Recommended Product")
        product = model.predict([dominant_color])[0]
        st.text(product)

def load_data():
    data = './data/shades.json'
    df = pd.read_json(data)
    df['Label'] = df['brand'] + ' ' + df['product'] + ' ' + df['shade']
    df['R'] = df['hex'].apply(lambda x: hex_to_rgb(x)[0])
    df['G'] = df['hex'].apply(lambda x: hex_to_rgb(x)[1])
    df['B'] = df['hex'].apply(lambda x: hex_to_rgb(x)[2])
    st.dataframe(df)

@st.cache(allow_output_mutation=True)
def load_knn_model():
    # load the model from disk
    model = pickle.load(open('./models/knnpickle.sav', 'rb'))   
    return model

def hex_to_rgb(value):
    lv = len(value)
    return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
  (h, w) = image.shape[:2]

  if width is None and height is None:
    return image
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))

  return cv2.resize(image, dim, interpolation=inter)

# get the dominant color
def get_dominant_color(path, show=0):
    color_thief = ColorThief(path)
    dominant_color = color_thief.get_color(quality=1)
    
    if show == 1:
        dominant_palette = np.zeros((300,300,3), np.uint8)
        dominant_palette[:] = dominant_color
        plt.imshow(dominant_palette)
        plt.title('Dominant Color')
        plt.show()
        
    return list(dominant_color)

# White-balacing function    
def white_balancing(inImg, imshow=1, imwrite=1, imcrop=0):
    # use upgraded_model= 1 to load our new model that is upgraded with new
    # training examples.
    upgraded_model = 0
    # use gamut_mapping = 1 for scaling, 2 for clipping (our paper's results
    # reported using clipping). If the image is over-saturated, scaling is
    # recommended.
    gamut_mapping = 2
    # processing
    # create an instance of the WB model
    wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping,
                             upgraded=upgraded_model)
    # I = cv2.imread(in_img)  # read the image
    outImg = wbModel.correctImage(inImg) # white balance i
    outImg =(outImg*255).astype(np.uint8) 
    if imcrop == 1:
        detector = dlib.get_frontal_face_detector()
        faces = detector(outImg)
        x1 = faces[0].left() # left point
        y1 = faces[0].top() # top point
        x2 = faces[0].right() # right point
        y2 = faces[0].bottom() # bottom point
        outImg = outImg[y1:y2, x1:x2]
        inImg = inImg[y1:y2, x1:x2]
    
    #inImg = ResizeWithAspectRatio(inImg, width=600)
    #b,g,r = cv2.split(inImg)
    #inImg = cv2.merge((r,g,b))

    #outImg = ResizeWithAspectRatio(outImg, width=600)
    #b,g,r = cv2.split(outImg)
    #outImg = cv2.merge((g,b,r))

    path = ''
    if imwrite == 1:
        cv2.imwrite('/' + 'result.jpg', outImg)  # save it   
        path = '/' + 'result.jpg'
    
    return inImg, outImg, path


if __name__ == "__main__":
    main()
    