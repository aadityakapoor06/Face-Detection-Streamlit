import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Detect Face
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    # Draw Rectangle
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img,faces

def detect_eyes(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Detect Eyes
    eyes = eye_cascade.detectMultiScale(gray,1.1,4)
    # Draw Rectangle
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def detect_smiles(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Detect Smiles
    smiles = smile_cascade.detectMultiScale(gray,1.1,4)
    # Draw Rectangle around the smiles
    for (x,y,w,h) in smiles:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # Edges
    gray = cv2.medianBlur(gray,3)
    edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,2)
    # Color
    color = cv2.bilateralFilter(img,9,300,300)
    # Cartoon
    cartoon = cv2.bitwise_and(color,color,mask=edges)
    return cartoon

def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    img = cv2.GaussianBlur(img,(11,11),0)
    canny = cv2.Canny(img,100,200)
    return canny

def main():
    #Face Detection App

    st.title("Face Detection App")
    st.text("Build with streamlit and OpenCV")

    activities = ['Detection','About']
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice == 'Detection':
        st.subheader("Face Detection")

        image_file = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            st.image(our_image)

        enhance_type = st.sidebar.radio('Enhance Type',['Original','GrayScale','Contrast','Brightness','Blurring'])
        if enhance_type == 'GrayScale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            #st.write(new_img)
            st.image(gray)

        if enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast',0.5,3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        if enhance_type == 'Brightness':
            c_rate = st.sidebar.slider('Brightness',0.5,3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        if enhance_type == 'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider('Brightness',0.5,3.5)
            img = cv2.cvtColor(new_img,1)
            blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
            #st.write(new_img)
            st.image(blur_img)
        
        #else:
            #st.image(our_image,width=300)

        #Face Detection
        task = ['Faces','Smiles','Eyes','Cannize','Cartonize']
        feature_choice = st.sidebar.selectbox('Find Features',task)
        if st.button('Process'):
            if feature_choice == 'Faces':
                result_img,result_faces = detect_faces(our_image)
                st.image(result_img)
                st.success('Found {} faces'.format(len(result_faces)))
            
            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img)
                #st.success('Found {} eyes'.format(len(result_eyes)))

            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img)
                #st.success('Found {} smiles'.format(len(result_smiles)))

            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_image)
                st.image(result_img)

            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_image)
                st.image(result_canny)

    elif choice == 'About':
        st.subheader('About')


if __name__=='__main__':
    main()

