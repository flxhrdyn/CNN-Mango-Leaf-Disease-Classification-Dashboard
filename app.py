import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import io

# Set page configuration
st.set_page_config(page_title="Mango Leaf Disease Classifier", page_icon="🍃", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mango_cnn.keras')

model = load_model()

# Sidebar Navigation
st.sidebar.title("☑ Menu")
page = st.sidebar.radio("", ["Classify Image", "About The Disease", "About The Model"], label_visibility="collapsed")

if page == "Classify Image":
    st.title("🍃 Mango Leaf Disease Classifier")
    st.divider()
    st.markdown("Upload an image of a mango leaf to classify its condition.")

    # Upload image for real-time prediction
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"], help="Choose an image file to be classified")

    if uploaded_file is not None:
        st.divider()
        st.subheader("Uploaded Image:")
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Perform prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        class_labels = ['Anthracnose', 'Bacterial Canker', 'Healthy', 'Powdery Mildew']  # Update as per dataset

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.subheader(f"**Predicted Class:** {class_labels[predicted_class]}")
            st.subheader(f"**Confidence:** {confidence:.2f}%")

        st.success("✅ Prediction complete!")

elif page == "About The Disease":
    st.title("🌿 About Mango Leaf Disease")
    st.divider()
    st.subheader("""
    1. **Anthracnose** 

    Anthracnose is a disease caused by the fungus *Colletotrichum gloeosporioides*. 
    It is one of the diseases that affect mango leaves, flowers, and fruits. 
    Symptoms of anthracnose on leaves include the appearance of gray or brown spots with dark edges. O
    n mango fruits, the disease manifests as dark circular lesions, sometimes sunken, reaching a size of 4–5 cm. 
    This fungus is spread through rainwater and can develop in a relative humidity of around 95%.
    """)
    st.image("assets/anthracnose.jpg", caption="Mango Leaf with Anthracnose", width=300)
    
    st.subheader("""
    2. **Bacterial Canker**

    Black spot (*Bacterial Canker*) is a disease caused by the bacterium *Xanthomonas citri*. 
    This disease appears on both mango fruits and leaves. In severe cases, it can lead to the drying and shedding of mango leaves and fruits.
    The black spots on the fruit may exude sap, which can spread the pathogen to other mango trees. 
    This disease can reduce the quality of mango fruits and weaken the tree trunk.
    """)
    st.image("assets/bacterial_canker.jpg", caption="Mango Leaf with Bacterial Canker", width=300)
    
    st.subheader("""
    3. **Powdery Mildew**

    Powdery Mildew is a disease caused by the pathogen *Oidium mangiferae*. 
    This disease leads to fungal growth that can infect leaves, young fruits, and flowers, ultimately reducing mango fruit yields. 
    Symptoms on leaves include the appearance of gray spots that can gradually cover the entire leaf surface, causing them to wither and die. 
    This disease also causes mango flowers and young fruits to die and fall off, further decreasing the harvest. 
    Warm temperatures between 10–31°C and air humidity of 60–90% provide ideal conditions for the development of this fungus.
    """)
    st.image("assets/powdery_mildew.jpg", caption="Mango Leaf with Powdery Mildew", width=300)
    
    st.subheader("""
    4. **Healthy Leaves**

    A healthy mango leaf is vibrant green, firm, and free from blemishes or deformities. 
    It has a smooth, glossy surface with a well-defined central vein and smaller veins branching symmetrically. 
    The leaf margins are usually slightly wavy but intact, without signs of curling, yellowing, or browning. 
    A healthy leaf should not have spots, lesions, or fungal growth. 
    Additionally, it remains securely attached to the tree, showing no signs of premature wilting or shedding. 
    Proper nutrition, adequate watering, and good air circulation contribute to maintaining healthy mango leaves.
    """)
    st.image("assets/healthy.jpg", caption="Mango Leaf with Healthy Leaf", width=300)

elif page == "About The Model":
    st.title("🤖 About the CNN Model")
    st.divider()
    st.markdown("""
    This classifier uses a **Convolutional Neural Network (CNN)** trained on 2000 labeled mango leaf images. 
    The goal is to develop an automated system that can accurately identify diseased and healthy leaves, aiding farmers and researchers in early disease detection and crop management.
    """)

    st.subheader("""
    1. **The Model** 

    The Convolutional Neural Network (CNN) model is designed for classifying mango leaf diseases using deep learning. 
    It begins with five **convolutional layers**, each using **3×3 filters** with **ReLU activation** to detect features such as edges, textures, and patterns in the images. 
    After each convolutional layer, a **max-pooling layer** reduces the spatial dimensions, improving computational efficiency and helping the model focus on the most important features. 
    The extracted features are then **flattened** and passed through a **fully connected dense layer** with **256 neurons**, which allows the model to learn complex patterns. 
    A **dropout layer (0.2)** is applied to prevent overfitting by randomly deactivating neurons during training. 
    Finally, the **output layer** consists of **four neurons** with **softmax activation**, enabling the model to classify images into one of four categories. 
    This architecture ensures the model effectively captures disease patterns while maintaining generalization ability.
    """)
    st.image("assets/model_architecture.png", caption="Model Architecture", width=600)

    st.subheader("""
    2. **Model Performance** 

    The model demonstrates strong performance, achieving **92% test accuracy** with a low loss of **0.227**. 
    Both training and validation accuracy steadily increase, reaching around **91% and 95%**, 
    respectively, while the loss decreases significantly, indicating effective learning. 
    Overall, the model generalizes well, as reflected in the high test accuracy and low test loss, making it reliable for classifying mango leaf diseases.
    """)
    st.image("assets\model_performance.png", caption="Model Accuracy and Loss During Training and Validation", width=600)
    st.image("assets\model_eval.png", caption="Model Accuracy and Loss During Testing", width=600)

    st.subheader("""
    3. **Confusion Matrix** 

    The confusion matrix shows that the model performs well across all classes, with most predictions correctly classified. 
    The **healthy** and **powdery mildew** classes have near-perfect predictions (48 correct each), while **anthracnose** has only **2 misclassifications**. 
    However, **bacterial canker** has **9 misclassifications**, often confused with anthracnose, indicating some difficulty in distinguishing between these two diseases. 
    Overall, the model exhibits high accuracy but could be improved in differentiating bacterial canker from anthracnose.
    """)
    st.image("assets/conf_matrix.png", caption="Confusion Matrix", width=600)