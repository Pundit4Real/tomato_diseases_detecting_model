from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import xgboost as xgb
import joblib

"""
Loading the Various Models
"""
# Load CNN Model
cnn_model = tf.keras.models.load_model('cnn_model_96.h5')

# Build the model right after loading with dummy input
dummy_input = np.zeros((1, 256, 256, 3))  # Adjust to your input shape
cnn_model.predict(dummy_input)  # Make a dummy call to initialize the model

# Load the saved Stacked Model (meta model)    
stacked_model = joblib.load('meta_model_96.pkl')

# Load the saved XGBoost model
xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model_96.json')

labels = {
    0: {
        'disease': 'Tomato___Bacterial_spot',
        'causes': [
            "Bacteria Xanthomonas campestris pv. vesicatoria",
            "Warm, wet, and humid conditions",
            "Overhead irrigation or rain splashing",
            "Infected seeds or transplants",
            "Poor field sanitation and equipment hygiene"
        ],
        'preventions': [
            "Use certified disease-free seeds and transplants",
            "Apply copper-based bactericides",
            "Practice crop rotation with non-host crops",
            "Remove and destroy infected plant debris",
            "Avoid overhead irrigation to minimize leaf wetness"
        ]
    },
    1: {
        'disease': 'Tomato___Early_blight',
        'causes': [
            "Fungus Alternaria solani",
            "High humidity and warm temperatures",
            "Overcrowded planting",
            "Infected crop residue or seeds",
            "Continuous tomato planting in the same field"
        ],
        'preventions': [
            "Use resistant tomato varieties",
            "Apply fungicides like chlorothalonil or mancozeb",
            "Practice crop rotation with non-susceptible crops",
            "Ensure proper plant spacing for airflow",
            "Remove and destroy crop debris after harvest"
        ]
    },
    9: {
        'disease': 'Tomato___Healthy',
        'causes': [
            "Adequate watering and sunlight",
            "Proper fertilization and soil management",
            "Regular pest and disease monitoring",
            "Timely pruning and staking",
            "Maintaining good field hygiene"
        ],
        'preventions': [
            "Continue good agricultural practices",
            "Maintain balanced soil nutrition",
            "Use mulches to conserve moisture",
            "Rotate crops to reduce soil-borne diseases",
            "Encourage beneficial insects for pest control"
        ]
    },
    2: {
        'disease': 'Tomato___Late_blight',
        'causes': [
            "Fungus-like pathogen Phytophthora infestans",
            "Cool, moist, and humid conditions",
            "Spread through wind-borne spores",
            "Overhead irrigation during late evening or night",
            "Contaminated seed or soil"
        ],
        'preventions': [
            "Use resistant varieties of tomatoes",
            "Apply preventive fungicides like metalaxyl",
            "Avoid wetting foliage, especially in late evening",
            "Remove and destroy infected plants immediately",
            "Rotate crops to prevent build-up of pathogens in the soil"
        ]
    },
    3: {
        'disease': 'Tomato___Leaf_Mold',
        'causes': [
            "Fungus Cladosporium fulvum",
            "High humidity and poor air circulation",
            "Cool, wet, and shaded environments",
            "Infected seeds or transplants",
            "Overcrowded planting or dense canopy"
        ],
        'preventions': [
            "Ensure good ventilation in greenhouses",
            "Apply fungicides like mancozeb or copper-based solutions",
            "Practice crop rotation and use resistant varieties",
            "Remove infected leaves and destroy them",
            "Avoid overhead watering and ensure proper spacing"
        ]
    },
    4: {
        'disease': 'Tomato___Septoria_leaf_spot',
        'causes': [
            "Fungus Septoria lycopersici",
            "Wet, humid weather conditions",
            "Overhead irrigation or rain splashing",
            "Infected crop debris or seeds",
            "Dense plant canopy"
        ],
        'preventions': [
            "Apply fungicides like chlorothalonil or copper sprays",
            "Remove and destroy infected leaves",
            "Ensure good airflow by proper spacing and pruning",
            "Avoid overhead watering",
            "Use resistant tomato varieties if available"
        ]
    },
    5: {
        'disease': 'Tomato___Spider_mites Two-spotted_spider_mite',
        'causes': [
            "Spider mite Tetranychus urticae infestation",
            "Hot, dry weather conditions",
            "Over-fertilization with nitrogen",
            "Weed presence that harbors mites",
            "Lack of natural predators like ladybugs"
        ],
        'preventions': [
            "Use insecticidal soaps or horticultural oils",
            "Encourage natural predators like ladybugs",
            "Avoid over-fertilizing with nitrogen-rich fertilizers",
            "Keep weeds under control",
            "Ensure adequate irrigation to reduce stress on plants"
        ]
    },
    6: {
        'disease': 'Tomato___Target_Spot',
        'causes': [
            "Fungus Corynespora cassiicola",
            "Warm and humid conditions",
            "Overcrowded planting and poor airflow",
            "Infected seeds or transplants",
            "Contaminated farm equipment or workers"
        ],
        'preventions': [
            "Use resistant varieties of tomatoes",
            "Apply fungicides like chlorothalonil",
            "Ensure proper plant spacing for airflow",
            "Remove and destroy infected plant material",
            "Practice crop rotation with non-host crops"
        ]
    },
    8: {
        'disease': 'Tomato___Tomato_mosaic_virus',
        'causes': [
            "Tomato mosaic virus (ToMV) infection",
            "Handling by infected workers or tools",
            "Infected seeds or transplants",
            "Contaminated soil or water",
            "Tobacco use by workers while handling plants"
        ],
        'preventions': [
            "Use virus-free seeds and transplants",
            "Disinfect tools and hands before handling plants",
            "Remove and destroy infected plants immediately",
            "Practice crop rotation to prevent soil contamination",
            "Ban tobacco use near tomato fields"
        ]
    },
    7: {
        'disease': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'causes': [
            "Tomato yellow leaf curl virus (TYLCV)",
            "Whitefly Bemisia tabaci transmission",
            "Presence of infected plants nearby",
            "Poor field hygiene and management",
            "Lack of resistant tomato varieties"
        ],
        'preventions': [
            "Use TYLCV-resistant tomato varieties",
            "Control whitefly populations with insecticides",
            "Remove and destroy infected plants immediately",
            "Practice good field hygiene and sanitation",
            "Use reflective mulches to repel whiteflies"
        ]
    }
}


def load_and_preprocess_image(image_bytes: bytes, img_path="", target_size=(256, 256)):
    if not isinstance(image_bytes, bytes):
        # Load the image
        img = image.load_img(img_path, target_size=target_size)
        # Convert the image to array
        img_array = image.img_to_array(img)
    else:
        # Load the image from bytes
        image_file = Image.open(BytesIO(image_bytes))
        image_file = image_file.resize(target_size)  # Assign back after resize
        img_array = image.img_to_array(image_file)
    
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array (if normalization was used during training)
    img_array /= 255.0
    return img_array

def cnn_prediction(processed_image):
    prediction = cnn_model.predict(processed_image)
    print("CNN Probabilities:",(prediction))
    cnn_predicted_class = np.argmax(prediction, axis=1)
    return cnn_predicted_class[0]


def xgb_prediction(processed_image):
    # Use the feature extractor part of CNN to get the features for XGBoost
    feature_extractor = tf.keras.models.Model(
        [cnn_model.inputs],  # Input tensor of the main model
        outputs=cnn_model.get_layer('my_dense').output  # Output tensor of the 'my_dense' layer
    )
    image_features = feature_extractor.predict(processed_image)
    

    # Reshape for XGBoost prediction
    image_features_reshaped = image_features.reshape(image_features.shape[0], -1)
    dimage = xgb.DMatrix(image_features_reshaped)

    # Predict with the XGBoost model
    xgb_prediction = xgb_model.predict(dimage)
    print("XGB Probabilities:",(xgb_prediction))
    # Check the shape of xgb_prediction to determine correct axis for argmax
    if len(xgb_prediction.shape) > 1:
        xgb_predicted_class = np.argmax(xgb_prediction, axis=1)  # For 2D array
    else:
        xgb_predicted_class = np.argmax(xgb_prediction, axis=0)

    return xgb_predicted_class[0]

def stacked_model_prediction(cnn_predicted_class, xgb_predicted_class):
    # Prepare the combined input for the Stacked model
    stacked_features = np.column_stack((cnn_predicted_class, xgb_predicted_class))
    # Predict with the stacked model
    stacked_prediction = stacked_model.predict(stacked_features)
    results = labels[int(stacked_prediction[0])]
    print("Meta Probablities", stacked_prediction)
    results['confidence'] = np.max(stacked_model.predict_proba(stacked_features))
    return results  # Ensure output is integer index

def make_prediction(processed_image):
    # Get predictions from CNN and XGBoost
    cnn_class = cnn_prediction(processed_image)
    xgb_class = xgb_prediction(processed_image)
    
    # Get final prediction from stacked model
    return stacked_model_prediction(cnn_class, xgb_class)

# Example usage:
# processed_img = load_and_preprocess_image(image_bytes=None, img_path="path_to_image.jpg")
# prediction = make_prediction(processed_img)
# print("Predicted Class:", prediction)
