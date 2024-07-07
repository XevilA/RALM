import tensorflow as tf
import numpy as np
import cv2


model = tf.keras.models.load_model('path_to_your_trained_model.h5')


IMG_HEIGHT = 224
IMG_WIDTH = 224


def preprocess_frame(frame):
    img_array = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array

def predict_frame(frame):
    preprocessed_frame = preprocess_frame(frame)
    predictions = model.predict(preprocessed_frame)
    score = predictions[0]
    if score > 0.5:
        return 'Cancerous'
    else:
        return 'Non-Cancerous'


cap = cv2.VideoCapture(0)  # Use 1 or 2 for external cameras, or specify the IR camera ID

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    prediction = predict_frame(frame)

  
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Eye Cancer Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
