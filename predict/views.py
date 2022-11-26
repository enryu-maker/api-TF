from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import cv2

from django.views.decorators.csrf import csrf_exempt




MODEL = tf.keras.models.load_model("model.h5")

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@csrf_exempt
def predict(request):
    if request.method == 'POST':

        file = request.FILES['img']
        image = read_file_as_image(file.read())
        output = cv2.resize(image, (256, 256))
        img_batch = np.expand_dims(output, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        result = {
            'class': predicted_class,
            'confidence': float(confidence)
            }

        return JsonResponse(result)

    else:
        return JsonResponse({'error': 'Method not allowed'})

    