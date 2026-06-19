import numpy as np

CONFIDENCE_THRESHOLD = 0.60
CLASSES = ['refrigerator', 'laptop', 'plumbing', 'washing_machine']


def predict_image(model, img_array):

    # preprocessing
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    predictions = model.predict(img_array,verbose=0)[0]

    idx = np.argmax(predictions)
    confidence = float(predictions[idx])
    category = CLASSES[idx]

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "service": category,
            "confidence": round(confidence * 100, 2),
            "status": "DETECTED",
        }

    return {
        "service": None,
        "confidence": round(confidence * 100, 2),
        "status": "FALLBACK_REQUIRED",
    }