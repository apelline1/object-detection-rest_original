import tensorflow as tf
import base64
import os

model_dir = 'models/openimages_v4_ssd_mobilenet_v2_1'

# Load model with error handling
try:
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    saved_model = tf.saved_model.load(model_dir)
    detector = saved_model.signatures['default']
except Exception as e:
    print(f"ERROR: Failed to load model from {model_dir}: {str(e)}")
    raise


def predict(body):
    print("predict() called", flush=True)
    
    if not body or 'image' not in body:
        raise ValueError("Missing required field: 'image'")
    
    base64img = body.get('image')
    if not base64img:
        raise ValueError("Image field is empty")
    
    print(f"Image data length: {len(base64img)}", flush=True)
    
    try:
        # Handle base64 strings that may or may not have data URL prefix
        if isinstance(base64img, str) and base64img.startswith('data:image'):
            # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            base64img = base64img.split(',', 1)[1]
            print("Removed data URL prefix", flush=True)
        
        # Ensure base64 string is properly formatted (add padding if needed)
        base64img = base64img.strip()
        # Add padding if necessary
        missing_padding = len(base64img) % 4
        if missing_padding:
            base64img += '=' * (4 - missing_padding)
            print(f"Added {4 - missing_padding} padding characters", flush=True)
        
        print("Decoding base64...", flush=True)
        img_bytes = base64.b64decode(base64img)
        print(f"Decoded image size: {len(img_bytes)} bytes", flush=True)
    except Exception as e:
        print(f"Base64 decode error: {str(e)}", flush=True)
        raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    if not img_bytes:
        raise ValueError("Decoded image is empty")
    
    try:
        print("Running detection...", flush=True)
        detections = detect(img_bytes)
        print("Cleaning detections...", flush=True)
        cleaned = clean_detections(detections)
        print(f"Returning {len(cleaned)} detections", flush=True)
    except Exception as e:
        print(f"Detection/cleaning error: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        raise ValueError(f"Failed to process image: {str(e)}")

    return { 'detections': cleaned }


def detect(img):
    print("detect() called", flush=True)
    try:
        print("Decoding JPEG...", flush=True)
        image = tf.image.decode_jpeg(img, channels=3)
        print("JPEG decoded successfully", flush=True)
    except Exception as e:
        print(f"JPEG decode error: {str(e)}", flush=True)
        raise ValueError(f"Failed to decode JPEG image: {str(e)}")
    
    try:
        print("Converting image dtype...", flush=True)
        converted_img = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
        print("Running detector...", flush=True)
        result = detector(converted_img)
        print("Detection complete", flush=True)
        num_detections = len(result["detection_scores"])
        print(f"Number of detections: {num_detections}", flush=True)

        print("Converting to dict...", flush=True)
        output_dict = {key:value.numpy().tolist() for key, value in result.items()}
        output_dict['num_detections'] = num_detections
        print("Detection dict created", flush=True)

        return output_dict
    except Exception as e:
        print(f"Detection error: {str(e)}", flush=True)
        import traceback
        print(traceback.format_exc(), flush=True)
        raise ValueError(f"Failed to run detection: {str(e)}")


def clean_detections(detections):
    cleaned = []
    max_boxes = 10
    num_detections = min(detections['num_detections'], max_boxes)

    for i in range(0, num_detections):
        # Handle both bytes and string types for detection_class_entities
        class_entity = detections['detection_class_entities'][i]
        if isinstance(class_entity, bytes):
            class_str = class_entity.decode('utf-8')
        else:
            class_str = str(class_entity)
        
        d = {
            'box': {
                'yMin': detections['detection_boxes'][i][0],
                'xMin': detections['detection_boxes'][i][1],
                'yMax': detections['detection_boxes'][i][2],
                'xMax': detections['detection_boxes'][i][3]
            },
            'class': class_str,
            'label': class_str,
            'score': float(detections['detection_scores'][i]),
        }
        cleaned.append(d)

    return cleaned


def preload_model():
    try:
        if os.path.exists('blank.jpeg'):
            blank_jpg = tf.io.read_file('blank.jpeg')
            blank_img = tf.image.decode_jpeg(blank_jpg, channels=3)
            detector(tf.image.convert_image_dtype(blank_img, tf.float32)[tf.newaxis, ...])
    except Exception as e:
        print(f"WARNING: Failed to preload model with blank image: {str(e)}")
        # Continue anyway - model is still loaded


try:
    preload_model()
except Exception as e:
    print(f"WARNING: Model preload failed: {str(e)}")
