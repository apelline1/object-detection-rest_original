import tensorflow as tf
import base64

model_dir = 'models/openimages_v4_ssd_mobilenet_v2_1'
saved_model = tf.saved_model.load(model_dir)
detector = saved_model.signatures['default']


def predict(body):
    if not body or 'image' not in body:
        raise ValueError("Missing required field: 'image'")
    
    base64img = body.get('image')
    if not base64img:
        raise ValueError("Image field is empty")
    
    try:
        # Handle base64 strings that may or may not have data URL prefix
        if base64img.startswith('data:image'):
            # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            base64img = base64img.split(',', 1)[1]
        
        img_bytes = base64.decodebytes(base64img.encode())
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")
    
    detections = detect(img_bytes)
    cleaned = clean_detections(detections)

    return { 'detections': cleaned }


def detect(img):
    image = tf.image.decode_jpeg(img, channels=3)
    converted_img  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    num_detections = len(result["detection_scores"])

    output_dict = {key:value.numpy().tolist() for key, value in result.items()}
    output_dict['num_detections'] = num_detections

    return output_dict


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
    blank_jpg = tf.io.read_file('blank.jpeg')
    blank_img = tf.image.decode_jpeg(blank_jpg, channels=3)
    detector(tf.image.convert_image_dtype(blank_img, tf.float32)[tf.newaxis, ...])


preload_model()
