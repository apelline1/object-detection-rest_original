import json
import traceback
import sys
from flask import Flask, jsonify, request
from prediction import predict

application = Flask(__name__)

# Configure logging to stdout/stderr for better visibility in containers
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)


@application.route('/')
@application.route('/status')
def status():
    return jsonify({'status': 'ok'})


@application.route('/predictions', methods=['POST'])
def create_prediction():
    try:
        data = request.data or '{}'
        body = json.loads(data)
        return jsonify(predict(body))
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON', 'message': str(e)}), 400
    except KeyError as e:
        return jsonify({'error': 'Missing required field', 'message': str(e)}), 400
    except Exception as e:
        application.logger.error(f'Error in create_prediction: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500


@application.route('/api/images', methods=['POST'])
def api_images():
    """Endpoint for frontend application to submit images for object detection"""
    try:
        application.logger.info('Received request to /api/images')
        
        # Get JSON data from request
        if request.is_json:
            body = request.get_json()
            application.logger.info('Request is JSON, parsed successfully')
        else:
            data = request.data or '{}'
            application.logger.info(f'Request is not JSON, data length: {len(data)}')
            body = json.loads(data)
        
        # Validate that image is present
        if not body or 'image' not in body:
            application.logger.error('Missing image field in request')
            return jsonify({'error': 'Missing required field: image', 'message': 'The image field is required'}), 400
        
        image_data = body.get('image')
        application.logger.info(f'Image field present, length: {len(image_data) if image_data else 0}')
        
        # Call prediction function
        application.logger.info('Calling predict function...')
        result = predict(body)
        application.logger.info(f'Prediction successful, returning {len(result.get("detections", []))} detections')
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        application.logger.error(f'JSON decode error: {str(e)}')
        print(f'JSON decode error: {str(e)}', flush=True)
        return jsonify({'error': 'Invalid JSON', 'message': str(e)}), 400
    except ValueError as e:
        error_trace = traceback.format_exc()
        application.logger.error(f'Value error: {str(e)}\n{error_trace}')
        print(f'Value error: {str(e)}\n{error_trace}', flush=True)
        return jsonify({'error': 'Invalid request', 'message': str(e)}), 400
    except KeyError as e:
        application.logger.error(f'Key error: {str(e)}')
        print(f'Key error: {str(e)}', flush=True)
        return jsonify({'error': 'Missing required field', 'message': str(e)}), 400
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        application.logger.error(f'Error in api_images: {error_msg}\n{error_trace}')
        print(f'ERROR in api_images: {error_msg}\n{error_trace}', flush=True)
        # Return a more detailed error message for debugging
        return jsonify({
            'error': 'Internal server error', 
            'message': error_msg,
            'type': type(e).__name__
        }), 500
