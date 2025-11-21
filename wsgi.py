import json
import traceback
from flask import Flask, jsonify, request
from prediction import predict

application = Flask(__name__)


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
        # Get JSON data from request
        if request.is_json:
            body = request.get_json()
        else:
            data = request.data or '{}'
            body = json.loads(data)
        
        # Validate that image is present
        if not body or 'image' not in body:
            return jsonify({'error': 'Missing required field: image'}), 400
        
        # Call prediction function
        result = predict(body)
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        return jsonify({'error': 'Invalid JSON', 'message': str(e)}), 400
    except KeyError as e:
        return jsonify({'error': 'Missing required field', 'message': str(e)}), 400
    except Exception as e:
        application.logger.error(f'Error in api_images: {str(e)}\n{traceback.format_exc()}')
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500
