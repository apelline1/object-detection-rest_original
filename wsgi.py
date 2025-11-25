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

@application.route('/test', methods=['POST'])
def test():
    """Test endpoint to verify backend is working"""
    try:
        print("TEST endpoint called", flush=True)
        data = request.get_json() if request.is_json else {}
        return jsonify({
            'status': 'ok',
            'message': 'Backend is working',
            'received_data': str(data)[:100] if data else 'no data'
        })
    except Exception as e:
        print(f"TEST endpoint error: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500


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
        print("=" * 80, flush=True)
        print("REQUEST RECEIVED", flush=True)
        print(f"Method: {request.method}", flush=True)
        print(f"Content-Type: {request.content_type}", flush=True)
        print(f"Content-Length: {request.content_length}", flush=True)
        application.logger.info('Received request to /api/images')
        
        # Get JSON data from request - try multiple methods
        body = None
        try:
            if request.is_json:
                body = request.get_json(force=False)
                print("Parsed JSON using get_json()", flush=True)
            else:
                print("Request is not JSON, trying request.data", flush=True)
                data = request.data
                if data:
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    print(f"Data type: {type(data)}, length: {len(data)}", flush=True)
                    body = json.loads(data)
                    print("Parsed JSON from request.data", flush=True)
                else:
                    print("No data in request", flush=True)
                    body = {}
        except json.JSONDecodeError as json_err:
            print(f"JSON decode error: {str(json_err)}", flush=True)
            print(f"Data preview: {str(request.data)[:200]}", flush=True)
            raise
        except Exception as parse_err:
            print(f"Error parsing request: {str(parse_err)}", flush=True)
            print(f"Error type: {type(parse_err).__name__}", flush=True)
            raise ValueError(f"Failed to parse request: {str(parse_err)}")
        
        if body is None:
            print("Body is None after parsing", flush=True)
            return jsonify({'error': 'Invalid request', 'message': 'Request body is empty or invalid'}), 400
        
        print(f"Body type: {type(body)}, keys: {list(body.keys()) if isinstance(body, dict) else 'not a dict'}", flush=True)
        
        # Validate that image is present
        if not isinstance(body, dict):
            print(f"Body is not a dict: {type(body)}", flush=True)
            return jsonify({'error': 'Invalid request', 'message': 'Request body must be a JSON object'}), 400
        
        if 'image' not in body:
            application.logger.error('Missing image field in request')
            print("Missing image field", flush=True)
            return jsonify({'error': 'Missing required field: image', 'message': 'The image field is required'}), 400
        
        image_data = body.get('image')
        if not image_data:
            print("Image field is empty", flush=True)
            return jsonify({'error': 'Invalid request', 'message': 'Image field is empty'}), 400
        
        print(f"Image field present, type: {type(image_data)}, length: {len(str(image_data))}", flush=True)
        application.logger.info(f'Image field present, length: {len(str(image_data))}')
        
        # Check image size (rough estimate - base64 is ~33% larger than binary)
        image_str = str(image_data)
        estimated_size = len(image_str) * 3 // 4
        if estimated_size > 10 * 1024 * 1024:  # 10MB
            print(f"Image too large: estimated {estimated_size} bytes", flush=True)
            return jsonify({'error': 'Image too large', 'message': f'Image exceeds 10MB limit (estimated {estimated_size} bytes)'}), 400
        
        # Call prediction function
        print("Calling predict function...", flush=True)
        application.logger.info('Calling predict function...')
        try:
            result = predict(body)
        except MemoryError as e:
            print(f"Memory error: {str(e)}", flush=True)
            return jsonify({'error': 'Out of memory', 'message': 'Image is too large to process'}), 500
        except RuntimeError as e:
            print(f"Runtime error: {str(e)}", flush=True)
            return jsonify({'error': 'Runtime error', 'message': str(e)}), 500
        print(f"Prediction successful, returning {len(result.get('detections', []))} detections", flush=True)
        application.logger.info(f'Prediction successful, returning {len(result.get("detections", []))} detections')
        print("=" * 80, flush=True)
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        error_trace = traceback.format_exc()
        application.logger.error(f'JSON decode error: {str(e)}\n{error_trace}')
        print(f'JSON decode error: {str(e)}\n{error_trace}', flush=True)
        return jsonify({'error': 'Invalid JSON', 'message': str(e)}), 400
    except ValueError as e:
        error_trace = traceback.format_exc()
        application.logger.error(f'Value error: {str(e)}\n{error_trace}')
        print(f'Value error: {str(e)}\n{error_trace}', flush=True)
        return jsonify({'error': 'Invalid request', 'message': str(e)}), 400
    except KeyError as e:
        error_trace = traceback.format_exc()
        application.logger.error(f'Key error: {str(e)}\n{error_trace}')
        print(f'Key error: {str(e)}\n{error_trace}', flush=True)
        return jsonify({'error': 'Missing required field', 'message': str(e)}), 400
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        application.logger.error(f'Error in api_images: {error_msg}\n{error_trace}')
        print(f'ERROR in api_images: {error_msg}\n{error_trace}', flush=True)
        print("=" * 80, flush=True)
        # Return a more detailed error message for debugging
        return jsonify({
            'error': 'Internal server error', 
            'message': error_msg,
            'type': type(e).__name__
        }), 500
