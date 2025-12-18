"""
Flask REST API for Fake News Detection - AWS Deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import time
import logging

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.claim_extractor import extract_text_from_url, extract_claim_from_text, clean_text
from app.core.query_generator import generate_queries
from app.core.web_search import web_search
from app.core.evidence_aggregator import build_evidence
from app.core.verdict_engine import compute_final_verdict

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return jsonify({
        'name': 'Fake News Detector API',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/api/check', methods=['POST'])
def check_claim():
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        text_input = data.get('text', '')
        url_input = data.get('url', '')
        claim_input = data.get('claim', '')
        max_results = min(int(data.get('max_results', 3)), 10)
        
        if not text_input and not url_input and not claim_input:
            return jsonify({'error': 'Must provide text, url, or claim'}), 400
        
        # Extract claim
        if claim_input:
            claim = claim_input
        elif url_input:
            full_text = extract_text_from_url(url_input)
            if not full_text:
                return jsonify({'error': 'Could not extract text from URL'}), 400
            claim = extract_claim_from_text(full_text)
        else:
            claim = extract_claim_from_text(text_input)
        
        # Process
        queries = generate_queries(claim)
        search_results = web_search(queries, max_results=max_results)
        evidences = build_evidence(claim, search_results)
        verdict_result = compute_final_verdict(evidences)
        
        # Response
        response = {
            'claim': claim,
            'verdict': verdict_result['verdict'],
            'confidence': verdict_result['confidence'],
            'net_score': verdict_result['net_score'],
            'evidences': evidences,
            'processing_time': round(time.time() - start_time, 2),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)
