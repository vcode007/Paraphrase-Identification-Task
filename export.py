import os
from flask import Blueprint, send_from_directory, jsonify, request
import docx
import logging

try:
    from server import app, document_analysis_results
    logging.info("Successfully imported app and document_analysis_results from server.py")
except ImportError:
    logging.error("Failed to import app or document_analysis_results from server.py. Ensure server.py is in the same directory.")
    app = None
    document_analysis_results = {}


export_bp = Blueprint('export', __name__)

GENERATED_FILES_FOLDER = 'generated_files'
os.makedirs(GENERATED_FILES_FOLDER, exist_ok=True)

def generate_txt(text_content, filename):
    file_path = os.path.join(GENERATED_FILES_FOLDER, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        logging.debug(f"Generated TXT file: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error generating TXT file {filename}: {e}")
        return None

def generate_docx(text_content, filename):
    file_path = os.path.join(GENERATED_FILES_FOLDER, filename)
    try:
        document = docx.Document()
        for paragraph_text in text_content.split('\n'):
             document.add_paragraph(paragraph_text)

        document.save(file_path)
        logging.debug(f"Generated DOCX file: {file_path}")
        return file_path
    except Exception as e:
        logging.error(f"Error generating DOCX file {filename}: {e}")
        return None

@export_bp.route('/export/<upload_id>/<file_type>', methods=['GET'])
def export_document(upload_id, file_type):
    logging.debug(f"Received export request for upload_id: {upload_id}, file_type: {file_type}")

    if upload_id not in document_analysis_results:
        logging.warning(f"Upload ID {upload_id} not found for export.")
        return jsonify({"error": "Document not found for export."}), 404

    current_text = document_analysis_results[upload_id].get('original_text', '')

    if not current_text:
        logging.warning(f"No text content found for upload ID {upload_id}.")
        return jsonify({"error": "No text content available for export."}), 404

    original_filename = document_analysis_results[upload_id].get('filename', f'document_{upload_id}')
    base_filename, _ = os.path.splitext(original_filename)
    export_filename = f"{base_filename}_edited.{file_type}"

    generated_file_path = None
    if file_type == 'txt':
        generated_file_path = generate_txt(current_text, export_filename)
    elif file_type == 'docx':
        generated_file_path = generate_docx(current_text, export_filename)
    else:
        logging.warning(f"Unsupported file type requested for export: {file_type}")
        return jsonify({"error": "Unsupported file type for export. Choose txt, docx."}), 400

    if generated_file_path and os.path.exists(generated_file_path):
        try:
            return send_from_directory(
                GENERATED_FILES_FOLDER,
                export_filename,
                as_attachment=True
            )
        except Exception as e:
             logging.error(f"Error sending generated file {export_filename}: {e}")
             return jsonify({"error": "Error sending file."}), 500
    else:
        logging.error(f"Failed to generate file for export: {export_filename}")
        return jsonify({"error": "Failed to generate file for export."}), 500
