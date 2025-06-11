import os
from flask import Flask, request, jsonify, render_template
import stanza
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from collections import defaultdict
import nltk
import docx
import PyPDF2
import logging
from werkzeug.utils import secure_filename
from parrot import Parrot

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Ensure NLTK 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
    logging.info("NLTK 'punkt' tokenizer found.")
except LookupError:
    logging.warning("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download('punkt')
        logging.info("NLTK 'punkt' tokenizer downloaded.")
    except Exception as e:
        logging.error(f"Failed to download NLTK 'punkt' tokenizer: {e}")

# --- Global variables for loaded models ---
nlp = None
try:
    stanza.download('en', verbose=False)
    nlp = stanza.Pipeline('en', processors='tokenize', keep_original_text=True, include_raw_tokens=True, verbose=False)
    logging.info("Stanza pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Stanza pipeline: {e}")

model_detection = None
try:
    model_detection = SentenceTransformer('all-MiniLM-L6-v2')
    logging.info("Sentence-BERT model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load Sentence-BERT model: {e}")

parrot_model = None
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize Parrot with the model tag from your first message
    parrot_model = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())
    logging.info("Parrot Paraphraser model loaded successfully.")
    logging.info(f"Using device for Parrot model: {device}")
except Exception as e:
    logging.error(f"Failed to load Parrot Paraphraser model: {e}")

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf'}
SIMILARITY_THRESHOLD = 0.75
document_analysis_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error extracting text from docx: {e}")
        return ""

def extract_text_from_pdf(file_path):
    text = []
    try:
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
        return ""
    return "\n".join(text)

def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {e}")
        return ""

def analyze_text_for_paraphrases(text, similarity_threshold):
    if nlp is None or model_detection is None:
        logging.error("Stanza pipeline or Sentence-BERT model not loaded. Cannot perform analysis.")
        return text, 0, [], []

    doc = nlp(text)
    sentences_with_indices = []
    for sentence in doc.sentences:
        sentence_text = sentence.text.strip()
        if sentence_text and sentence.tokens:
            start_char = sentence.tokens[0].start_char
            end_char = sentence.tokens[-1].end_char
            sentences_with_indices.append((sentence_text, start_char, end_char))
        else:
            logging.warning(f"Sentence with no tokens found: '{sentence_text}'")

    if not sentences_with_indices:
        logging.warning("No sentences found in the document text after segmentation.")
        return text, 0, [], []

    sentences = [s[0] for s in sentences_with_indices]
    try:
        sentence_embeddings = model_detection.encode(sentences, convert_to_tensor=True, device=device)
    except Exception as e:
        logging.error(f"Error generating sentence embeddings: {e}")
        return text, 0, [], []

    try:
        cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        return text, 0, [], []

    paraphrase_pairs = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = cosine_scores[i][j].item()
            if score > similarity_threshold:
                paraphrase_pairs.append((i, j, score))

    graph = defaultdict(list)
    for i, j, score in paraphrase_pairs:
        graph[i].append(j)
        graph[j].append(i)

    visited = set()
    paraphrase_groups = []
    def dfs(node, current_group):
        visited.add(node)
        current_group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, current_group)

    for i in range(len(sentences)):
        if i not in visited:
            current_group = []
            dfs(i, current_group)
            if len(current_group) > 1:
                paraphrase_groups.append(current_group)

    highlight_info = []
    paraphrase_groups_details = []
    for group_id, group_indices in enumerate(paraphrase_groups):
        group_segments_details = []
        for sentence_index in group_indices:
            if sentence_index < len(sentences_with_indices):
                start_char = sentences_with_indices[sentence_index][1]
                end_char = sentences_with_indices[sentence_index][2]
                highlight_info.append({
                    'start': start_char,
                    'end': end_char,
                    'group_id': group_id
                })
                group_segments_details.append({
                    'index': sentence_index,
                    'start': start_char,
                    'end': end_char,
                    'text': sentences_with_indices[sentence_index][0]
                })
            else:
                logging.warning(f"Sentence index {sentence_index} out of bounds.")

        group_pairwise_scores = []
        for i_idx in range(len(group_indices)):
            for j_idx in range(i_idx + 1, len(group_indices)):
                sentence1_idx = group_indices[i_idx]
                sentence2_idx = group_indices[j_idx]
                score = next((s for i, j, s in paraphrase_pairs if (i == sentence1_idx and j == sentence2_idx) or (i == sentence2_idx and j == sentence1_idx)), None)
                if score is not None:
                    group_pairwise_scores.append({
                        'segment1_index': sentence1_idx,
                        'segment2_index': sentence2_idx,
                        'score': score
                    })
                else:
                    logging.warning(f"Score not found for pair ({sentence1_idx}, {sentence2_idx}).")

        paraphrase_groups_details.append({
            'group_id': group_id,
            'segments': group_segments_details,
            'pairwise_scores': group_pairwise_scores
        })

    paraphrase_count = len(paraphrase_groups)
    return text, paraphrase_count, highlight_info, paraphrase_groups_details

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    global document_analysis_results
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            file.save(file_path)
            _, file_extension = os.path.splitext(filename)
            file_extension = file_extension.lower().lstrip('.')

            text_content = ""
            if file_extension == 'txt':
                text_content = extract_text_from_txt(file_path)
            elif file_extension == 'docx':
                text_content = extract_text_from_docx(file_path)
            elif file_extension == 'pdf':
                text_content = extract_text_from_pdf(file_path)
            else:
                raise ValueError("Unsupported file type for extraction.")

            if not text_content:
                raise ValueError("Could not extract text from the file.")

            original_text, count, highlights, groups_details = analyze_text_for_paraphrases(text_content, SIMILARITY_THRESHOLD)
            upload_id = f"upload_{len(document_analysis_results)}"
            document_analysis_results[upload_id] = {
                'original_text': original_text,
                'paraphrase_count': count,
                'highlights': highlights,
                'paraphrase_groups_details': groups_details,
                'filename': filename
            }

            os.remove(file_path)
            return jsonify({
                "message": "File uploaded and analyzed successfully",
                "filename": filename,
                "upload_id": upload_id,
                "original_text": original_text,
                "paraphrase_count": count,
                "highlights": highlights,
                "paraphrase_groups_details": groups_details
            }), 200

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Error processing file {filename}: {e}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/generate_paraphrases', methods=['POST'])
def generate_paraphrases():
    data = request.json
    selected_sentence = data.get("sentence", "").strip()
    tone = data.get("tone", "")  # Note: Tone is not used by Parrot

    logging.debug(f"Received request to paraphrase: '{selected_sentence}' with tone '{tone}'")
    if not selected_sentence:
        logging.error("No sentence provided for paraphrasing")
        return jsonify({"error": "No sentence provided"}), 400

    if parrot_model is None:
        logging.error("Parrot Paraphraser model not loaded.")
        return jsonify({"error": "Paraphrase generation model not loaded."}), 500

    generated_paraphrases = []
    try:
        para_phrases_with_scores = parrot_model.augment(input_phrase=selected_sentence)
        if para_phrases_with_scores:
            generated_paraphrases = [p[0] for p in para_phrases_with_scores]
        else:
            logging.warning(f"No paraphrases generated for: '{selected_sentence}'")
    except Exception as e:
        logging.error(f"Error generating paraphrase with Parrot model: {e}")
        return jsonify({"error": "Error generating paraphrase"}), 500

    return jsonify({"paraphrases": [p for p in generated_paraphrases if p.strip()]})

@app.route('/replace_sentence', methods=['POST'])
def replace_sentence():
    global document_analysis_results
    data = request.json
    upload_id = data.get("upload_id")
    original_sentence = data.get("original_sentence", "").strip()
    new_sentence = data.get("new_sentence", "").strip()

    if not upload_id or not original_sentence or not new_sentence:
        return jsonify({"error": "Missing required data (upload_id, original_sentence, new_sentence)"}), 400

    if upload_id not in document_analysis_results:
        return jsonify({"error": "Document not found for the given ID."}), 404

    current_analysis = document_analysis_results[upload_id]
    current_text = current_analysis['original_text']
    try:
        start_index = current_text.index(original_sentence)
        new_text = current_text[:start_index] + new_sentence + current_text[start_index + len(original_sentence):]
        current_analysis['original_text'] = new_text
        logging.debug(f"Replaced '{original_sentence[:50]}...' with '{new_sentence[:50]}...' in document {upload_id}")
        return jsonify({"message": "Sentence replaced successfully", "upload_id": upload_id})
    except ValueError:
        logging.warning(f"Original sentence '{original_sentence[:50]}...' not found in document {upload_id}.")
        return jsonify({"error": "Original sentence not found in document for replacement."}), 400
    except Exception as e:
        logging.error(f"Error replacing sentence in document {upload_id}: {e}")
        return jsonify({"error": f"Error replacing sentence: {str(e)}"}), 500

from export import export_bp
app.register_blueprint(export_bp)

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    if nlp is None or model_detection is None:
        logging.error("Core NLP models (Stanza or Sentence-BERT) failed to load.")
    if parrot_model is None:
        logging.warning("Parrot Paraphraser model failed to load. Generation feature will not function.")
    app.run(debug=True, host='0.0.0.0')