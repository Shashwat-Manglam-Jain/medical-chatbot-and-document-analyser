from flask import Flask, request, render_template, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import pytesseract
import numpy as np
import re
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Setup vectorstore and QA model
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    vs = FAISS.load_local("faiss_index1", emb)
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    raise
retriever = vs.as_retriever(search_kwargs={"k": 5})

# Initialize tokenizer and model with adjusted settings
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
pipe = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=512, 
    temperature=0.7,  
    do_sample=True, 
    min_length=150, 
    repetition_penalty=2.5  
)

llm = HuggingFacePipeline(pipeline=pipe)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# EasyOCR reader with GPU support if available
reader = easyocr.Reader(['en'], gpu=True if os.environ.get('CUDA_AVAILABLE', 'False') == 'True' else False)

# Helper function to truncate text to fit within token limit
def truncate_to_token_limit(text, max_tokens=512):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

# Enhanced Text Preprocessing
def correct_typos(text):
    try:
        corrected = str(TextBlob(text).correct())
        medical_terms = {
            'mg', 'ml', 'tablet', 'caplet', 'capsule', 'dose', 'vicodin', 'acetaminophen',
            'hrs', 'prescription', 'nonprescription', 'pharmacist', 'drowsiness', 'constipation',
            'liver', 'allergic', 'mfg'
        }
        for term in medical_terms:
            if term in text.lower() and term not in corrected.lower():
                corrected = corrected.replace(term.lower(), term)
        # Manual fixes for common OCR errors
        corrected = corrected.replace('camlet', 'caplet').replace('thrs', 'this')
        corrected = corrected.replace('violin', 'vicodin').replace('his', 'hrs').replace('my', 'mg')
        corrected = corrected.replace('river', 'liver').replace('alkergic lo', 'allergic to')
        corrected = corrected.replace('pharmaclst', 'pharmacist').replace('phamacist', 'pharmacist')
        corrected = corrected.replace('nonprescriptionj', 'nonprescription').replace('nnprescriptionj', 'nonprescription')
        corrected = corrected.replace('am ', 'do ').replace('mug', 'mfg').replace('acelaminophen', 'acetaminophen')
        corrected = corrected.replace('clay', 'cla')
        return corrected
    except Exception as e:
        logger.warning(f"Typo correction failed: {e}")
        return text

def preprocess_query(query):
    query = query.strip().lower()
    query = re.sub(r'[^\w\s\d/-]', '', query)
    return correct_typos(query)

# Enhanced Image Preprocessing
def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        # Resize image to improve readability
        img = img.resize((int(img.width * 2), int(img.height * 2)), Image.LANCZOS)
        
        # Convert to grayscale
        gray = img.convert('L')
        
        # Enhance contrast
        contrast = ImageEnhance.Contrast(gray).enhance(1.5)
        
        # Sharpen the image
        sharp = contrast.filter(ImageFilter.SHARPEN)
        
        # Reduce noise with a lighter filter
        denoised = sharp.filter(ImageFilter.MedianFilter(size=1))
        
        # Adjust threshold to preserve text details
        thresh = denoised.point(lambda p: 255 if p > 120 else 0)
        
        # Save preprocessed image for debugging
        thresh.save("preprocessed.png")
        return thresh
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

# Improved Medication Extraction
def extract_medications(text: str):
    meds = []
    dosages = []
    
    # Pattern for medication name
    med_pat = re.compile(r'\b(vicodin\s*es|acetaminophen\s*\d+\s*mg)\b', re.I)
    found_meds = med_pat.findall(text)
    meds.extend([m.strip() for m in found_meds if m.strip()])
    
    # Pattern for dosage instructions, linked to Vicodin ES
    dose_pat = re.compile(r'(?:take|by\s*mouth)\s*(\d+)\s*(?:tablet|caplet)\s*(?:every|as\s*needed)?\s*(\d+-\d+\s*hrs)?(?:\s*or\s*as\s*needed)?(?:\s*no\s*more\s*than\s*(\d+)\s*(?:tablet|caplet)s?\s*in\s*24\s*hrs)?', re.I)
    found_doses = dose_pat.findall(text)
    if found_doses:
        tablet_count, schedule, max_dose = found_doses[0]
        dosage_info = f"Take {tablet_count} tablet(s) {schedule or 'as needed'}"
        if max_dose:
            dosage_info += f", no more than {max_dose} tablets in 24 hrs"
        dosages.append(dosage_info.strip())
    
    return {"medications": meds, "dosages": dosages} if meds or dosages else {}

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_sources", methods=["POST"])
def get_sources():
    try:
        q = preprocess_query(request.form.get("question", ""))
        if not q:
            return jsonify({"error": "No question provided"}), 400
        docs = retriever.get_relevant_documents(q)
        chunks = [d.page_content for d in docs]
        return jsonify({"sources": [chunks[i % len(chunks)] for i in range(25)]})
    except Exception as e:
        logger.error(f"Error in get_sources: {e}")
        return jsonify({"error": "Failed to retrieve sources"}), 500

@app.route("/get_answer", methods=["POST"])
def get_answer():
    try:
        question = request.form.get("question", "")
        if not question:
            return jsonify({"answer": "No question provided"}), 400
        query = preprocess_query(question)
        query = truncate_to_token_limit(query, max_tokens=200)
        docs = retriever.get_relevant_documents(query)
        truncated_docs = [truncate_to_token_limit(d.page_content, max_tokens=512) for d in docs]
        combined_input = f"Question: {query}\nContext: {' '.join(truncated_docs)}"
        combined_input = truncate_to_token_limit(combined_input, max_tokens=512)
        result = qa.invoke({"query": combined_input})
        answer = result.get("result") if isinstance(result, dict) else result
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error in get_answer: {e}")
        return jsonify({"error": "Failed to generate answer"}), 500

@app.route("/analyze_prescription", methods=["POST"])
def analyze_prescription():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        img = Image.open(io.BytesIO(file.read()))
        proc_img = preprocess_image(img)
        img_np = np.array(proc_img)

        # OCR with EasyOCR, expanded allowlist
        lines = reader.readtext(img_np, detail=0, paragraph=True, allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-# ')
        text = "\n".join(lines).strip()

        # Improved fallback to Tesseract
        if not text or len(text.split()) < 5:
            logger.info("Falling back to Tesseract OCR")
            text = pytesseract.image_to_string(proc_img, config='--psm 6 --oem 3 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-# ').strip()

        if not text or len(text.split()) < 5:
            logger.warning("No meaningful text extracted from image")
            return jsonify({
                "extracted_text": "No text extracted.",
                "patient_info": "No information extracted.",
                "medications": "No medications extracted.",
                "analysis": "No analysis available."
            }), 200

        clean_text = preprocess_query(text)
        meds_info = extract_medications(clean_text)

        # Extract patient information
        name_match = re.search(r'\b[A-Z]+\s+[A-Z]+\b', clean_text)
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
        pharmacist_match = re.search(r'rph\s*([a-z\s]+)', clean_text, re.I)
        patient_info = {
            "name": name_match.group(0) if name_match else "N/A",
            "date": date_match.group(0) if date_match else "N/A",
            "dispensed_by": f"Pharmacist: {pharmacist_match.group(1).title()}" if pharmacist_match else "N/A"
        }

        prompt = (
            "Analyze this medical prescription label. Extract patient information, medication names, dosages, instructions, and warnings. "
            "Provide a concise summary based only on the extracted text, avoiding assumptions, fabricated details, or repetition. Ensure all warnings are included. Include patient information, medications, dosages, and relevant warnings or clarifications:\n\n"
            f"Extracted Text:\n{clean_text}\n\n"
            "Format your response clearly, separating patient information, medications, dosages, and a summary."
        )
        prompt = truncate_to_token_limit(prompt, max_tokens=512)
        llm_output = pipe(prompt)[0]['generated_text']

        response = {
            "extracted_text": clean_text,
            "patient_info": patient_info,
            "medications": meds_info.get("medications", []),
            "dosages": meds_info.get("dosages", []),
            "analysis": llm_output
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in analyze_prescription: {e}")
        return jsonify({"error": f"An error occurred while analyzing the prescription: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
