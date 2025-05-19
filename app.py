from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import PyPDF2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # Max 5MB

# Load label encoder and Hugging Face model/tokenizer
le = joblib.load("label_encoder.pkl")
model_path = "bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Extract text from .txt or .pdf
def extract_text(file_path):
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    return text.strip()
@app.route("/", methods=["GET", "POST"])
def classify():
    results = []
    if request.method == "POST":
        files = request.files.getlist("documents")
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(save_path)

                print("----- FILE RECEIVED -----")
                print(f"Filename: {filename}")
                print(f"Saved to: {save_path}")

                text = extract_text(save_path)
                print(f"Extracted Text Preview (first 100 chars): {text[:100]}")

                if text:
                    result = classifier(text[:512])[0]
                    label_index = int(result["label"].split("_")[-1])
                    label = le.inverse_transform([label_index])[0]
                    confidence = round(result["score"] * 100, 2)

                    print(f"Prediction: {label} ({confidence}%)")

                    results.append({
                        "filename": filename,
                        "prediction": label,
                        "confidence": f"{confidence}%",
                        "preview": text[:500]
                    })
                else:
                    print("⚠️ No text extracted from file.")
                    results.append({
                        "filename": filename,
                        "prediction": "Unable to extract text",
                        "confidence": "-",
                        "preview": "N/A"
                    })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
