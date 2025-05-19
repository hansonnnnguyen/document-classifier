# ocument Classifier (BERT + Flask)

This is a machine learning web app that classifies uploaded `.txt` or `.pdf` documents into categories like **resume**, **invoice**, or **contract** using a fine-tuned BERT model.

## Features
- Upload multiple `.txt` or `.pdf` files
- Extracts and previews document text
- Classifies documents with Hugging Face Transformers (`bert-base-uncased`)
- Displays prediction and confidence
- Built with Flask + Bootstrap UI

## How to Run It Locally

```bash
git clone https://github.com/your-username/document-classifier.git
cd document-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
