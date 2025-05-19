# ocument Classifier (BERT + Flask)

This is a machine learning web app that classifies uploaded `.txt` or `.pdf` documents into categories like **resume**, **invoice**, or **contract** using a fine-tuned BERT model.

## Features
- Upload multiple `.txt` or `.pdf` files
- Extracts and previews document text
- Classifies documents with Hugging Face Transformers (`bert-base-uncased`)
- Displays prediction and confidence
- Built with Flask + Bootstrap UI

## Demo

<img width="629" alt="Screenshot 2025-05-19 at 4 31 43 PM" src="https://github.com/user-attachments/assets/cfc24f47-2ccb-4fa3-a5f5-4d7ef5ac1f34" />

<img width="698" alt="Screenshot 2025-05-19 at 4 31 28 PM" src="https://github.com/user-attachments/assets/f701720e-1d0c-4484-b9d1-28d2618a15e1" />



## How to Run It Locally

```bash
git clone https://github.com/your-username/document-classifier.git
cd document-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py



