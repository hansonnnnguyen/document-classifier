import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your labeled dataset
df = pd.read_csv("dataset.csv")  # Make sure this exists with `text,label` columns

# Encode string labels into numbers
le = LabelEncoder()
df["labels"] = le.fit_transform(df["label"]) # hugging face trainer looks for labels
joblib.dump(le, "label_encoder.pkl")  # 🔥 This fixes your error

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["text", "labels"]])
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

# Training args
training_args = TrainingArguments(
    output_dir="bert_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()

# Save everything
trainer.save_model("bert_model")
tokenizer.save_pretrained("bert_model")
