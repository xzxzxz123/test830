from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification
import torch
import re
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# Function to read a .txt file
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Example: Load your .txt document
file_path = "D:/lab/语料.txt"  # Path to your .txt file
text = read_txt_file(file_path)
# print(text)

# Function to clean the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

# Preprocess the extracted text
cleaned_text = preprocess_text(text)
# print(cleaned_text)



# Load pre-trained BERT tokenizer and model for sequence classification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)  # 假设有5个类别

# Example text for tokenization and encoding
inputs = tokenizer(cleaned_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Example label (you need real labels for training)
labels = torch.tensor([1])  # 假设该文本对应标签为1

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # 保存结果的目录
    num_train_epochs=3,      # 训练的轮数
    per_device_train_batch_size=4,  # 每个设备的批次大小
    logging_dir='./logs',    # 日志保存目录
)

# Define a simple dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create a dataset
encodings = tokenizer([cleaned_text], truncation=True, padding=True, max_length=512, return_tensors="pt")
dataset = CustomDataset(encodings, [1])  # 假设标签为1

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Predict labels for new text
new_text = "The city lacks adequate earthquake preparedness and response."
new_inputs = tokenizer(new_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
outputs = model(**new_inputs)
predictions = torch.argmax(outputs.logits, dim=1)

print(f"Predicted label: {predictions.item()}")
