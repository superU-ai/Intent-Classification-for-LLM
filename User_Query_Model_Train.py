from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch


# Import Dataset
final_df = pd.read_csv("data/merged_data.csv").dropna(subset=['intent'])


# Pre-Processing
final_df = final_df[["question", "intent"]].dropna().drop_duplicates()
data = final_df[final_df["intent"].isin(["information", "emotion support", "support", "faq"])]


# Function to ensure equal datapoints for each intent class
def equal_split(data, column_name):
    value_counts = data[column_name].value_counts()
    min_samples = min(value_counts)
    sampled_indices = []
    for intent, count in value_counts.items():
        sampled_indices.extend(data[data[column_name] == intent].sample(min_samples, replace=False).index)
    return data.loc[sampled_indices]
data_equal_samples = equal_split(data, 'intent')

data = data_equal_samples.reset_index().iloc[:,1:]
data['question'] = data['question'].apply(lambda x: x.lower())


# Split dataset for training and validation
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['intent'])


# Initialize tokenizer and model for training
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = len(data['intent'].unique())
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)


# Tokenize data
train_encodings = tokenizer(list(train_data['question']), truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(list(val_data['question']), truncation=True, padding=True, return_tensors='pt')


# Encode intent classes
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['intent'])
val_labels = label_encoder.transform(val_data['intent'])

# Format Dataset for training and validation
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, train_labels)
val_dataset = MyDataset(val_encodings, val_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize optimizer and loss function for training
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()


# Start training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

epochs = 15 # Change as per use case
for epoch in range(epochs):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        labels = labels.to(torch.long)
        attention_mask = attention_mask.to(torch.long)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(predictions == labels).item()
        total_samples += labels.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
    

# Check Validation Accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predicted = torch.argmax(outputs.logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy:.4f}')


# Save model
torch.save(model.state_dict(), 'user_query_model.pth')