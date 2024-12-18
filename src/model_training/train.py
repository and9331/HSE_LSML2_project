# src/model_training/train.py

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import random
import json

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

class SupportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SupportDataset(
        texts=df['normalized_text'].to_numpy(),
        labels=df['intent'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)

def train_epoch(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    n_examples
):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / n_examples
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, np.mean(losses), f1

def main():
    # Paths
    train_path = '../../data/processed/train.csv'
    test_path = '../../data/processed/test.csv'
    model_save_path = '../../api/models/'

    os.makedirs(model_save_path, exist_ok=True)

    # Hyperparameters
    PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
    TOKENIZER = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    # Load data
    train_df, test_df = load_data(train_path, test_path)

    # **Subsample 5% of the training data using stratified sampling**
    # train_df, _ = train_test_split(
    #     train_df,
    #     train_size=0.05,
    #     stratify=train_df['intent'],
    #     random_state=42
    # )
    # train_df = train_df.reset_index(drop=True)

    # **Subsample 10% of the train data using stratified sampling**
    # _, test_df = train_test_split(
    #     test_df,
    #     test_size=0.1,
    #     stratify=test_df['intent'],
    #     random_state=42
    # )
    # test_df = test_df.reset_index(drop=True)

    # Encode intent labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df['intent'])
    test_labels = le.transform(test_df['intent'])

    # Assign encoded labels back to DataFrames
    train_df['intent'] = train_labels
    test_df['intent'] = test_labels

    # Save label mapping
    label_mapping = {i: label for i, label in enumerate(le.classes_)}
    with open(os.path.join(model_save_path, 'intent_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)

    # Create data loaders
    train_data_loader = create_data_loader(train_df, TOKENIZER, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_df, TOKENIZER, MAX_LEN, BATCH_SIZE)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=len(le.classes_),
        output_attentions=False,
        output_hidden_states=False
    )
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps//2, gamma=0.1)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Training loop
    best_f1 = 0
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_df)
        )

        print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}')

        val_acc, val_loss, val_f1 = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(test_df)
        )

        print(f'Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}')

        if val_f1 > best_f1:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model_state.bin'))
            best_f1 = val_f1
            print("✅ Best model saved.")

    print("\nTraining complete.")

if __name__ == '__main__':
    main()
