# api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
import os
import json

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model and tokenizer
MODEL_PATH = "./models/"
MODEL_NAME = 'bert-base-uncased'

# Load label mappings
with open(os.path.join(MODEL_PATH, 'intent_mapping.json'), 'r') as f:
    INTENT_MAPPING = json.load(f)  # e.g., {"0": "billing_inquiry", "1": "technical_support", ...}

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(INTENT_MAPPING))
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'best_model_state.bin'), map_location=torch.device('cpu')))
model.eval()

class Query(BaseModel):
    user_query: str


@app.api_route("/predict", methods=['POST'])
def predict(query: Query):
    print(query)
    user_input = query.user_query
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        inputs = tokenizer.encode_plus(
            user_input,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
        intent = INTENT_MAPPING[str(predicted.item())]

        # Generate responses based on intent
        response = generate_response(intent, user_input)

        return {"bot_response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_response(intent, user_input):
    # Response templates for each intent
    responses = {
        "billing_inquiry": "I'm sorry to hear you're having billing issues. Could you please provide more details?",
        "technical_support": "I'm sorry you're facing technical difficulties. Could you elaborate on the problem?",
        "product_information": "I'd be happy to provide more information about our products. What specifically would you like to know?",
        "account_management": "I can help you with your account. What do you need assistance with?",
        "general_inquiry": "Thank you for reaching out. How can I assist you today?",
    }
    return responses.get(intent, "I'm here to help! Could you please provide more information?")

@app.get("/health")
def health():
    return {"status": "Service is up and running"}
