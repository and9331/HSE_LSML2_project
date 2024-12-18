# Intelligent Customer Support Chatbot

## Project Overview

An ML-powered customer support chatbot that understands and responds to customer inquiries in natural language, enhancing customer experience and reducing the workload on human support agents.

## Directory Structure

project-root/ <br />
│ <br />
├── data/ <br />
│   ├── raw/ <br />
│   │   └── customer_support_on_twitter.csv <br />
│   └── processed/ <br />
│       ├── labeled_customer_support.csv <br />
│       ├── train.csv <br />
│       └── test.csv <br />
│ <br />
├── src/ <br />
│   ├── data_preprocessing/ <br />
│   │   └── preprocess.py <br />
│   ├── model_training/ <br />
│   │   ├── train.py <br />
│   │   └── requirements.txt <br />
│   └── utils/ <br />
│       └── helpers.py <br />
│ <br />
├── api/ <br />
│   ├── Dockerfile <br />
│   ├── requirements.txt <br />
│   ├── main.py <br />
│   └── models/ <br />
│       ├── best_model_state.bin <br />
│       └── intent_mapping.json <br />
│ <br />
├── client/ <br />
│   ├── Dockerfile <br />
│   └── index.html <br />
│ <br />
├── docker-compose.yml <br />
└── README.md <br />




## Setup Instructions

### 1. Data Preprocessing

#### 1.1. Label the Data

- Define your intents based on the nature of customer inquiries.
- Use keyword-based semi-automatic labeling or manually label the dataset.
- Save the labeled data as labeled_customer_support.csv in data/processed/.

#### 1.2. Preprocess the Data

1. **Navigate to Data Preprocessing Directory**
- src/data_preprocessing

2. **Install Dependencies**
- pip install pandas nltk scikit-learn

3. **Run Preprocessing Script**
- python preprocess.py

  This will generate train.csv and test.csv in the data/processed/ directory.

### 2. Model Training

1. **Navigate to Model Training Directory**

2. **Create Virtual Environment and Activate**
python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
	
3. **Install Dependencies**
pip install -r requirements.txt

4. **Run Training Script**
python train.py
    
The best model (best_model_state.bin) and intent_mapping.json will be saved in api/models/.

### 3. API Setup

1. **Navigate to API Directory**
- cd api

2. **Create Virtual Environment and Activate**
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate    

3. **Install Dependencies**
    pip install -r requirements.txt
	
4. **Run API Server**
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
	
### 4. Client Setup

1. **Navigate to Client Directory**
- cd client


2. **Serve index.html**
    python -m http.server 80 
	OR python -m http.server 8080 (if port 80 is busy)
	
3. **Access the Chatbot**
    - Open your browser and go to http://localhost:80.

### 5. Docker Deployment

1. **Navigate to Project Root Directory**
    cd project-root

2. **Build and Run Containers**
    docker-compose up --build
	
3. **Access the Services**
    - **API:** http://localhost:8000
    - **Client:** http://localhost:3000

4. **Stopping the Containers**
    - Press CTRL+C in the terminal where Docker Compose is running.
    - To remove the containers, networks, and volumes:
      docker-compose down
	  

## Technologies and Tools

- **Programming Languages:** Python, JavaScript
- **Frameworks & Libraries:**
  - **Backend:** FastAPI, PyTorch, Hugging Face Transformers
  - **Frontend:** HTML, CSS, JavaScript
- **Containerization:** Docker, Docker Compose
- **Version Control:** Git & GitHub
- **Others:** Nginx (for serving the client), Uvicorn (ASGI server)

## Potential Enhancements

- **Multilingual Support**
- **Contextual Understanding for Multi-turn Conversations**
- **Sentiment Analysis**
- **Integration with CRM Systems**
- **Voice Interface**

## License

[MIT](LICENSE)

## Contact
ambaranov_2@edu.hse.ru 

