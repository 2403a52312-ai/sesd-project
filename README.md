# JobGuard AI
AI-Based Fake Job and Internship Detection System using NLP and Machine Learning.

## 1. Requirements
- Python 3.10 or newer
- VS Code
- Internet connection to download the dataset

## 2. Dataset
Download the Kaggle **Real or Fake Job Posting Prediction** dataset:
https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Extract the CSV file and place it here:
`data/fake_job_postings.csv`

The dataset should contain a target column named `fraudulent` (0 = genuine, 1 = fraudulent).

## 3. Setup in VS Code
Open this folder in VS Code. Open Terminal and run:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

Install libraries:
```bash
pip install -r requirements.txt
```

## 4. Train the machine learning model
After placing the CSV file in `data/`, run:
```bash
python train_model.py
```

This creates `models/fake_job_model.joblib`.

## 5. Run the website
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Notes
- Register a user account on the website, then log in.
- The prediction is a machine-learning estimate based on the dataset, not proof that a company or job is genuine.
- A low risk score does not guarantee safety. Independently verify the employer and never pay money to obtain a job.
- This is a student project. Use a strong secret key and HTTPS before any real deployment.
