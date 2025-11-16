# real-time-emotion-detector

This project detects human emotions in real‑time using your webcam.  
It uses a trained model (emotion_model.h5) and a label encoder (label_encoder.pkl) to classify emotions.


## Files

- realtime.py → Main script to run the webcam and detect emotions.  
- emotion_model.h5 → Trained emotion detection model.  
- label_encoder.pkl → Encodes the emotion labels.  
- requirements.txt → Python packages required to run the project.  
- .gitignore → Ignores virtual environments and cache files.

## Requirements

- Python 3.10+  
- Install dependencies from requirements.txt:


## Model File

Download emotion_model.h5 here: [Download from Google Drive](https://drive.google.com/uc?export=download&id=17QjKIqkAj0CzzK-7G0FE66T8CF6f30E8)

Place the file in the same folder as realtime.py.


How to Run

1. Make sure your webcam is connected.

2. (Optional) Activate your virtual environment:



# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

3. Run the script:

python realtime.py

4. Press q to stop the webcam.

## Install the requirements
```bash
pip install ‑r requirements.txt
