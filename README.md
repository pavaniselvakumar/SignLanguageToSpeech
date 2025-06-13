# 🧠 Sign Language to Speech Converter 🔊

A Python-based real-time sign language to speech converter using **OpenCV**, **MediaPipe**, **scikit-learn**, **pyttsx3**, and **tkinter**. This project enables users to train custom hand gesture classes and convert them into spoken words in real time — making communication more inclusive and accessible.

---

## 📸 How It Works

1. **Data Collection** – Capture hand gesture images for custom classes via webcam.
2. **Feature Extraction** – Use MediaPipe to extract hand landmarks and normalize them.
3. **Model Training** – Train a Random Forest Classifier on extracted features.
4. **Real-Time Inference** – Predict gesture classes live and convert the output to speech.

---

## 🗃️ Project Structure

📁 data/ # Collected gesture images (auto-created) ├── 0/ ├── 1/ ... 📄 collect_imgs.py # Collects gesture images from webcam 📄 create_dataset.py # Extracts features from images and creates dataset 📄 train_classifier.py # Trains ML model using scikit-learn 📄 inference.py # Runs live predictions and text-to-speech 📄 data.pickle # Pickled dataset 📄 model.p # Pickled trained model


---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/alston06/Sign-language-to-speech
cd Sign-language-to-speech
2. Install Dependencies
pip install opencv-python mediapipe scikit-learn pyttsx3 matplotlib
On Windows, use: pip install pyttsx3 pypiwin32

🧪 Step-by-Step Usage
✅ Step 1: Collect Images
python collect_imgs.py
Collects 100 images for 5 gesture classes.

Press Q to begin capture for each class.

✅ Step 2: Create Dataset
python create_dataset.py
Uses MediaPipe to extract hand landmarks.

Saves data to data.pickle.

✅ Step 3: Train Classifier
python train_classifier.py
Trains a RandomForest model on the dataset.

Saves model to model.p.

✅ Step 4: Run Real-time Inference
python inference.py
Select a voice (Male/Female) using the GUI.

Webcam will start and predict signs as spoken words.

🧠 Current Labels
The model currently supports 5 gestures:
labels_dict = {
    0: 'water',
    1: 'money',
    2: 'I love you',
    3: 'Yes',
    4: 'Hello'
}
You can customize these by changing labels_dict in inference.py and updating your dataset.

🌟 Features
📷 Real-time gesture recognition via webcam

🧠 Train your own model with new gesture classes

🔊 Speech output using pyttsx3

🪟 Voice selection through a Tkinter GUI

📁 Easy dataset and model management with Pickle

💡 Future Enhancements
Deep learning integration with CNNs

Support for multi-hand recognition

Multilingual TTS options

Save recognized gestures as full sentences

Web integration for video conferencing

🤝 Contributing
We welcome all contributions!

To Contribute:
Fork this repo

Create your branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add some feature'

Push to the branch: git push origin feature/AmazingFeature

Open a Pull Request

📄 License
This project is licensed under the MIT License.

⭐ Show Your Support
If you found this useful, drop a ⭐ on the repo to show your support!
