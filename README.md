🐶 Dog Breed Classification

A deep learning project that classifies dog breeds from images using InceptionResNetV2. The model is deployed with Streamlit for an interactive web interface.

📌 About the Project

This project demonstrates a multiclass image classifier trained on 120 dog breeds.

Backbone model: InceptionResNetV2 (Transfer Learning).

Deployment: Streamlit app for easy predictions.

Outputs: Top predicted breed + confidence scores + top 5 predictions.

📂 Dataset

Dataset used: https://drive.google.com/drive/folders/1z31bsh7gNrUiwameOEWhqtWNZuKEdKQ7

Or you can use the Kaggle version
.

(Dataset not included in this repo due to large size. Please download from the links above.)

🛠️ Tech Stack

Python 3.10+

TensorFlow / Keras (Model training & inference)

Streamlit (Web interface)

NumPy, Pandas, Matplotlib (Data handling & visualization)

🚀 Features

✅ Upload or capture dog image for classification
✅ Predicts breed from 120 classes
✅ Shows confidence score & top 5 predictions
✅ Clean & interactive Streamlit UI

⚙️ Installation

Clone the repository

git clone [https://github.com/your-username/Dog-Breed-Classification.git]
cd Dog-Breed-Classification


Create virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Run Streamlit app

streamlit run app.py

📊 Model Details

Model trained using InceptionResNetV2 on dog breed dataset.

Loss function: categorical crossentropy

Optimizer: Adam

Achieved ~91% validation accuracy.

📦 Model File

The trained .keras model is too large for GitHub, so it has been uploaded to Google Drive:
🔗 [https://drive.google.com/file/d/18TWbFawj2xcVtXvoYrWxiskJRr38oOpi/view?usp=drive_link]

The app.py script is updated to automatically download and load the model from Google Drive during runtime.

🌐 Deployment

Deployed on Streamlit Cloud for easy access.
👉 Demo Link Here


📷 Example Predictions
Breed	Confidence (%)
Beagle	89.25
English Foxhound	9.70
Walker Hound	0.69
📜 License

This project is licensed under the MIT License – feel free to use and modify.

👨‍💻 Author

Developed by Tashfeen Raza ✨
For more projects, visit [(https://github.com/tashfeenraza297)]