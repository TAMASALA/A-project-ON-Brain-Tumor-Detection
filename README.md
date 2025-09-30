# A-project-ON-Brain-Tumor-Detection

This project focuses on detecting and classifying brain tumors from MRI images using Deep Learning and Hybrid Features Fusion (HFF).
The model extracts deep features using Convolutional Neural Networks (CNNs) and combines them with static features to enhance performance. A Support Vector Machine (SVM) is then used for final classification into Glioma, Meningioma, or Pituitary tumors.

⚙️ Workflow

Input Data: Brain MRI images resized to 224×224

Feature Extraction:

Convolutional layers (Conv2D + ReLU + Batch Normalization)

Average Pooling & Max Pooling layers

Fully Connected Layers: Dense layers with dropout for regularization

Hybrid Features Fusion (HFF): Combines deep + static features for richer representation

Classification: SVM predicts tumor type

🔬 Model Architecture

CNN Backbone → Extracts deep spatial features

Hybrid Features Fusion (HFF) → Merges deep and static features

SVM Classifier → Final tumor classification

Tumor Types Classified:

🧩 Glioma
🧠 Meningioma
🟢 Pituitary
🛠️ Technologies Used

Python
TensorFlow / Keras
CNN model
OpenCV
NumPy, Pandas, Matplotlib
