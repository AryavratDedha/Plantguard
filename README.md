#  Plant Disease Detection using Deep Learning (CNN)

Hi!  
This is a Machine Learning / Deep Learning project where I built a model that can detect plant diseases from leaf images.

The main goal of this project is to help farmers or users identify plant diseases early, so the crop can be treated on time.

---

##  What this project does
This project takes an image of a plant leaf and predicts:
- which plant it belongs to 
- the disease name (if diseased)

---

##  Model Used
I used a **CNN (Convolutional Neural Network)** model for image classification.

I trained the model using TensorFlow/Keras and used image preprocessing + augmentation to improve performance.

I used EfficientNET as base for transfer learning.

---

## Dataset
The dataset contains images of plant leaves belonging to different categories such as:
- Healthy leaves
- Diseased leaves (different types)

Dataset source: Kaggle, Google

---

## Technologies & Tools Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

---

##  How to run this project
### Clone the repository
```bash
git clone <https://github.com/AryavratDedha/Plantguard>
cd <plantguard>
