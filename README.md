# Dog Breed Classifier 🐶

## Motivation
This project was developed as part of the Udacity **Data Scientist Nanodegree**.  
The goal is to create a deep learning model capable of:
- Detecting whether an image contains a **dog** or a **human**.
- Predicting the **dog breed** if a dog is detected.
- Predicting the **resembling dog breed** if a human is detected.

It showcases the power of **convolutional neural networks (CNNs)** and **transfer learning** in computer vision tasks.

---

## Libraries and Dependencies
This project uses the following main Python libraries:
- **Python 3.10+**
- **TensorFlow / Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **glob**
- **PIL (Pillow)**

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Repository Structure

| File/Folder | Description |
|--------------|-------------|
| `dog_app.ipynb` | Main Jupyter notebook containing all code and analysis. |
| `saved_models/` | Folder storing model weights (e.g. `weights.best.ResNet50.keras`). |
| `haarcascades/` | Haar cascade XML files used for human face detection. |
| `bottleneck_features/` | Pre-computed ResNet50 bottleneck features. |
| `images/` | Example and test images for predictions. |
| `report.md` | Detailed project report (model architecture, training results, discussion). |
| `README.md` | (This file) Project summary, setup instructions, and acknowledgments. |

---

## Summary of Results
- **CNN trained from scratch:** 3.3% test accuracy.  
- **Transfer Learning (ResNet50):** 80.1% validation accuracy, 80% test accuracy.  
- Detects humans and dogs with nearly perfect accuracy using Haar cascades and ImageNet-based classification.

Example Output:
```
Detected DOG in image: English_springer_spaniel_04452.jpg
Top 3 Predicted Breeds:
   English springer spaniel (82.47%)
   Brittany (7.21%)
   Welsh springer spaniel (4.68%)
```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/DavidAtCupra/Data_Scientist_Dog_Breed_Classifier.git
   cd dog-breed-classifier
   ```

2. Run Jupyter Notebook:
   ```bash
   jupyter notebook dog_app.ipynb
   ```


---

## Acknowledgements
This project was developed as part of the **Udacity Data Scientist Nanodegree** program.  
Special thanks to:
- **Udacity** for the dataset and starter code.
- **Kaggle and ImageNet** for the dog images and pre-trained ResNet50 model.
- **OpenCV** for Haar cascade face detection.
- **TensorFlow / Keras** for model training support.

---

