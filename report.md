# Dog Breed Classifier

## 1. Project Overview
The goal of this project is to build a convolutional neural network (CNN) that can identify dog breeds from images.  
The application should:
- Detect whether an image contains a **dog**, **human**, or **neither**.
- If a dog is detected → predict its **breed**.
- If a human is detected → predict the **dog breed that it resembles most**.
- Otherwise → indicate that no dog or human was detected.

This project demonstrates the use of **transfer learning** and **pre-trained deep CNNs** for computer vision tasks.

---

## 2. Data and Preprocessing
### Datasets
- **Dog dataset**: 8,351 total images, organized by 133 dog breeds.
  - Training set: 6,675 images  
  - Validation set: 835 images  
  - Test set: 835 images  
- **Human dataset**: Labeled Faces in the Wild (LFW), used to train the human detector.

### Preprocessing
- All images were resized to 224×224×3.
- Pixel values normalized using ResNet50’s preprocessing function.
- Bottleneck features were extracted using the pre-trained **ResNet50** model (`include_top=False`).

---

## 3. Human and Dog Detection
### Human Detection
- Implemented with OpenCV’s **Haar Cascade** classifier (`haarcascade_frontalface_alt.xml`).
- Tested on 100 random human and dog images:
  - Human detection accuracy: **100%**
  - False positives on dogs: **0%**

### Dog Detection
- Implemented with **ResNet50** pre-trained on ImageNet.
- ImageNet dog class indices: 151–268.
- Tested on 100 random human and dog images:
  - Dog detection accuracy: **100%**
  - False positives on humans: **0%**

---

## 4. CNN Architecture from Scratch
A lightweight CNN was implemented for dog breed classification:

| Layer | Details |
|--------|----------|
| Conv2D | 16 filters, (2×2), ReLU |
| MaxPooling2D | (2×2) |
| Conv2D | 32 filters, (2×2), ReLU |
| MaxPooling2D | (2×2) |
| Conv2D | 64 filters, (2×2), ReLU |
| MaxPooling2D | (2×2) |
| GlobalAveragePooling2D | – |
| Dense | 133 units, softmax |

**Training results:**
- Optimizer: Adam (learning_rate=0.001)  
- Epochs: 5  
- Test Accuracy: **3.31%**

This confirmed that a CNN from scratch is insufficient for this complex, high-dimensional dataset.

---

## 5. Transfer Learning with ResNet50

### Model Architecture
Bottleneck features were extracted from **ResNet50** and passed through a small custom classifier:

| Layer | Output Shape | Parameters |
|--------|---------------|-------------|
| GlobalAveragePooling2D | (None, 2048) | 0 |
| Dense (softmax) | (None, 133) | 272,517 |

### Rationale
ResNet50 was chosen for its deep residual connections that effectively prevent vanishing gradients and capture hierarchical image features.  
By freezing ResNet50’s convolutional layers and training only the top classifier, the model efficiently learns to map general image features to specific dog breeds.

### Training Summary
- Optimizer: Adam (learning_rate=0.001)  
- Epochs: 30  
- Batch size: 20  
- Training Accuracy: 99.75%  
- Validation Accuracy: **80.12%**
- Validation Loss: **0.93**

### Test Results
- **Test Accuracy:** 80.12%
- The model achieved strong generalization with minimal overfitting.

---

## 6. Algorithm Behavior

### Algorithm Steps
1. **Check for a Dog:**  
   Use `dog_detector()` (ResNet50 on ImageNet classes 151–268).
2. **Check for a Human:**  
   Use `face_detector()` (OpenCV Haar cascade).
3. **Classify Image:**
   - If a **dog** is detected → predict top 3 dog breeds using the trained ResNet50 classifier.
   - If a **human** is detected → predict the most resembling dog breed.
   - Otherwise → return “Neither dog nor human detected.”

### Example Outputs

#### 🐶 Dog Example
```
Detected DOG in image: dog_images/test/063.English_springer_spaniel/English_springer_spaniel_04452.jpg
Top 3 Predicted Breeds:
   English springer spaniel         (82.47%)
   Brittany                         (7.21%)
   Welsh springer spaniel           (4.68%)
```

#### 🙂 Human Example
```
Detected HUMAN in image: lfw/George_W_Bush/George_W_Bush_0004.jpg
This human looks like a Labrador retriever!
```

#### ❌ Neither Example
```
Error: Neither a human nor a dog was detected in the image.
```

---

## 7. Discussion and Future Improvements

### Observations
- Transfer learning drastically improved results from 3% → 80%.
- The model performs robustly across breeds, but visually similar breeds (e.g., Spaniels) can still confuse predictions.
- Some humans trigger “dog detected” false positives if lighting or pose resembles animals.

### Possible Improvements
1. **Data Augmentation:** Rotation, flipping, and scaling to improve generalization.
2. **Fine-tuning ResNet50:** Unfreeze upper convolutional layers for more domain-specific training.
3. **Model Ensemble:** Combine predictions from multiple architectures (e.g., InceptionV3, Xception).
4. **Explainability:** Add Grad-CAM visualizations to interpret what the model “sees”.

---

## 8. Conclusion
The project successfully:
- Detects humans and dogs with near-perfect accuracy.
- Classifies 133 dog breeds with ~80% accuracy using transfer learning.
- Provides a working, interpretable pipeline for breed identification.

This demonstrates the power of **transfer learning** with **ResNet50** for high-level visual classification tasks.
