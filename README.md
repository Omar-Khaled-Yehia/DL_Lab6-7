# Image Captioning using Flickr8k Dataset

This repository contains a deep learning pipeline to generate image captions using the Flickr8k dataset. The workflow involves feature extraction using a pre-trained InceptionV3 model, text processing, and training a sequence-to-sequence model with an LSTM decoder.

---

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Caption Preprocessing](#caption-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Future Work](#future-work)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-captioning-flickr8k.git
   cd image-captioning-flickr8k
Install required Python packages:

bash

pip install tensorflow numpy opencv-python kagglehub
Download the dataset using KaggleHub:


import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")
print("Path to dataset files:", path)
Dataset
The Flickr8k Dataset consists of 8,000 images with corresponding captions. After downloading, ensure the dataset directory is structured as follows:


Flickr8k/
│
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── captions.txt
Feature Extraction
Images are resized to 299x299 and preprocessed.

Features are extracted from the penultimate layer of the InceptionV3 model.

Features are saved in image_features.npy for reuse.


# Extract and save image features
features = extract_features(image_dir)
np.save("image_features.npy", features)
Caption Preprocessing
Captions are converted to lowercase and special characters are removed.

<start> and <end> tokens are added for sequence generation.

Captions are tokenized and padded for uniformity.


# Example cleaned caption
<start> a dog is playing in the park <end>
Model Architecture
Image Encoder: Extracted features are processed through a Dense layer.

Text Decoder: Captions are tokenized, embedded, and passed through an LSTM.

The outputs from the encoder and decoder are combined to predict the next word.


Training
A generator feeds batches of data to the model.

The model predicts the next word in the sequence using image features and partial captions as input.


history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=test_dataset,
    validation_steps=test_steps
)
Results
After 20 epochs, the model achieves:

Training Accuracy: ~30%
Validation Accuracy: ~29%
Validation Loss: ~3.7
Epoch	Training Accuracy	Validation Accuracy	Validation Loss
1	11.22%	19.68%	4.85
20	29.55%	29.71%	3.68
Future Work
Improve accuracy by experimenting with:
Larger vocabulary sizes.
Deploy the model as a web application using Flask or Streamlit.
License
This project is licensed under the MIT License.

Acknowledgments
Flickr8k Dataset
TensorFlow and Keras
KaggleHub for easy dataset download






Dataset Setup Functions
Kaggle Dataset Download
```
!pip install kagglehub
import kagglehub
path = kagglehub.dataset_download("adityajn105/flickr8k")
```
Downloads the Flickr8k dataset which contains images and their corresponding captions1.
Image Processing Components
Model Initialization
```
base_model = InceptionV3(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
```
Creates an InceptionV3 model pre-trained on ImageNet, removing the final classification layer to use it as a feature extractor4.
Image Preprocessing Functions
```
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
```
Prepares images for the InceptionV3 model by:
Resizing to 299x299 pixels (InceptionV3's required input size)
Converting to array format
Adding batch dimension
Applying InceptionV3's preprocessing14
Caption Processing Functions
Caption Cleaning
```
def clean_caption(caption):
    caption = caption.lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    return caption.strip()
```
Standardizes captions by:
Converting to lowercase
Removing punctuation
Normalizing whitespace5
Caption Loading
```
def load_captions(file_path):
    captions = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            image_id, caption = line.split(",", 1)
            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption.strip())
    return captions
```
Reads and organizes captions from the dataset file into a dictionary structure5.
Model Architecture
Caption Generation Model
```
def build_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    return Model(inputs=[inputs1, inputs2], outputs=outputs)
```
Creates a neural network that combines:
Image features processing path
Text processing path with embedding and LSTM
Decoder that merges both paths4
Training Components
Data Generator

```def data_generator(features, captions, tokenizer, max_length, batch_size=32):
    while True:
        all_image_ids = list(captions.keys())
        np.random.shuffle(all_image_ids)
        X1, X2, y = [], [], []
        for img_id in all_image_ids:
            feature = features.get(img_id)
            if feature is None:
                continue
            feature = feature.reshape(-1)
            for caption in captions[img_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    input_seq = seq[:i]
                    output_seq = seq[i]
                    input_seq = pad_sequences([input_seq], maxlen=max_length, padding='post')[0]
                    output_seq = to_categorical([output_seq], num_classes=vocab_size)[0]
                    X1.append(feature)
                    X2.append(input_seq)
                    y.append(output_seq)
                    if len(X1) == batch_size:
                        yield [np.array(X1), np.array(X2)], np.array(y)
                        X1, X2, y = [], [], []

```
Generates training batches by:
Shuffling image IDs
Creating sequences for each caption
Preparing input-output pairs for training
Yielding batches of specified size

