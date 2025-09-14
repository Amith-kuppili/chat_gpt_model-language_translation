# chat_gpt_model-language_translation

NOTE:for vs code view code editior to better clarification

English → French Transformer Translation

This project implements a Transformer Neural Network from scratch in TensorFlow/Keras to translate English sentences into French.
It is based on the fra-eng dataset provided by TensorFlow.

📌 Features

Data preprocessing with normalization and tokenization

Custom Positional Embedding

Self-Attention and Cross-Attention layers

Encoder–Decoder Transformer architecture

Custom learning rate schedule

Masked loss and accuracy

Visualization of training history

Inference pipeline for translation

📂 Dataset

We use the English-French sentence pairs from TensorFlow:

http://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip


After extraction:

fra-eng_extracted/
  ├── fra.txt      # sentence pairs (English ↔ French)
  └── _about.txt   # metadata

⚙️ Installation

Clone this repo and install dependencies:

git clone https://github.com/your-repo/transformer-translation.git
cd transformer-translation

pip install tensorflow numpy matplotlib


(Optional in Colab)

!pip install tensorflow numpy matplotlib

🚀 Usage
1. Preprocess Data
python preprocess.py


Normalizes text

Adds start/end tokens to French

Saves preprocessed pairs to text_pairs.pickle

2. Train Model
python train.py


Builds Transformer model

Uses custom learning rate

Trains for 20 epochs

Logs training/validation loss and accuracy

3. Translate Sentences
from inference import translate

print(translate("How are you?"))
# Output: ['[start]', 'comment', 'ça', 'va', '?', '[end]']

📊 Training History

The model tracks:

Loss vs. Epochs

Masked Accuracy vs. Epochs

Example plot:

Training history
 ├── Loss (train/val)
 └── Accuracy (train/val)

🧠 Model Architecture

Positional Embedding Layer

4 Encoder + 4 Decoder layers

Multi-Head Attention (8 heads, key_dim=128)

Feed-Forward networks

Final Dense layer (20,000 French vocab size)

Model size: ~13.8M trainable parameters

📈 Results

Can generate basic French translations after training.

Example:

English: I will act on your advice.
French (target): [start] j'agirai selon tes conseils . [end]
French (pred)  : [start] j'agirai selon vos conseils . [end]

📜 Files Overview
├── preprocess.py     # data cleaning & pickle saving
├── model.py          # Transformer architecture
├── train.py          # training loop
├── inference.py      # translation pipeline
├── utils.py          # positional encoding, custom loss, accuracy
├── text_pairs.pickle # preprocessed dataset
└── README.md         # project documentation
