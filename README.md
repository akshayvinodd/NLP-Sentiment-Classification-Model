# IMDB Movie Review Sentiment Analysis with TensorFlow/Keras

## Project Overview

This project uses **TensorFlow and Keras** to build a deep learning model for sentiment analysis on the **IMDB movie review dataset**. The model classifies reviews as **positive** or **negative**, showcasing the use of Natural Language Processing (NLP) with neural networks.

## Tech Stack

* Python
* TensorFlow / Keras
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn

## Features

* Preprocessing of text data (tokenization, padding)
* Deep learning model using embeddings and LSTMs/Dense layers
* Training and validation on IMDB dataset
* Evaluation using accuracy, precision, recall, and F1-score

## Dataset

The IMDB dataset is available directly through **Keras datasets API**:

```python
from tensorflow.keras.datasets import imdb
```

It contains **50,000 movie reviews**, labeled as positive or negative.

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/imdb-sentiment-analysis-tf-keras.git
   cd imdb-sentiment-analysis-tf-keras
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script to train and evaluate the model.

## Results

The model achieves strong accuracy on IMDB reviews, effectively predicting sentiment from raw text data.

## License

This project is open-source and available under the MIT License.
