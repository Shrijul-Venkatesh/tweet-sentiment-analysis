# Sentiment Analysis Performed on Twitter Dataset

A machine learning project for identifying the positivity score of a tweet, written in python

## Features

- Utilizes the Logistic Regression model for sentiment identification.
- Model training is performed on the Sentiment140 dataset of 1.6 million preprocessed tweets
- Uses the accuracy score metric functions from sklearn to easily identify the accuracy obtained through the given model.

## Installation

To install the necessary dependencies, you can use pip:

```bash
pip install -r requirements.txt

```

## Usage

In order to execute this code, download the dataset from the link given below and unzip the file into the same directory as the main.py file. Next, Uncomment the

```python
print(twitter_data.shape)
twitter_data.head()
```

to identify if the dataset has been downloaded correctly, and also uncomment the

```python
nltk.download("stopwords")
```

code provided in the main.py file to download the stopwords file from the nltk library, and execute it using the following command :

```bash
python main.py

```

## Dataset

This project uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) of 1.6 million preprocessed tweets in a csv file.

## Contributing

Contributions are most welcome! You can contact me at [venkateshshrijul@gmail.com](mailto:venkateshshrijul@gmail.com) to discuss further

## License

This project is licensed under the CC0 License. See the LICENSE file for more details
