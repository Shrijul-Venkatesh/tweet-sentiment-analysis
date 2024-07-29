import pandas

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from stemmer import stemming
from accuracy_test import training_accuracy, testing_accuracy

# Creating the dataframe
column_names = ["target", "id", "date", "flag", "user", "text"]
twitter_data = pandas.read_csv(
    "./archive/training.1600000.processed.noemoticon.csv",
    names=column_names,
    encoding="ISO-8859-1",
)

# Run the below commands to check if the dataset csv has been downloaded and imported correctly
# print(twitter_data.shape)
# twitter_data.head()

# Run the below command only ONCE to download the stopwords folder from nltk
# nltk.download("stopwords")

# Converting the distribution to binary for better classification understanding
twitter_data.replace({"target": {4: 1}}, inplace=True)

# Stemming
port_stem = PorterStemmer()

# Note that performing stemming on the a dataset of this size is time consuming
twitter_data["stemmed_content"] = twitter_data["text"].apply(
    lambda x: stemming(x, port_stem)
)

print(twitter_data.head())

X = twitter_data["stemmed_content"].values
Y = twitter_data["target"].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2  # type: ignore
)

# Vectorization and converting the text into numerical data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Accuracy calculation on training and testing data
training_accuracy(model, X_train, Y_train)
testing_accuracy(model, X_test, Y_test)
