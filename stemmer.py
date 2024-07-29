import re
from nltk.corpus import stopwords


def stemming(content, port_stem):
    stemmed_content = re.sub("[^a-zA-Z]", "", content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [
        port_stem.stem(word)
        for word in stemmed_content
        if not word in stopwords.words("english")
    ]
    stemmed_content = " ".join(stemmed_content)

    return stemmed_content
