import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer

# Download resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Sample text
text = """
Data Analytics is the process of analyzing raw data
to find useful information. Machine learning and
artificial intelligence are important fields in data science.
"""

# Tokenization
tokens = word_tokenize(text)
print("Tokens:")
print(tokens)

# POS tagging
pos = pos_tag(tokens)
print("\nPOS Tags:")
print(pos)

# Stopword removal
stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print("\nAfter Stopword Removal:")
print(filtered_words)

# Stemming
ps = PorterStemmer()

stemmed_words = []

for word in filtered_words:
    stemmed_words.append(ps.stem(word))

print("\nStemmed Words:")
print(stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in filtered_words:
    lemmatized_words.append(
        lemmatizer.lemmatize(word)
    )

print("\nLemmatized Words:")
print(lemmatized_words)

# TF-IDF
documents = [
    "Data analytics is important",
    "Machine learning is part of data science",
    "Artificial intelligence and machine learning"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)

print("\nFeature Names:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(X.toarray())
