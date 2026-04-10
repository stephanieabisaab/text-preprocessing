
# Lab 2 Text Preprocessing
# Stephanie Abi Saab

# !pip install nltk spacy
# !python -m spacy download en_core_web_sm

import nltk
import spacy
import re

from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm')

print("Setup done")



# PART 1 

# Basic tokenization
text = "Natural Language Processing is fascinating!"
tokens = text.split()
print("\nBasic split:", tokens)


# NLTK tokenization
text = "Hello! How are you doing today? I'm learning NLP."
print("\nWord tokenize:", word_tokenize(text))
print("Sentence tokenize:", sent_tokenize(text))


# Exercise 1.1
text = "Dr. Smith's patients can't understand why they're feeling unwell. Is it the flu?"

tokens = word_tokenize(text)
num_tokens = len(tokens)

print("\nEx 1.1 tokens:", tokens)
print("Ex 1.1 count:", num_tokens)


# spaCy tokenization
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
tokens = [token.text for token in doc]
print("\nspaCy tokens:", tokens)


# Exercise 1.2
text = "The quick, brown fox jumps over the lazy dog! Isn't it amazing?"
doc = nlp(text)

clean_tokens = [t.text for t in doc if not t.is_punct and not t.is_space]

print("\nEx 1.2 clean tokens:", clean_tokens)


#Exercise 1.3
text = "I have 3 cats and 2 dogs! Their names are Max123 and Bella."

tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
tokens = tokenizer.tokenize(text)

print("\nEx 1.3 alphabetic tokens:", tokens)



# PART 2 

# Stemming
porter = PorterStemmer()
print("\nStemming example:", porter.stem("running"))


# Exercise 2.1
sentence = "The cats are running and jumping over the sleeping dogs"

tokens = word_tokenize(sentence.lower())
stemmed = [porter.stem(t) for t in tokens]

print("\nEx 2.1 stemmed:", stemmed)


#Lemmatization
lemmatizer = WordNetLemmatizer()
print("\nLemmatization example:", lemmatizer.lemmatize("running", pos="v"))


#Exercise 2.2
words = [
    ("flying", "v"),
    ("happily", "r"),
    ("worse", "a"),
    ("mice", "n"),
    ("are", "v"),
]

results = [lemmatizer.lemmatize(w, pos=p) for w, p in words]

print("\nEx 2.2 lemmas:", results)


#  Exercise 2.3
text = "The dogs were barking loudly at the cats who were climbing the trees."

doc = nlp(text)
lemmas = [t.lemma_.lower() for t in doc if not t.is_punct]

print("\nEx 2.3 lemmas:", lemmas)



# PART 3

# Stop words removal
stop_words = set(stopwords.words('english'))

text = "This is a sample sentence showing the removal of stop words from the text"

tokens = word_tokenize(text.lower())
filtered = [t for t in tokens if t not in stop_words]

print("\nStop words removed:", filtered)


#  Exercice 3.2
text = "Check out @OpenAI's new model! https://openai.com #AI #MachineLearning It's amazing!!!"

clean = text
clean = re.sub(r'https?://\S+', '', clean)
clean = re.sub(r'@\w+\'s', '', clean)
clean = re.sub(r'#\w+', '', clean)
clean = re.sub(r'[^\w\s]', '', clean)
clean = clean.lower()
clean = re.sub(r'\s+', ' ', clean).strip()

print("\nEx 3.2 cleaned text:", clean)



#   PART 4 

# Exercise 4.1
def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)

    doc = nlp(text)

    tokens = [
        t.lemma_.lower()
        for t in doc
        if not t.is_stop
        and not t.is_punct
        and not t.is_space
    ]

    tokens = [t for t in tokens if len(t) >= 2]

    return tokens


# Test
text = "Machine learning is AMAZING!!! Check https://ml.com #AI @user"
print("\nPipeline test:", preprocess_text(text))


# Exercise 4.2
docs = [
    "Machine learning is transforming the tech industry!",
    "I love Python programming",
]

processed_docs = [preprocess_text(doc) for doc in docs]

print("\nEx 4.2 processed docs:", processed_docs)
