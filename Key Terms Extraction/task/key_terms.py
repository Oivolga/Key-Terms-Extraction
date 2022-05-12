import string
from lxml import etree
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

docs = []
nltk.download('averaged_perceptron_tagger')
root = etree.parse("news.xml").getroot()
for x in range(0, len(root[0])):
    text_news = root[0][x][1].text
    tokens = nltk.tokenize.word_tokenize(text_news.lower())
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(i) for i in tokens]
    tokens_1 = []
    for word in tokens:
        if word not in stopwords.words('english'):
            tokens_1.append(word)

    for token in tokens_1:
        if token in list(string.punctuation):
            tokens_1.remove(token)

    tokens_2 = ''
    for word in tokens_1:
        if nltk.pos_tag([word])[0][1] == 'NN':
            tokens_2 += word + ' '
    docs.append(tokens_2)

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(docs)
dim = matrix.shape
matrix = matrix.toarray()
terms = vectorizer.get_feature_names_out()

for row in range(0, dim[0]):
    print(root[0][row][0].text + ':')
    word_rate = []
    for column in range(0, dim[1]):
        if matrix[row][column] != 0:
            word_rate.append((terms[column], matrix[row][column]))
    sorted_words = sorted(word_rate, key=lambda x: (x[1], x[0]), reverse=True)
    print(sorted_words[0][0], sorted_words[1][0], sorted_words[2][0], sorted_words[3][0], sorted_words[4][0])
    print()
