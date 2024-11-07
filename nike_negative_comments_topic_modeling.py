import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# 데이터 불러오기
csv_filename = '/content/drive/MyDrive/negative_sentiments.csv'
df = pd.read_csv(csv_filename)

df['cleaned_comment'] = df['cleaned_comment'].astype(str)


# LDA 모델
texts = [comment.split() for comment in df['cleaned_comment']]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# 토픽 출력
print("LDA 모델 토픽:")
for idx, topic in lda_model.print_topics(-1):
    print(f'Topic: {idx} \nWords: {topic}')

# NMF 모델
vectorizer = TfidfVectorizer(max_features=1000)
tfidf = vectorizer.fit_transform(df['cleaned_comment'])
nmf_model = NMF(n_components=5)
nmf_topics = nmf_model.fit_transform(tfidf)

print("\nNMF 모델 토픽:")
for idx, topic in enumerate(nmf_model.components_):
    print(f"Topic {idx}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

