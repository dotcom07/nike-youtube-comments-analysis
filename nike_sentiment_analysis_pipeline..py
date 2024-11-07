import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import re
import emoji
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import logging
from tqdm.asyncio import tqdm_asyncio
import asyncio

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 불용어 설정
stop_words = set(stopwords.words('english'))

# RoBERTa 모델 및 토크나이저 로드
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# CSV 파일 불러오기
csv_filename = "/content/drive/MyDrive/nike_official_comments_cleaned.csv"
df = pd.read_csv(csv_filename)

# ----- VADER 감정 사전 업데이트 -----
new_words = {
    "fire": 2.0,
    "lit": 1.5,
    "dope": 2.0,
    "sick": 2.0,
    "beast": 1.8,
    "epic": 2.2,
    "savage": 1.7,
    "awesome": 3.0,
    "amazing": 2.5,
    "love": 3.0,
    "great": 2.0,
    "fantastic": 2.5,
    "cool": 1.5,
    "best": 3.0,
    "wow": 2.0,
    "favorite": 2.5,
    "fun": 1.5,
    "hilarious": 2.5,
    "excellent": 2.5,
    "beautiful": 2.5,
    "perfect": 3.0,
    "superb": 2.5,
    "chill": 1.2,
    "crazy": 1.5,
    "insane": 1.8,
    "legend": 2.0,
    "legendary": 2.5,
    "unstoppable": 2.2,
    "phenomenal": 2.8,
    "mindblowing": 2.5,
    "stunning": 2.4,
    "rockstar": 2.0
}
# -------------------------------------

# 감정 분석기 초기화
analyzer = SentimentIntensityAnalyzer()
analyzer.lexicon.update(new_words)

# 특정 구문 및 브랜드 명 보호를 위한 사전
protected_phrases = ["nike by you", "nike", "by you"]

# 특정 구문 보호 함수
def protect_phrases(text):
    for phrase in protected_phrases:
        text = re.sub(r'\b{}\b'.format(re.escape(phrase)), phrase.replace(" ", "_"), text, flags=re.IGNORECASE)
    return text

# 불용어 제거 함수 (이모티콘 및 특수문자는 유지)
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# 품사 태깅을 위한 함수
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

# 표제어 추출 함수
lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)

# 텍스트 전처리 함수
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URL 제거
    text = re.sub(r'[^A-Za-z0-9\s.]+', lambda match: match.group() if emoji.is_emoji(match.group()) else '', text)  # 특수문자 제거
    text = protect_phrases(text)  # 특정 구문 보호
    text = remove_stopwords(text)  # 불용어 제거
    text = lemmatize_text(text)  # 표제어 추출
    text = re.sub(r'\.\s', ' ', text)  # 문장 끝의 마침표 제거
    return text

# 감정 분석 함수 (VADER 사용)
def analyze_sentiment_vader(text):
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict

# 감정 분석 함수 (TextBlob 사용)
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# 감정 분석 함수 (RoBERTa 사용)
def analyze_sentiment_roberta(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits[0].detach().numpy()
    scores = softmax(scores)
    return scores

# 결측치 처리 및 전처리 적용
df['cleaned_comment'] = df['comment'].fillna('')
df['cleaned_comment'] = df['cleaned_comment'].apply(preprocess_text)

# VADER 감정 점수 계산 및 데이터프레임에 추가
async def analyze_vader_async():
    vader_scores = []
    for i in tqdm_asyncio(range(len(df)), desc="VADER 감정 분석"):
        vader_scores.append(analyze_sentiment_vader(df['cleaned_comment'][i]))
    return vader_scores

# TextBlob 감정 점수 계산 및 데이터프레임에 추가
async def analyze_textblob_async():
    textblob_scores = []
    for text in tqdm_asyncio(df['cleaned_comment'], desc="TextBlob 감정 분석"):
        textblob_scores.append(analyze_sentiment_textblob(text))
    return textblob_scores

# RoBERTa 감정 점수 계산 및 데이터프레임에 추가
async def analyze_roberta_async():
    roberta_scores = []
    for i in tqdm_asyncio(range(len(df)), desc="RoBERTa 감정 분석"):
        roberta_scores.append(analyze_sentiment_roberta(df['cleaned_comment'][i]))
    return roberta_scores

# 비동기 작업 실행
async def main():
    vader_scores, textblob_scores, roberta_scores = await asyncio.gather(
        analyze_vader_async(),
        analyze_textblob_async(),
        analyze_roberta_async()
    )

    df_vader = pd.DataFrame(vader_scores, columns=['neg', 'neu', 'pos', 'compound'])
    df_textblob = pd.DataFrame(textblob_scores, columns=['textblob_score'])
    df_textblob['textblob_sentiment'] = df_textblob['textblob_score'].apply(
        lambda score: 'Positive' if score >= 0.1 else ('Negative' if score <= -0.1 else 'Neutral')
    )
    df_roberta = pd.DataFrame(roberta_scores, columns=['roberta_negative', 'roberta_neutral', 'roberta_positive'])

    df_combined = pd.concat([df, df_vader, df_textblob, df_roberta], axis=1)

    # 결합된 감정 분석 결과 계산 및 데이터프레임에 추가
    def get_ensemble_sentiment(row):
        if row['compound'] > 0.5:
            return "Positive"
        elif row['compound'] < -0.5:
            return "Negative"
        elif row['roberta_positive'] > 0.6:
            return "Positive"
        elif row['roberta_negative'] > 0.6:
            return "Negative"
        else:
            if row['textblob_sentiment'] == "Positive":
                return "Positive"
            elif row['textblob_sentiment'] == "Negative":
                return "Negative"
            else:
                return "Neutral"

    df_combined['ensemble_sentiment'] = df_combined.apply(get_ensemble_sentiment, axis=1)

    # 평균 감정 점수 계산
    average_vader_compound = df_combined['compound'].mean()
    average_textblob_score = df_combined['textblob_score'].mean()

    # RoBERTa 감정 분포 계산
    roberta_positive_distribution = df_combined['roberta_positive'].mean()
    roberta_negative_distribution = df_combined['roberta_negative'].mean()
    roberta_neutral_distribution = df_combined['roberta_neutral'].mean()

    # 감정 분석 결과를 포함하여 CSV 파일로 저장
    output_filename = "/content/drive/MyDrive/Sentiment_nike.csv"
    df_combined.to_csv(output_filename, index=False)

    # 평균 감정 점수와 RoBERTa 분포를 별도의 텍스트 파일로 저장
    with open("average_sentiment_scores.txt", "w") as file:
        file.write(f"Average VADER Compound Score: {average_vader_compound}\n")
        file.write(f"Average TextBlob Score: {average_textblob_score}\n")
        file.write(f"RoBERTa Positive Distribution: {roberta_positive_distribution}\n")
        file.write(f"RoBERTa Negative Distribution: {roberta_negative_distribution}\n")
        file.write(f"RoBERTa Neutral Distribution: {roberta_neutral_distribution}\n")

    print("Saved sentiment analysis results and average sentiment scores to /content/drive/MyDrive/Sentiment_nike and average_sentiment_scores.txt")

await main()
