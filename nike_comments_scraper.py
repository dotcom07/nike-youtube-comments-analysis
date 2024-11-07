#위 공식 링크들 댓글로 바꾸는 코드입니다
import asyncio
import aiohttp
from pytube import YouTube
from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm

# 텍스트 전처리 함수
def preprocess_text(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # URL 제거
    text = re.sub(r'[^A-Za-z\s]', '', text)  # 특수 문자 및 숫자 제거
    text = re.sub(r'\s+', ' ', text)  # 다중 공백 제거
    return text

# 유튜브 댓글 크롤링 함수
def download_comments(url):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(url, sort_by=0, language='en')
    comments_list = []
    for comment in comments:
        comments_list.append(comment['text'])
    return comments_list

# 유튜브 동영상 메타데이터 및 댓글 크롤링 함수
def get_video_metadata(url):
    yt = YouTube(url)
    data = {
        'title': yt.title,
        'length': yt.length,
        'author': yt.author,
        'publish_date': yt.publish_date,
        'views': yt.views,
        'keywords': yt.keywords,
        'description': yt.description
    }
    return data

def get_video_data(url):
    comments = download_comments(url)
    return comments

def process_video(url):
    try:
        video_metadata = get_video_metadata(url)
        comments = get_video_data(url)

        df = pd.DataFrame(comments, columns=['comment'])
        df['cleaned_comment'] = df['comment'].apply(preprocess_text)

        # 메타데이터 추가
        df['title'] = video_metadata['title']
        df['length'] = video_metadata['length']
        df['author'] = video_metadata['author']
        df['publish_date'] = video_metadata['publish_date']
        df['views'] = video_metadata['views']
        df['keywords'] = ','.join(video_metadata['keywords']) if video_metadata['keywords'] else ''
        df['description'] = video_metadata['description']

        return df
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None

async def main(urls):
    loop = asyncio.get_event_loop()  # 현재 이벤트 루프 가져오기
    with ThreadPoolExecutor(max_workers=10) as executor:
        tasks = [loop.run_in_executor(executor, process_video, url) for url in urls]

        # tqdm을 사용하여 진행도 표시
        results = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing videos"):
            result = await task
            if result is not None:
                results.append(result)

        all_comments_df = pd.concat(results, ignore_index=True)

        if not all_comments_df.empty:
            all_comments_df.to_csv('/content/drive/MyDrive/nike_official_comments.csv', index=False)
            print(f"Saved all comments and metadata to nike_official_comments.csv")
        else:
            print("No data to save.")

# CSV 파일에서 동영상 URL을 읽기
df_n = pd.read_csv('/content/drive/MyDrive/nike_official_202001_202405.csv')

# URL 컬럼을 리스트로 변환
video_urls = df_n['url'].tolist()

# 메인 함수 실행 (Colab에서는 asyncio.run() 대신 아래와 같이 실행)
await main(video_urls)