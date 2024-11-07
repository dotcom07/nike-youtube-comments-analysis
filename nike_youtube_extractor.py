#nike 공식 채널 영상 url들 youtube api로 url 뽑는 코드입니다

import os
import pandas as pd
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# youtube API 키 설정
#https://console.cloud.google.com/apis/api/youtube.googleapis.com
api_key = 
channel_id = 'UCUFgkRb0ZHc4Rpq15VRCICA'  # Nike 공식 채널 ID

# YouTube API 클라이언트 빌드
youtube = build('youtube', 'v3', developerKey=api_key)

def get_channel_videos_by_month(channel_id, year, month, max_results=500):
    video_details = []
    next_page_token = None

    start_date = f'{year}-{month:02}-01T00:00:00Z'
    end_date = f'{year}-{month+1:02}-01T00:00:00Z' if month < 12 else f'{year+1}-01-01T00:00:00Z'

    while len(video_details) < max_results:
        try:
            search_response = youtube.search().list(
                channelId=channel_id,
                part='snippet',
                type='video',
                publishedAfter=start_date,
                publishedBefore=end_date,
                maxResults=min(50, max_results - len(video_details)),
                pageToken=next_page_token
            ).execute()

            video_ids = [item['id']['videoId'] for item in search_response['items']]
            next_page_token = search_response.get('nextPageToken')

            if not video_ids:
                break

            video_response = youtube.videos().list(
                part='statistics',
                id=','.join(video_ids)
            ).execute()

            for item in video_response['items']:
                video_details.append({
                    'video_id': item['id'],
                    'viewCount': int(item['statistics'].get('viewCount', 0)),
                    'commentCount': int(item['statistics'].get('commentCount', 0)),
                    'url': f'https://www.youtube.com/watch?v={item["id"]}'
                })

            if not next_page_token:
                break

            time.sleep(3)

        except HttpError as e:
            print(f"An error occurred: {e}")
            time.sleep(10)
            continue

    return video_details


def get_all_videos_by_month(channel_id, start_year, start_month, end_year, end_month, max_results_per_month=500):
    all_videos = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            print(f"Fetching videos for {year}년 {month}월")
            videos = get_channel_videos_by_month(channel_id, year, month, max_results_per_month)
            print(f"{year}년 {month}월: {len(videos)}개 동영상 수집 완료")
            all_videos.extend(videos)
            time.sleep(5)  # 다음 달로 넘어가기 전 5초 대기
    return all_videos

# 2020년 1월부터 2024년 5월까지 데이터 수집
start_year = 2020
start_month = 1
end_year = 2024
end_month = 5
max_results_per_month = 500 # 월별 최대 결과 수 조정 가능

all_videos = get_all_videos_by_month(channel_id, start_year, start_month, end_year, end_month, max_results_per_month)

df_all_videos = pd.DataFrame(all_videos)
df_all_videos.to_csv('nike_videos_202001_202405.csv', index=False)

print("모든 동영상 정보가 nike_videos_202001_202405.csv에 저장되었습니다.")