from fastapi import FastAPI
from googleapiclient.discovery import build
from transformers import pipeline
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from typing import List, Dict
import io
import base64

nltk.download('punkt')

API_KEY = "AIzaSyANE1eR9kDd0Ng04w5rf3l5635vkTyyqa8"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

app = FastAPI()

# Initialize Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

def clean_text(text: str) -> str:
    """ Clean text by removing links and extra spaces """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def fetch_comments(video_id: str, max_results: int = 100) -> List[str]:
    """ Fetch comments from YouTube """
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        ).execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(clean_text(comment))
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments

def analyze_comments(comments: List[str]) -> Dict:
    """ Perform sentiment analysis on comments """
    total_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    sentiment_scores = []

    for comment in comments:
        result = sentiment_analyzer(comment)[0]
        sentiment_scores.append(result['score'])
        
        if result["label"] == "POSITIVE":
            total_sentiments["positive"] += 1
        elif result["label"] == "NEGATIVE":
            total_sentiments["negative"] += 1
        else:
            total_sentiments["neutral"] += 1

    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return total_sentiments, avg_sentiment_score

def generate_word_cloud(comments: List[str]) -> str:
    """ Generate a word cloud image and return it as base64 """
    all_words = " ".join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)
    
    # Save image to a BytesIO object
    img_io = io.BytesIO()
    wordcloud.to_image().save(img_io, format='PNG')
    img_io.seek(0)
    
    # Convert image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_base64

@app.get("/analyze/{video_id}")
async def analyze_video(video_id: str, max_results: int = 100):
    """ Analyze YouTube video comments and provide sentiment and word cloud """
    comments = fetch_comments(video_id, max_results)
    if not comments:
        return {"message": "No comments found or failed to fetch comments."}

    # Analyze sentiments
    sentiments, avg_sentiment_score = analyze_comments(comments)
    
    # Generate word cloud image
    word_cloud_image = generate_word_cloud(comments)
    
    # Return results
    return {
        "total_comments": len(comments),
        "sentiments": sentiments,
        "average_sentiment_score": avg_sentiment_score,
        "word_cloud_image": word_cloud_image
    }
