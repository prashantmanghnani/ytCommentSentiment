from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from transformers import pipeline
from googleapiclient.discovery import build
import nltk
import re

# Download necessary NLTK data
nltk.download('punkt')

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow requests from all origins (or specify your extension's origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your Chrome extension's URL, e.g. ["chrome-extension://your-extension-id"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Hugging Face sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# YouTube API setup
API_KEY = "AIzaSyANE1eR9kDd0Ng04w5rf3l5635vkTyyqa8"  # Replace with your API key
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

# Helper function to clean the comments
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Helper function to fetch comments
def fetch_comments(video_id, max_results=100):
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

# Helper function to analyze sentiments
def analyze_comments(comments):
    total_sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    sentiment_scores = []
    sentiments = []

    for comment in comments:
        result = sentiment_analyzer(comment)[0]
        sentiments.append(result)
        sentiment_scores.append(result['score'])

        if result["label"] == "POSITIVE":
            total_sentiments["positive"] += 1
        elif result["label"] == "NEGATIVE":
            total_sentiments["negative"] += 1
        else:
            total_sentiments["neutral"] += 1

    avg_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    return sentiments, total_sentiments, avg_sentiment_score

# Helper function to calculate metrics
def calculate_metrics(comments):
    unique_comments = set(comments)
    total_length = sum(len(comment) for comment in comments)
    avg_length = total_length / len(comments) if comments else 0
    return {
        "total_comments": len(comments),
        "unique_comments": len(unique_comments),
        "avg_length": avg_length
    }

# API endpoint to analyze video comments
@app.get("/analyze/{video_id}")
async def analyze_video(video_id: str):
    comments = fetch_comments(video_id, max_results=100)
    
    if not comments:
        return {"message": "No comments found or failed to fetch comments."}
    
    # Calculate metrics
    metrics = calculate_metrics(comments)

    # Sentiment analysis
    sentiments, sentiment_counts, avg_sentiment_score = analyze_comments(comments)

    # Return analysis results
    return {
        "message": "Analysis complete",
        "video_id": video_id,
        "total_comments": metrics["total_comments"],
        "unique_comments": metrics["unique_comments"],
        "avg_length": metrics["avg_length"],
        "sentiment_analysis": sentiment_counts,
        "avg_sentiment_score": avg_sentiment_score,
    }
