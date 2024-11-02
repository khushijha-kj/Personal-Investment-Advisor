# stock_utils.py

import yfinance as yf
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import tweepy
import praw
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import (TWITTER_CONSUMER_KEY,TWITTER_BEARER_ACCESS_TOKEN, TWITTER_CONSUMER_SECRET, 
                    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
                    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
                    NEWS_API_KEY)




def analyze_twitter_sentiment(symbol):
    """Analyze sentiment from Twitter."""
    # auth = tweepy.OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
    # auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET)
    # auth = tweepy.OAuth2BearerHandler(TWITTER_BEARER_ACCESS_TOKEN)
    # api = tweepy.API(auth)
    return []
    api=tweepy.Client(consumer_key=TWITTER_CONSUMER_KEY, 
    consumer_secret=TWITTER_CONSUMER_SECRET,
     access_token=TWITTER_ACCESS_TOKEN,
      access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
      bearer_token=TWITTER_BEARER_ACCESS_TOKEN)
    # api=tweepy.Client(TWITTER_BEARER_ACCESS_TOKEN)
    # api = tweepy.API(auth, wait_on_rate_limit=True)
    vader_analyzer = SentimentIntensityAnalyzer()
    tweets = api.search_recent_tweets(query=symbol, count=25, lang='en')
    tweets_sentiments=[]
    for tweet in tweets:
        sentiment=TextBlob(tweet.text).sentiment.polarity
        vader_sentiment = vader_analyzer.polarity_scores(tweet.text)['compound']
        tweets_sentiments.append({
            "text": tweet.text,
            "sentiment": sentiment,
            "vader_sentiment": vader_sentiment,
            "date": tweet.created_at,
            "url": tweet.url
        })
        

    
    return sum(sentiments) / len(sentiments) if sentiments else 0

def analyze_reddit_sentiment(symbol):
    """Analyze sentiment from Reddit."""
    reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                         client_secret=REDDIT_CLIENT_SECRET,
                         user_agent=REDDIT_USER_AGENT)

    subreddit = reddit.subreddit('all')
    posts = subreddit.search(symbol, limit=25, time_filter='month', sort='hot')
    # sentiments = [TextBlob(post.title).sentiment.polarity for post in posts]
    reddit_posts=[]
    vader_analyzer = SentimentIntensityAnalyzer()
    for post in posts:
        post.comments.replace_more(limit=0)
        comments=post.comment_sort="top"
        comments = post.comments.list()
        sentiments = [TextBlob(comment.body).sentiment.polarity for comment in comments]
        #remove 0 sentiment
        sentiments = [s for s in sentiments if s != 0]
        overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        vader_sentiments_comments = [vader_analyzer.polarity_scores(comment.body)['compound'] for comment in comments]
        #remove 0 sentiment
        vader_sentiments_comments = [s for s in vader_sentiments_comments if s != 0]
        vader_sentiment_comments = sum(vader_sentiments_comments) / len(vader_sentiments_comments) if vader_sentiments_comments else 0

        post_sentiment=TextBlob(post.title +'\n'+ post.selftext).sentiment.polarity
        post_vader_sentiment=vader_analyzer.polarity_scores(post.title +'\n'+ post.selftext)['compound']
        reddit_posts.append({
            'title': post.title,
            'url': post.url,
            'date': post.created_utc,
            'selftext': post.selftext,
            'sentiment': overall_sentiment,
            'vader_sentiment': vader_sentiment_comments,
            'post_sentiment': post_sentiment,
            'post_vader_sentiment': post_vader_sentiment,
            'link': post.url

        })
    return reddit_posts


def get_news_and_sentiment(symbol, days):
    """Fetch news articles and perform sentiment analysis."""
    url = f"https://newsapi.org/v2/everything?q={symbol}&from={days}daysAgo&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    
    response = requests.get(url)
    if response.status_code != 200:
        return {'error': f"Failed to fetch news data: {response.status_code}"}

    news_data = response.json()
    import json
    with open('news_data.json', 'w') as f:
        json.dump(news_data, f)
    analyzed_news = []
    vader_analyzer = SentimentIntensityAnalyzer()
    for article in news_data.get('articles', []):
        sentiment = TextBlob(article['content']).sentiment.polarity
        vader_sentiment = vader_analyzer.polarity_scores(article['content'])['compound']
        analyzed_news.append({
            'title': article['title'],
            'url': article['url'],
            'content': article['content'],
            'source': article['source']['name'],
            'date': article['publishedAt'],
            'sentiment': sentiment,
            'vader_sentiment': vader_sentiment
        })

    return analyzed_news

if __name__ == '__main__':
    print(get_news_and_sentiment('AAPL', 30))