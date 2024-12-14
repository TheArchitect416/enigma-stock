# sentiment.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import datetime
from typing import List, Tuple, Optional

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sia = SentimentIntensityAnalyzer()


def fetch_company_name(ticker: str) -> Optional[str]:
    """
    Fetches the company name for a given stock ticker from NSE India.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'RELIANCE').

    Returns:
        Optional[str]: Company name if found, else None.
    """
    url = f"https://www.nseindia.com/get-quotes/equity?symbol={ticker}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    session = requests.Session()
    session.headers.update(headers)

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Find company name in the HTML structure
        company_name_tag = soup.find("span", {"id": "securityName"})
        if company_name_tag:
            return company_name_tag.text.strip()
        else:
            return None
    except Exception as e:
        print(f"Error fetching company name: {e}")
        return None


def fetch_news_headlines(ticker: str, company_name: Optional[str], limit: int = 15) -> List[str]:
    """
    Fetches the latest news headlines for a given stock ticker from MoneyControl.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'RELIANCE').
        company_name (Optional[str]): Company name corresponding to the ticker.
        limit (int): Number of headlines to fetch.

    Returns:
        List[str]: A list of relevant news headlines.
    """
    news_url = f"https://www.moneycontrol.com/news/tags/{ticker.lower()}.html"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(news_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Find news articles based on the current structure of MoneyControl
        articles = soup.find_all('a', class_='news-link')  # Extract news article links

        news_data = []
        # Define keywords for filtering
        if company_name:
            keywords = [ticker, company_name]
        else:
            keywords = [ticker]

        for article in articles:
            title = article.get_text(strip=True)
            if any(keyword.lower() in title.lower() for keyword in keywords):
                news_data.append(title)
                if len(news_data) >= limit:
                    break

        # Fallback if not enough headlines found
        if len(news_data) < limit:
            additional_articles = soup.find_all('a', href=True)
            for article in additional_articles:
                title = article.get_text(strip=True)
                if any(keyword.lower() in title.lower() for keyword in keywords) and title not in news_data:
                    news_data.append(title)
                if len(news_data) >= limit:
                    break

        return news_data[:limit]
    except Exception as e:
        print(f"Error fetching news headlines: {e}")
        return []


def perform_sentiment_analysis(headlines: List[str]) -> pd.DataFrame:
    """
    Performs sentiment analysis on a list of news headlines.

    Args:
        headlines (List[str]): List of news headlines.

    Returns:
        pd.DataFrame: DataFrame containing headlines and their sentiment scores.
    """
    sentiment_data = []
    for headline in headlines:
        score = sia.polarity_scores(headline)['compound']
        sentiment_data.append({'headline': headline, 'score': score})

    sentiment_df = pd.DataFrame(sentiment_data)
    return sentiment_df


def train_sentiment_model(sentiment_df: pd.DataFrame) -> Tuple[Optional[LinearRegression], Optional[float]]:
    """
    Trains a simple Linear Regression model on sentiment scores.

    Args:
        sentiment_df (pd.DataFrame): DataFrame containing sentiment scores.

    Returns:
        Tuple[Optional[LinearRegression], Optional[float]]: Trained model and R² score.
    """
    if sentiment_df.empty or len(sentiment_df) < 5:
        print("Not enough data to train the sentiment model.")
        return None, None

    sentiment_df = sentiment_df.copy()
    sentiment_df['day'] = range(1, len(sentiment_df) + 1)

    X = sentiment_df[['day']]
    y = sentiment_df['score']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    return model, r2


def predict_future_sentiment(model: LinearRegression, last_day: int, days_ahead: int = 3) -> pd.DataFrame:
    """
    Predicts future sentiment scores using the trained model.

    Args:
        model (LinearRegression): Trained Linear Regression model.
        last_day (int): The last day number from the training data.
        days_ahead (int): Number of days to predict.

    Returns:
        pd.DataFrame: DataFrame containing predicted dates and their sentiment scores.
    """
    future_days = pd.DataFrame({'day': range(last_day + 1, last_day + days_ahead + 1)})
    predictions = model.predict(future_days)

    # Assuming today's date as the last date, calculate future dates
    last_date = datetime.date.today()
    prediction_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days_ahead + 1)]

    prediction_df = pd.DataFrame({
        'date': prediction_dates,
        'predicted_score': predictions
    })

    return prediction_df


def plot_sentiment(sentiment_df: pd.DataFrame, prediction_df: pd.DataFrame) -> go.Figure:
    """
    Plots historical and predicted sentiment scores.

    Args:
        sentiment_df (pd.DataFrame): DataFrame with historical sentiment scores.
        prediction_df (pd.DataFrame): DataFrame with predicted sentiment scores.

    Returns:
        go.Figure: Plotly figure object containing the sentiment plot.
    """
    fig = go.Figure()

    # Historical Sentiment
    fig.add_trace(go.Scatter(
        x=sentiment_df.index + 1,
        y=sentiment_df['score'],
        mode='lines+markers',
        name='Historical Sentiment',
        line=dict(color='blue')
    ))

    # Predicted Sentiment
    if not prediction_df.empty:
        fig.add_trace(go.Scatter(
            x=prediction_df['date'],
            y=prediction_df['predicted_score'],
            mode='lines+markers',
            name='Predicted Sentiment',
            line=dict(color='orange', dash='dash')
        ))

    fig.update_layout(
        title="Sentiment Analysis & Prediction",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        template="plotly_dark"
    )

    return fig
