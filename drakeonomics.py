# DRAKEconomics - Rap Lyrics Economic Sentiment Analysis
# A system to analyze rap lyrics sentiment and correlate with economic indicators

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from datetime import datetime, timedelta
import time
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
GENIUS_API_KEY = "your_genius_api_key_here"  # Replace with actual API key
GENIUS_BASE_URL = "https://api.genius.com"

class LyricsAnalyzer:
    def __init__(self):
        """Initialize the sentiment analysis pipeline"""
        try:
            # Use a financial sentiment model for better economic relevance
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except:
            # Fallback to general sentiment model
            self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def clean_lyrics(self, lyrics: str) -> str:
        """Clean and preprocess lyrics text"""
        # Remove unwanted patterns
        lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Verse 1], [Chorus], etc.
        lyrics = re.sub(r'\(.*?\)', '', lyrics)  # Remove parentheses
        lyrics = re.sub(r'http\S+', '', lyrics)  # Remove URLs
        lyrics = re.sub(r'[^\w\s]', ' ', lyrics)  # Remove special characters
        lyrics = ' '.join(lyrics.split())  # Remove extra whitespace
        return lyrics.lower()
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text and return scores"""
        try:
            # Split long text into chunks to avoid token limits
            max_length = 512
            words = text.split()
            chunks = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]
            
            sentiments = []
            for chunk in chunks[:5]:  # Limit to first 5 chunks
                if len(chunk.strip()) > 10:
                    result = self.sentiment_analyzer(chunk)[0]
                    sentiments.append(result)
            
            if not sentiments:
                return {'label': 'NEUTRAL', 'score': 0.5}
            
            # Average the sentiment scores
            if sentiments[0]['label'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                # Standard sentiment labels
                pos_scores = [s['score'] for s in sentiments if s['label'] == 'POSITIVE']
                neg_scores = [s['score'] for s in sentiments if s['label'] == 'NEGATIVE']
                
                avg_pos = np.mean(pos_scores) if pos_scores else 0
                avg_neg = np.mean(neg_scores) if neg_scores else 0
                
                if avg_pos > avg_neg:
                    return {'label': 'POSITIVE', 'score': avg_pos}
                else:
                    return {'label': 'NEGATIVE', 'score': avg_neg}
            else:
                # Financial sentiment labels (positive, negative, neutral)
                avg_score = np.mean([s['score'] for s in sentiments])
                return {'label': sentiments[0]['label'], 'score': avg_score}
                
        except Exception as e:
            st.error(f"Sentiment analysis error: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5}

class EconomicDataFetcher:
    def __init__(self):
        pass
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def get_economic_indicators(self) -> Dict:
        """Get various economic indicators"""
        indicators = {}
        
        # S&P 500
        sp500 = self.get_stock_data("^GSPC", "2y")
        if not sp500.empty:
            indicators['SP500'] = sp500['Close']
        
        # VIX (Volatility Index)
        vix = self.get_stock_data("^VIX", "2y")
        if not vix.empty:
            indicators['VIX'] = vix['Close']
        
        # Dollar Index
        dxy = self.get_stock_data("DX-Y.NYB", "2y")
        if not dxy.empty:
            indicators['DXY'] = dxy['Close']
        
        return indicators

class MockLyricsData:
    """Mock lyrics data generator for demonstration purposes"""
    
    def __init__(self):
        # Sample themes and sentiment indicators commonly found in rap
        self.positive_themes = [
            "success money wealth prosperity growth winning celebration achievement",
            "investing business entrepreneurship hustle grind dedication motivation",
            "luxury lifestyle expensive cars jewelry fashion status symbols",
            "family love relationships happiness joy celebration life"
        ]
        
        self.negative_themes = [
            "struggle poverty hardship difficulties challenges obstacles pain",
            "economy recession inflation prices rising costs expensive",
            "unemployment jobless work stress financial pressure debt",
            "system inequality injustice corruption politics broken promises"
        ]
        
        self.neutral_themes = [
            "everyday life routine normal activities simple pleasures",
            "reflection thoughts contemplation philosophy life lessons",
            "music art creativity expression passion dedication craft",
            "community neighborhood friends relationships social connections"
        ]
    
    def generate_mock_lyrics(self, sentiment: str, length: int = 100) -> str:
        """Generate mock lyrics with specified sentiment"""
        if sentiment == 'positive':
            themes = self.positive_themes
        elif sentiment == 'negative':
            themes = self.negative_themes
        else:
            themes = self.neutral_themes
        
        # Create mock lyrics by combining themes
        import random
        selected_themes = random.sample(themes, min(2, len(themes)))
        lyrics = " ".join(selected_themes)
        
        # Add some structure
        words = lyrics.split()
        mock_lyrics = " ".join(words[:length])
        
        return mock_lyrics
    
    def get_mock_data(self) -> List[Dict]:
        """Generate mock lyrics data with dates"""
        mock_data = []
        start_date = datetime.now() - timedelta(days=730)  # 2 years ago
        
        # Generate data points every 30 days
        for i in range(24):
            date = start_date + timedelta(days=i*30)
            
            # Simulate economic sentiment trends
            if i < 8:  # First 8 months - positive sentiment
                sentiment = 'positive'
                score_base = 0.7
            elif i < 16:  # Middle 8 months - declining sentiment
                sentiment = 'negative'
                score_base = 0.3
            else:  # Last 8 months - recovering sentiment
                sentiment = 'positive'
                score_base = 0.6
            
            # Add some noise
            import random
            score = score_base + random.uniform(-0.2, 0.2)
            score = max(0.1, min(0.9, score))  # Clamp between 0.1 and 0.9
            
            mock_data.append({
                'date': date,
                'artist': f'Artist_{i%5}',
                'song': f'Song_{i}',
                'lyrics': self.generate_mock_lyrics(sentiment),
                'sentiment': sentiment.upper(),
                'score': score
            })
        
        return mock_data

def main():
    st.set_page_config(
        page_title="DRAKEconomics",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 DRAKEconomics")
    st.subheader("Rap Lyrics as Economic Sentiment Indicators")
    
    # Initialize components
    lyrics_analyzer = LyricsAnalyzer()
    econ_fetcher = EconomicDataFetcher()
    mock_data_generator = MockLyricsData()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Mock Data (Demo)", "Live Analysis (Requires API Keys)"]
    )
    
    if data_source == "Live Analysis (Requires API Keys)":
        st.sidebar.warning("⚠️ Live analysis requires Genius API key and actual lyrics data.")
        genius_api_key = st.sidebar.text_input("Genius API Key", type="password")
        if not genius_api_key:
            st.error("Please provide a Genius API key for live analysis.")
            return
    
    # Economic indicators selection
    st.sidebar.subheader("Economic Indicators")
    show_sp500 = st.sidebar.checkbox("S&P 500", value=True)
    show_vix = st.sidebar.checkbox("VIX (Volatility)", value=True)
    show_dxy = st.sidebar.checkbox("Dollar Index", value=False)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    correlation_window = st.sidebar.slider("Correlation Window (days)", 30, 180, 60)
    
    # Main analysis
    if st.button("🚀 Run DRAKEconomics Analysis"):
        with st.spinner("Analyzing rap lyrics and economic data..."):
            
            # Get lyrics data
            if data_source == "Mock Data (Demo)":
                lyrics_data = mock_data_generator.get_mock_data()
            else:
                st.error("Live lyrics fetching not implemented in this demo.")
                return
            
            # Convert to DataFrame
            df_lyrics = pd.DataFrame(lyrics_data)
            df_lyrics['date'] = pd.to_datetime(df_lyrics['date'])
            df_lyrics = df_lyrics.sort_values('date')
            
            # Get economic data
            econ_data = econ_fetcher.get_economic_indicators()
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Lyrics Sentiment Over Time")
                
                # Convert sentiment to numeric
                df_lyrics['sentiment_numeric'] = df_lyrics['score'] * \
                    df_lyrics['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0})
                
                fig_sentiment = px.line(
                    df_lyrics, 
                    x='date', 
                    y='sentiment_numeric',
                    title="Rap Lyrics Sentiment Trend",
                    labels={'sentiment_numeric': 'Sentiment Score', 'date': 'Date'}
                )
                fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                st.subheader("💹 Economic Indicators")
                
                fig_econ = make_subplots(specs=[[{"secondary_y": True}]])
                
                if show_sp500 and 'SP500' in econ_data:
                    fig_econ.add_trace(
                        go.Scatter(
                            x=econ_data['SP500'].index,
                            y=econ_data['SP500'].values,
                            name="S&P 500",
                            line=dict(color='green')
                        ),
                        secondary_y=False
                    )
                
                if show_vix and 'VIX' in econ_data:
                    fig_econ.add_trace(
                        go.Scatter(
                            x=econ_data['VIX'].index,
                            y=econ_data['VIX'].values,
                            name="VIX",
                            line=dict(color='red')
                        ),
                        secondary_y=True
                    )
                
                fig_econ.update_layout(title="Economic Indicators")
                fig_econ.update_yaxes(title_text="S&P 500", secondary_y=False)
                fig_econ.update_yaxes(title_text="VIX", secondary_y=True)
                
                st.plotly_chart(fig_econ, use_container_width=True)
            
            # Correlation analysis
            st.subheader("🔗 Correlation Analysis")
            
            # Prepare data for correlation
            if 'SP500' in econ_data:
                # Resample lyrics sentiment to daily frequency
                df_daily = df_lyrics.set_index('date').resample('D')['sentiment_numeric'].mean().fillna(method='ffill')
                
                # Align with economic data
                common_dates = df_daily.index.intersection(econ_data['SP500'].index)
                if len(common_dates) > 30:
                    sentiment_aligned = df_daily.loc[common_dates]
                    sp500_aligned = econ_data['SP500'].loc[common_dates]
                    
                    correlation = sentiment_aligned.corr(sp500_aligned)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sentiment-SP500 Correlation", f"{correlation:.3f}")
                    with col2:
                        st.metric("Data Points", len(common_dates))
                    with col3:
                        interpretation = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
                        st.metric("Correlation Strength", interpretation)
                    
                    # Scatter plot
                    fig_corr = px.scatter(
                        x=sentiment_aligned.values,
                        y=sp500_aligned.values,
                        title=f"Lyrics Sentiment vs S&P 500 (r = {correlation:.3f})",
                        labels={'x': 'Lyrics Sentiment', 'y': 'S&P 500'},
                        trendline="ols"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write("**Lyrics Sentiment Stats**")
                st.write(f"- Average Sentiment: {df_lyrics['sentiment_numeric'].mean():.3f}")
                st.write(f"- Sentiment Volatility: {df_lyrics['sentiment_numeric'].std():.3f}")
                st.write(f"- Positive Songs: {(df_lyrics['sentiment'] == 'POSITIVE').sum()}")
                st.write(f"- Negative Songs: {(df_lyrics['sentiment'] == 'NEGATIVE').sum()}")
            
            with stats_col2:
                if 'SP500' in econ_data:
                    returns = econ_data['SP500'].pct_change().dropna()
                    st.write("**S&P 500 Stats**")
                    st.write(f"- Average Return: {returns.mean()*100:.2f}%")
                    st.write(f"- Volatility: {returns.std()*100:.2f}%")
                    st.write(f"- Current Level: {econ_data['SP500'].iloc[-1]:.0f}")
            
            # Sample lyrics display
            st.subheader("🎵 Sample Analyzed Lyrics")
            sample_idx = len(df_lyrics) // 2
            sample = df_lyrics.iloc[sample_idx]
            
            st.write(f"**Date:** {sample['date'].strftime('%Y-%m-%d')}")
            st.write(f"**Artist:** {sample['artist']}")
            st.write(f"**Song:** {sample['song']}")
            st.write(f"**Sentiment:** {sample['sentiment']} (Score: {sample['score']:.3f})")
            
            with st.expander("View Sample Lyrics"):
                st.write(sample['lyrics'][:500] + "..." if len(sample['lyrics']) > 500 else sample['lyrics'])
            
            # Methodology
            with st.expander("📚 Methodology"):
                st.write("""
                **DRAKEconomics Methodology:**
                
                1. **Data Collection**: Lyrics are collected from popular rap artists using the Genius API
                2. **Text Preprocessing**: Lyrics are cleaned by removing metadata, special characters, and normalizing text
                3. **Sentiment Analysis**: Using FinBERT (Financial BERT) model for economic sentiment classification
                4. **Economic Data**: Financial indicators fetched from Yahoo Finance API
                5. **Correlation Analysis**: Pearson correlation between sentiment scores and economic indicators
                6. **Temporal Analysis**: Time-series analysis to identify trends and patterns
                
                **Limitations:**
                - Correlation does not imply causation
                - Sample size and artist selection bias
                - Sentiment models may not capture all nuances
                - Economic indicators have multiple influencing factors
                """)

if __name__ == "__main__":
    main()