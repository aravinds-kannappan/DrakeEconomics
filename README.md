# DRAKEconomics

**Using Rap Lyrics as Proxy Indicators of Economic Sentiment**

A Python application that analyzes sentiment in rap lyrics and correlates it with economic indicators like stock market performance, volatility, and currency strength. Because sometimes the streets know the market better than Wall Street does.

---

## Overview

DRAKEonomics is a novel approach to economic sentiment analysis that leverages the cultural zeitgeist expressed in rap music.

### Key Capabilities
- Scrapes and analyzes rap lyrics for **economic sentiment**
- Correlates lyrical sentiment with **real-time financial data**
- Provides **interactive visualizations** of trends and relationships
- Offers **statistical analysis** of sentiment-market correlations

---

## Features

### Core Functionality
- **Sentiment Analysis**: Uses FinBERT for economically-relevant classification  
- **Economic Data Integration**: Real-time market data from Yahoo Finance  
- **Correlation Analysis**: Pearson & rolling correlation with market indicators  
- **Visualization**: Interactive time-series and scatter plots via Plotly  
- **Mock Data Mode**: Use without external APIs

---

### Supported Economic Indicators
- S&P 500 Index
- VIX (Volatility Index)
- US Dollar Index (DXY)  
> *Extensible for more indicators*

---

### Analytics Features
- Pearson correlation coefficients  
- Rolling window analysis  
- Sentiment volatility  
- Statistical significance testing  
- Trend recognition and interpretation

---

## Tech Stack

- **Frontend**: Streamlit  
- **NLP**: FinBERT (via HuggingFace Transformers)  
- **Financial Data**: Yahoo Finance (`yfinance`)  
- **Visualization**: Plotly  
- **Data Processing**: Pandas, NumPy  
- **Lyrics**: Genius API

---

## Installation

### Prerequisites
- Python 3.8+
- `pip` package manager

### Quick Start

```bash
git clone https://github.com/yourusername/drakeonomics.git
cd drakeonomics
pip install -r requirements.txt
streamlit run drakeonomics.py
