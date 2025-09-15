# 🏆 Universal Stock Analyzer

**Professional-grade stock analysis system with AI-powered predictions, comprehensive technical analysis, and interactive web interface**

## 🚀 **Quick Start**

### **Command Line Analysis**
```bash
./analyze AAPL
./analyze TSLA
./analyze NVDA
```

### **Web Interface**
```bash
./analyze --web
```
Then open browser to `http://localhost:8501`

## 📊 **What You Get**

### **🤖 AI-Powered Predictions**
- **Multi-Model Ensemble**: RandomForest + GradientBoosting + XGBoost
- **Probability Predictions**: "70% probability of +5-10% in 1 month"
- **Confidence Intervals**: Probability bands instead of hard BUY/SELL
- **SHAP Explanations**: Explainable AI with feature importance
- **Scenario Analysis**: Bullish, Neutral, Bearish probabilities
- **Model Performance**: Accuracy metrics for each model

### **💰 Comprehensive Fundamental Analysis**
- **Valuation Ratios**: P/E, Forward P/E, PEG, Price-to-Book, Price-to-Sales
- **Growth Metrics**: Revenue growth, earnings growth, ROE, profit margins
- **Risk Metrics**: Beta, debt-to-equity, current ratio, quick ratio
- **Analyst Data**: Target price, recommendation consensus
- **Market Metrics**: Market cap, enterprise value, 52-week high/low

### **😊 Enhanced Sentiment Analysis**
- **Multi-Source Sentiment**: News + social media + analyst ratings
- **Weighted Analysis**: Impact-weighted sentiment scoring
- **News Sources**: Yahoo Finance, MarketWatch, Bloomberg, CNBC, Benzinga
- **Social Media**: Twitter, Reddit, StockTwits integration
- **Confidence Scoring**: Sentiment confidence levels

### **📈 Advanced Technical Analysis**
- **20+ Technical Indicators**: RSI, MACD, Stochastic, Williams %R, ATR, OBV
- **Multiple Timeframes**: 5, 10, 20, 50, 200-day averages
- **Volume Analysis**: Volume ratios, institutional flow indicators
- **Volatility Metrics**: Annual volatility, ATR, Bollinger Band position
- **Performance Metrics**: 1D, 5D, 1M, 3M, 1Y returns

### **🎯 Professional Recommendation Engine**
- **Comprehensive Scoring**: Technical + Fundamental + Sentiment + ML (7 factors)
- **Risk Assessment**: Multiple risk factors identification
- **Confidence Levels**: HIGH/MEDIUM/LOW with detailed reasoning
- **Investment Recommendation**: STRONG BUY, BUY, HOLD, SELL

### **📊 Interactive Web Interface**
- **Collapsible Sections**: Organized, space-efficient layout
- **Interactive Charts**: Plotly-powered technical analysis
- **Real-time Data**: Live market data updates
- **Professional Styling**: Clean, modern interface
- **Responsive Design**: Works on all devices

## 🎯 **Usage Examples**

### **Command Line**
```bash
# Basic analysis
./analyze AAPL

# Analysis without ML predictions
./analyze TSLA --no-ml

# Analysis without fundamentals
./analyze NVDA --no-fundamentals

# Analysis without sentiment
./analyze MSFT --no-sentiment
```

### **Web Interface**
```bash
# Launch web interface
./analyze --web
# or
./analyze web
```

## 📊 **Sample Output**

### **Command Line Output**
```
🏆 STOCK ANALYSIS: AAPL
==================================================

📊 BASIC INFORMATION
Current Price: $236.70
Price Change: +1.12%
Volume: 42,522,506
Market Cap: $3,512,722,522,112
Company: Apple Inc.
Sector: Technology
Industry: Consumer Electronics

🔧 TECHNICAL ANALYSIS
RSI (14): 61.6
MACD: 3.684
SMA 20: $231.96
SMA 50: $221.47
Volatility: 1.5%
BB Position: 65.2%

📈 PERFORMANCE
1 Day: +1.1%
5 Days: +2.3%
1 Month: +5.7%
3 Months: +12.4%
1 Year: +28.9%

💰 FUNDAMENTAL ANALYSIS
P/E Ratio: 35.86
Forward P/E: 28.45
PEG Ratio: 2.15
Price-to-Book: 45.23
Revenue Growth: 8.5%
ROE: 147.8%
Beta: 1.25
Relative Strength: +5.2%

😊 SENTIMENT ANALYSIS
Overall Sentiment: POSITIVE
Sentiment Score: +0.650
News Sentiment: +0.650
Social Sentiment: +0.300
News Count: 5
Positive Ratio: 80.0%

🤖 ML PREDICTION
Direction: BULLISH
Confidence: 65.3%
Probability Up: 65.3%
Probability Down: 34.7%
Price Target: $245.67

🎯 MODEL PERFORMANCE
RandomForest: 0.623
GradientBoosting: 0.618
XGBoost: 0.631

🔍 TOP FEATURES
• returns_3d: 0.089
• macd_histogram: 0.088
• ma5: 0.082
• price_ma50_ratio: 0.081
• macd: 0.076

🎯 RECOMMENDATION
Recommendation: STRONG BUY
Confidence: HIGH
Score: 5/7

🔍 KEY FACTORS
✅ RSI not overbought
✅ MACD bullish
✅ Above 20-day SMA
✅ Reasonable P/E ratio
✅ Outperforming market

⚠️ RISK ASSESSMENT
✅ Low risk factors identified
```

## 🏆 **Competitive Comparison**

| Feature | Our System | Bloomberg | Yahoo Finance | TradingView |
|---------|------------|-----------|---------------|-------------|
| **ML Predictions** | ✅ Multi-model ensemble | ✅ | ✅ | ✅ |
| **Probability Analysis** | ✅ | ✅ | ✅ | ✅ |
| **Fundamental Analysis** | ✅ Comprehensive | ✅ | ✅ | ❌ |
| **Sentiment Analysis** | ✅ Multi-source | ✅ | ✅ | ✅ |
| **Feature Importance** | ✅ SHAP | ✅ | ❌ | ❌ |
| **Model Performance** | ✅ | ✅ | ❌ | ❌ |
| **Command Line** | ✅ | ❌ | ❌ | ❌ |
| **Web Interface** | ✅ | ✅ | ✅ | ✅ |
| **Cost** | 🆓 Free | 💰 $2,000/mo | 💰 $35/mo | 💰 $15/mo |

## 🔧 **Installation**

### **Dependencies**
```bash
pip install -r requirements.txt
```

### **Required Packages**
- `streamlit` - Web interface
- `yfinance` - Market data
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `plotly` - Interactive charts
- `scikit-learn` - Machine learning
- `xgboost` - Advanced ML (optional)
- `requests` - Web requests
- `beautifulsoup4` - Web scraping
- `textblob` - Sentiment analysis
- `vaderSentiment` - Advanced sentiment
- `shap` - Explainable AI

## 📁 **File Structure**

```
stock_analyzer/
├── stock_analyzer.py              # 🏆 Universal analyzer (CLI + Web)
├── sidebar_web.py                 # 🌐 Enhanced web interface
├── enhanced_data_collector.py     # 📊 Multi-source data collection
├── analyze                        # 🚀 Universal launcher
├── requirements.txt               # 📦 Dependencies
├── README.md                      # 📚 This file
└── portfolio.json                 # 💼 Portfolio data (auto-created)
```

## 🎯 **Key Features**

### **✅ What Makes This Universal**
- **Single File**: One file works for both CLI and web
- **AI-Powered**: Multi-model ensemble with probability predictions
- **Comprehensive**: Technical + Fundamental + Sentiment analysis
- **Professional**: Commercial-grade UI with interactive charts
- **Explainable**: SHAP feature importance and model performance
- **Risk-Aware**: Multi-factor risk assessment
- **Free**: No subscription fees required

### **🚀 Advanced Capabilities**
- **Real-time Data**: Live market data from Yahoo Finance
- **Interactive Charts**: Plotly-powered technical analysis (web only)
- **Professional Metrics**: All standard financial ratios
- **Responsive Design**: Works on desktop and mobile (web only)
- **Color Coding**: Intuitive positive/negative indicators
- **Collapsible UI**: Space-efficient, organized interface

## 🎯 **Command Line Options**

```bash
./analyze TICKER [OPTIONS]

Options:
  --no-ml           Disable ML predictions
  --no-fundamentals  Disable fundamental analysis
  --no-sentiment    Disable sentiment analysis
  --web             Launch web interface
```

## 🌐 **Web Interface Features**

- **Collapsible Sections**: Organized, space-efficient layout
- **Interactive Charts**: Candlestick charts with technical overlays
- **Company Overview**: Combined company info, key metrics, and price chart
- **ML Predictions**: Probability visualization with SHAP explanations
- **Professional Styling**: Clean, modern interface
- **Responsive Design**: Works on all devices
- **Fast Mode**: Skip ML/backtest for quicker analysis

## 📊 **Data Sources**

### **Primary Sources**
- 📈 Yahoo Finance (Market Data)
- 📰 News APIs (Yahoo Finance, MarketWatch, Bloomberg, CNBC, Benzinga)
- 🐦 Social Media (Twitter, Reddit, StockTwits)
- 👔 Analyst Ratings (Yahoo Finance)

### **Additional Sources**
- 📊 Options Flow Data
- 🏛️ Institutional Holdings
- 🌍 Economic Indicators (VIX, Treasury Yields, Dollar Index)
- 🔍 Alternative Data (Google Trends, Web Traffic)

## ⚠️ **Disclaimer**

This analysis is for educational purposes only. Always consult with qualified financial advisors before investing. Past performance does not guarantee future results.

## 🏆 **Conclusion**

This is a **professional-grade stock analysis system** that rivals commercial platforms like Bloomberg Terminal, Yahoo Finance Pro, and TradingView - but it's completely free!

**Single file, dual interface, professional results! 🚀📈**

### **Quick Commands**
```bash
# Command line analysis
./analyze AAPL

# Web interface
./analyze --web
```

**Ready to analyze stocks like a pro! 🏆**

## 🔄 **Recent Updates**

- ✅ **Collapsible UI**: Space-efficient, organized interface
- ✅ **Enhanced Layout**: Company overview with horizontal key metrics
- ✅ **SHAP Explanations**: Explainable AI with feature importance
- ✅ **Multi-Source Data**: Enhanced data collection from multiple sources
- ✅ **Professional Styling**: Clean, modern interface
- ✅ **Fast Mode**: Quick analysis without ML/backtest
- ✅ **Comprehensive Analysis**: Technical + Fundamental + Sentiment + ML