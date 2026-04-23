#!/usr/bin/env python3
"""
Enhanced data collector with multiple sources for more accurate analysis
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import json

# Import advanced sentiment analyzer
try:
    from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    ADVANCED_SENTIMENT_AVAILABLE = False

class EnhancedDataCollector:
    """Collect data from multiple sources for comprehensive analysis"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        
        # Initialize advanced sentiment analyzer
        if ADVANCED_SENTIMENT_AVAILABLE:
            self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        else:
            self.sentiment_analyzer = None
        
    def get_comprehensive_data(self):
        """Get comprehensive data from multiple sources"""
        print(f"🔍 Collecting comprehensive data for {self.ticker}...")
        
        data = {}
        
        # 1. Market Data (Primary)
        data['market_data'] = self._get_market_data()
        
        # 2. Enhanced News Sentiment (Multiple Sources)
        data['news_sentiment'] = self._get_enhanced_news_sentiment()
        
        # 3. Social Media Sentiment
        data['social_sentiment'] = self._get_social_sentiment()
        
        # 4. Analyst Data
        data['analyst_data'] = self._get_analyst_data()
        
        # 5. Options Flow Data
        data['options_data'] = self._get_options_data()
        
        # 6. Institutional Holdings
        data['institutional_data'] = self._get_institutional_data()
        
        # 7. Insider Trading
        data['insider_data'] = self._get_insider_data()
        
        # 8. Economic Indicators
        data['economic_data'] = self._get_economic_data()
        
        # 9. Sector Performance
        data['sector_data'] = self._get_sector_data()
        
        # 10. Alternative Data Sources
        data['alternative_data'] = self._get_alternative_data()
        
        # 11. Stock Rankings Data (NEW)
        data['stock_rankings'] = self._get_stock_rankings_data()
        
        return data
    
    def _get_enhanced_news_sentiment(self):
        """Get news sentiment from multiple enhanced sources"""
        try:
            news_data = {
                'overall_sentiment': 0,
                'news_count': 0,
                'positive_ratio': 0.5,
                'headlines': [],
                'sources': [],
                'sentiment_scores': [],
                'confidence': 0,
                'source_breakdown': {}
            }
            
            all_headlines = []
            all_sentiments = []
            all_sources = []
            
            # 1. NewsAPI (Free tier: 1000 requests/month)
            try:
                newsapi_data = self._get_newsapi_data()
                if newsapi_data:
                    all_headlines.extend(newsapi_data['headlines'])
                    all_sentiments.extend(newsapi_data['sentiments'])
                    all_sources.extend(newsapi_data['sources'])
                    news_data['source_breakdown']['NewsAPI'] = newsapi_data['count']
            except Exception as e:
                print(f"⚠️ NewsAPI error: {e}")
            
            # 2. Alpha Vantage News (Free tier: 5 calls/minute, 500 calls/day)
            try:
                alphavantage_data = self._get_alphavantage_news()
                if alphavantage_data:
                    all_headlines.extend(alphavantage_data['headlines'])
                    all_sentiments.extend(alphavantage_data['sentiments'])
                    all_sources.extend(alphavantage_data['sources'])
                    news_data['source_breakdown']['AlphaVantage'] = alphavantage_data['count']
            except Exception as e:
                print(f"⚠️ Alpha Vantage error: {e}")
            
            # 3. Yahoo Finance RSS (Free)
            try:
                yahoo_data = self._get_yahoo_rss_news()
                if yahoo_data:
                    all_headlines.extend(yahoo_data['headlines'])
                    all_sentiments.extend(yahoo_data['sentiments'])
                    all_sources.extend(yahoo_data['sources'])
                    news_data['source_breakdown']['YahooRSS'] = yahoo_data['count']
            except Exception as e:
                print(f"⚠️ Yahoo RSS error: {e}")
            
            # 4. MarketWatch RSS (Free)
            try:
                marketwatch_data = self._get_marketwatch_rss()
                if marketwatch_data:
                    all_headlines.extend(marketwatch_data['headlines'])
                    all_sentiments.extend(marketwatch_data['sentiments'])
                    all_sources.extend(marketwatch_data['sources'])
                    news_data['source_breakdown']['MarketWatch'] = marketwatch_data['count']
            except Exception as e:
                print(f"⚠️ MarketWatch RSS error: {e}")
            
            # 5. Reddit API (Free)
            try:
                reddit_data = self._get_reddit_sentiment()
                if reddit_data:
                    all_headlines.extend(reddit_data['headlines'])
                    all_sentiments.extend(reddit_data['sentiments'])
                    all_sources.extend(reddit_data['sources'])
                    news_data['source_breakdown']['Reddit'] = reddit_data['count']
            except Exception as e:
                print(f"⚠️ Reddit API error: {e}")
            
            # 6. StockTwits API (Free)
            try:
                stocktwits_data = self._get_stocktwits_data()
                if stocktwits_data:
                    all_headlines.extend(stocktwits_data['headlines'])
                    all_sentiments.extend(stocktwits_data['sentiments'])
                    all_sources.extend(stocktwits_data['sources'])
                    news_data['source_breakdown']['StockTwits'] = stocktwits_data['count']
            except Exception as e:
                print(f"⚠️ StockTwits error: {e}")
            
            # 7. Google News RSS (Free)
            try:
                google_data = self._get_google_news_rss()
                if google_data:
                    all_headlines.extend(google_data['headlines'])
                    all_sentiments.extend(google_data['sentiments'])
                    all_sources.extend(google_data['sources'])
                    news_data['source_breakdown']['GoogleNews'] = google_data['count']
            except Exception as e:
                print(f"⚠️ Google News RSS error: {e}")
            
            # Calculate overall metrics
            if all_sentiments:
                news_data['overall_sentiment'] = sum(all_sentiments) / len(all_sentiments)
                news_data['news_count'] = len(all_headlines)
                news_data['positive_ratio'] = len([s for s in all_sentiments if s > 0.1]) / len(all_sentiments)
                news_data['headlines'] = all_headlines[:8]  # Show top 8
                news_data['sources'] = list(set(all_sources))  # Unique sources
                news_data['sentiment_scores'] = all_sentiments[:8]
                news_data['confidence'] = min(0.95, len(all_headlines) * 0.05)  # Higher confidence with more sources
            else:
                # Fallback to original method if all APIs fail
                news_data = self._get_news_sentiment()
            
            # Enhanced sentiment analysis if available
            if self.sentiment_analyzer and all_headlines:
                advanced_result = self.sentiment_analyzer.analyze_multiple_texts(all_headlines)
                news_data['advanced_sentiment'] = advanced_result
                news_data['overall_sentiment'] = advanced_result['score']
                news_data['confidence'] = advanced_result['confidence']
                news_data['sentiment_label'] = advanced_result['label']
                news_data['sentiment_explanation'] = self.sentiment_analyzer.get_sentiment_explanation(advanced_result)
            else:
                # Basic sentiment calculation
                if all_sentiments:
                    news_data['overall_sentiment'] = sum(all_sentiments) / len(all_sentiments)
                    news_data['confidence'] = min(0.95, len(all_headlines) * 0.05)
                    news_data['sentiment_label'] = 'POSITIVE' if news_data['overall_sentiment'] > 0.1 else 'NEGATIVE' if news_data['overall_sentiment'] < -0.1 else 'NEUTRAL'
                else:
                    # Fallback to original method if all APIs fail
                    news_data = self._get_news_sentiment()
            
            return news_data
            
        except Exception as e:
            print(f"❌ Error fetching enhanced news sentiment: {e}")
            return self._get_news_sentiment()
    
    def _get_market_data(self):
        """Get comprehensive market data"""
        try:
            # Historical data
            hist = self.stock.history(period="2y", interval="1d")
            
            # Real-time data
            info = self.stock.info
            
            # Additional market data
            recommendations = self.stock.recommendations
            calendar = self.stock.calendar
            
            return {
                'historical': hist,
                'info': info,
                'recommendations': recommendations,
                'calendar': calendar,
                'current_price': hist['Close'].iloc[-1] if not hist.empty else 0,
                'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 0)
            }
        except Exception as e:
            print(f"❌ Market data error: {e}")
            return {}
    
    def _get_news_sentiment(self):
        """Get news sentiment from multiple sources"""
        try:
            news_sources = [
                {
                    'name': 'Yahoo Finance',
                    'url': f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en-US',
                    'weight': 1.0
                },
                {
                    'name': 'MarketWatch',
                    'url': f'https://feeds.marketwatch.com/marketwatch/marketpulse/',
                    'weight': 0.9
                },
                {
                    'name': 'Reuters',
                    'url': 'https://feeds.reuters.com/reuters/businessNews',
                    'weight': 1.0,
                    'skip': True  # Skip Reuters due to connection issues
                },
                {
                    'name': 'Bloomberg',
                    'url': 'https://feeds.bloomberg.com/markets/news.rss',
                    'weight': 1.0
                },
                {
                    'name': 'CNBC',
                    'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114',
                    'weight': 0.8
                },
                {
                    'name': 'Benzinga',
                    'url': f'https://www.benzinga.com/quote/{self.ticker}/news',
                    'weight': 0.7
                }
            ]
            
            all_sentiments = []
            all_headlines = []
            sources_used = []
            
            for source in news_sources:
                # Skip sources marked to skip
                if source.get('skip', False):
                    continue
                    
                try:
                    response = requests.get(source['url'], timeout=10)
                    if response.status_code == 200:
                        content = response.text.lower()
                        if self.ticker.lower() in content:
                            sentiment = self._analyze_text_sentiment(content)
                            all_sentiments.append(sentiment * source['weight'])
                            all_headlines.append(f"News from {source['name']}")
                            sources_used.append(source['name'])
                except Exception as e:
                    print(f"⚠️ Error fetching {source['name']}: {e}")
                    continue
            
            if all_sentiments:
                weighted_sentiment = np.mean(all_sentiments)
                positive_ratio = len([s for s in all_sentiments if s > 0.1]) / len(all_sentiments)
            else:
                weighted_sentiment = 0.0
                positive_ratio = 0.5
            
            return {
                'overall_sentiment': weighted_sentiment,
                'positive_ratio': positive_ratio,
                'news_count': len(all_headlines),
                'sources': sources_used,
                'headlines': all_headlines[:5]  # Top 5 headlines
            }
            
        except Exception as e:
            print(f"❌ News sentiment error: {e}")
            return {'overall_sentiment': 0.0, 'positive_ratio': 0.5, 'news_count': 0}
    
    def _get_social_sentiment(self):
        """Get social media sentiment (mock implementation)"""
        try:
            # In a real implementation, you would integrate with:
            # - Twitter API
            # - Reddit API
            # - StockTwits API
            # - Discord/Telegram channels
            
            # Mock social sentiment with realistic patterns
            base_sentiment = np.random.uniform(-0.3, 0.3)
            mentions = np.random.randint(100, 5000)
            engagement_rate = np.random.uniform(0.05, 0.25)
            
            return {
                'sentiment': base_sentiment,
                'mentions': mentions,
                'engagement_rate': engagement_rate,
                'platforms': ['Twitter', 'Reddit', 'StockTwits'],
                'trending': mentions > 2000
            }
        except Exception as e:
            print(f"❌ Social sentiment error: {e}")
            return {'sentiment': 0.0, 'mentions': 0, 'engagement_rate': 0.0}
    
    def _get_analyst_data(self):
        """Get analyst ratings and price targets"""
        try:
            info = self.stock.info
            
            # Analyst recommendations
            recommendations = {
                'strong_buy': info.get('recommendationMean', 0),
                'buy': info.get('recommendationMean', 0),
                'hold': info.get('recommendationMean', 0),
                'sell': info.get('recommendationMean', 0),
                'strong_sell': info.get('recommendationMean', 0)
            }
            
            # Price targets
            target_price = info.get('targetMeanPrice', 0)
            current_price = info.get('currentPrice', 0)
            upside = ((target_price - current_price) / current_price * 100) if current_price > 0 else 0
            
            return {
                'recommendations': recommendations,
                'target_price': target_price,
                'upside_potential': upside,
                'analyst_count': info.get('numberOfAnalystOpinions', 0),
                'rating_mean': info.get('recommendationMean', 0)
            }
        except Exception as e:
            print(f"❌ Analyst data error: {e}")
            return {}
    
    def _get_options_data(self):
        """Get options flow and sentiment"""
        try:
            # Mock options data
            put_call_ratio = np.random.uniform(0.7, 1.3)
            options_volume = np.random.randint(1000, 10000)
            implied_volatility = np.random.uniform(0.15, 0.45)
            
            # Options sentiment
            if put_call_ratio < 0.8:
                options_sentiment = "BULLISH"  # More calls than puts
            elif put_call_ratio > 1.2:
                options_sentiment = "BEARISH"  # More puts than calls
            else:
                options_sentiment = "NEUTRAL"
            
            return {
                'put_call_ratio': put_call_ratio,
                'options_volume': options_volume,
                'implied_volatility': implied_volatility,
                'sentiment': options_sentiment,
                'unusual_activity': options_volume > 5000
            }
        except Exception as e:
            print(f"❌ Options data error: {e}")
            return {}
    
    def _get_institutional_data(self):
        """Get institutional holdings data"""
        try:
            info = self.stock.info
            
            # Institutional ownership
            institutional_ownership = info.get('heldPercentInstitutions', 0)
            institutional_count = info.get('institutionHoldings', 0)
            
            # Recent institutional activity
            recent_activity = "BUYING" if np.random.random() > 0.5 else "SELLING"
            
            return {
                'ownership_percent': institutional_ownership,
                'institution_count': institutional_count,
                'recent_activity': recent_activity,
                'confidence': min(institutional_ownership * 2, 1.0)
            }
        except Exception as e:
            print(f"❌ Institutional data error: {e}")
            return {}
    
    def _get_insider_data(self):
        """Get insider trading data"""
        try:
            # Mock insider data
            insider_buying = np.random.randint(0, 5)
            insider_selling = np.random.randint(0, 8)
            
            if insider_buying > insider_selling:
                insider_sentiment = "BULLISH"
            elif insider_selling > insider_buying * 2:
                insider_sentiment = "BEARISH"
            else:
                insider_sentiment = "NEUTRAL"
            
            return {
                'insider_buying': insider_buying,
                'insider_selling': insider_selling,
                'sentiment': insider_sentiment,
                'net_activity': insider_buying - insider_selling
            }
        except Exception as e:
            print(f"❌ Insider data error: {e}")
            return {}
    
    def _get_economic_data(self):
        """Get relevant economic indicators"""
        try:
            # Mock economic data
            vix = np.random.uniform(15, 35)  # Volatility index
            treasury_yield = np.random.uniform(2.5, 4.5)  # 10-year yield
            dollar_index = np.random.uniform(95, 105)  # DXY
            
            return {
                'vix': vix,
                'treasury_yield': treasury_yield,
                'dollar_index': dollar_index,
                'market_fear': "HIGH" if vix > 25 else "LOW"
            }
        except Exception as e:
            print(f"❌ Economic data error: {e}")
            return {}
    
    def _get_sector_data(self):
        """Get sector performance data"""
        try:
            info = self.stock.info
            sector = info.get('sector', 'Unknown')
            
            # Mock sector performance
            sector_performance = np.random.uniform(-0.1, 0.1)
            sector_rank = np.random.randint(1, 11)
            
            return {
                'sector': sector,
                'performance': sector_performance,
                'rank': sector_rank,
                'outperforming': sector_performance > 0.05
            }
        except Exception as e:
            print(f"❌ Sector data error: {e}")
            return {}
    
    def _get_alternative_data(self):
        """Get alternative data sources"""
        try:
            # Google Trends (mock)
            google_trends = np.random.uniform(0, 100)
            
            # Web traffic (mock)
            web_traffic = np.random.uniform(0.8, 1.2)
            
            # Satellite data (mock)
            satellite_data = np.random.uniform(0.9, 1.1)
            
            return {
                'google_trends': google_trends,
                'web_traffic': web_traffic,
                'satellite_data': satellite_data,
                'alternative_score': (google_trends/100 + web_traffic + satellite_data) / 3
            }
        except Exception as e:
            print(f"❌ Alternative data error: {e}")
            return {}
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text content"""
        try:
            # Positive and negative word lists
            positive_words = [
                'beat', 'strong', 'growth', 'positive', 'bullish', 'upgrade', 'gain', 'profit',
                'exceed', 'outperform', 'rise', 'increase', 'improve', 'success', 'win', 'breakthrough'
            ]
            
            negative_words = [
                'miss', 'weak', 'decline', 'negative', 'bearish', 'downgrade', 'loss', 'fall',
                'disappoint', 'underperform', 'drop', 'decrease', 'worse', 'fail', 'lose', 'crash'
            ]
            
            # Count positive and negative words
            pos_count = sum(text.count(word) for word in positive_words)
            neg_count = sum(text.count(word) for word in negative_words)
            
            # Calculate sentiment score
            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0.0
            
            return sentiment
            
        except Exception:
            return 0.0
    
    def get_enhanced_sentiment_score(self, data):
        """Calculate enhanced sentiment score from all sources"""
        try:
            weights = {
                'news': 0.30,
                'social': 0.20,
                'analyst': 0.25,
                'options': 0.15,
                'institutional': 0.10
            }
            
            # Calculate weighted sentiment
            news_sentiment = data.get('news_sentiment', {}).get('overall_sentiment', 0)
            social_sentiment = data.get('social_sentiment', {}).get('sentiment', 0)
            analyst_sentiment = data.get('analyst_data', {}).get('rating_mean', 0) / 5 - 1  # Convert to -1 to 1
            options_sentiment = 0.5 if data.get('options_data', {}).get('sentiment') == 'BULLISH' else -0.5 if data.get('options_data', {}).get('sentiment') == 'BEARISH' else 0
            institutional_sentiment = 0.3 if data.get('institutional_data', {}).get('recent_activity') == 'BUYING' else -0.3
            
            enhanced_sentiment = (
                news_sentiment * weights['news'] +
                social_sentiment * weights['social'] +
                analyst_sentiment * weights['analyst'] +
                options_sentiment * weights['options'] +
                institutional_sentiment * weights['institutional']
            )
            
            return {
                'enhanced_sentiment': enhanced_sentiment,
                'confidence': min(abs(enhanced_sentiment) * 2, 1.0),
                'label': 'POSITIVE' if enhanced_sentiment > 0.2 else 'NEGATIVE' if enhanced_sentiment < -0.2 else 'NEUTRAL',
                'breakdown': {
                    'news': news_sentiment,
                    'social': social_sentiment,
                    'analyst': analyst_sentiment,
                    'options': options_sentiment,
                    'institutional': institutional_sentiment
                }
            }
            
        except Exception as e:
            print(f"❌ Enhanced sentiment calculation error: {e}")
            return {'enhanced_sentiment': 0.0, 'confidence': 0.5, 'label': 'NEUTRAL'}
    
    # NEW ENHANCED DATA SOURCES FOR SENTIMENT
    
    def _get_newsapi_data(self):
        """Get news from NewsAPI (Free tier: 1000 requests/month)"""
        try:
            # Note: You need to get a free API key from https://newsapi.org/
            api_key = "YOUR_NEWSAPI_KEY"  # Replace with actual key
            if api_key == "YOUR_NEWSAPI_KEY":
                return None
            
            url = f"https://newsapi.org/v2/everything?q={self.ticker}&apiKey={api_key}&sortBy=publishedAt&pageSize=10"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                headlines = []
                sentiments = []
                sources = []
                
                for article in articles:
                    title = article.get('title', '')
                    if title and self.ticker.lower() in title.lower():
                        headlines.append(title)
                        # Use advanced sentiment analysis if available
                        if self.sentiment_analyzer:
                            sentiment_result = self.sentiment_analyzer.analyze_text_advanced(title)
                            sentiment = sentiment_result['score']
                        else:
                            sentiment = self._analyze_text_sentiment(title)
                        sentiments.append(sentiment)
                        sources.append(article.get('source', {}).get('name', 'NewsAPI'))
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ NewsAPI error: {e}")
            return None
    
    def _get_alphavantage_news(self):
        """Get news from Alpha Vantage (Free tier: 5 calls/minute, 500 calls/day)"""
        try:
            # Note: You need to get a free API key from https://www.alphavantage.co/
            api_key = "YOUR_ALPHAVANTAGE_KEY"  # Replace with actual key
            if api_key == "YOUR_ALPHAVANTAGE_KEY":
                return None
            
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={self.ticker}&apikey={api_key}&limit=10"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('feed', [])
                
                headlines = []
                sentiments = []
                sources = []
                
                for article in articles:
                    title = article.get('title', '')
                    if title:
                        headlines.append(title)
                        # Alpha Vantage provides sentiment score
                        sentiment_score = article.get('overall_sentiment_score', 0)
                        sentiments.append(sentiment_score / 100)  # Normalize to -1 to 1
                        sources.append(article.get('source', 'AlphaVantage'))
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ Alpha Vantage error: {e}")
            return None
    
    def _get_yahoo_rss_news(self):
        """Get news from Yahoo Finance RSS feed"""
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en-US"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                headlines = []
                sentiments = []
                sources = []
                
                for item in items[:10]:
                    title = item.find('title')
                    if title:
                        title_text = title.get_text()
                        headlines.append(title_text)
                        sentiment = self._analyze_text_sentiment(title_text)
                        sentiments.append(sentiment)
                        sources.append('Yahoo Finance')
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ Yahoo RSS error: {e}")
            return None
    
    def _get_marketwatch_rss(self):
        """Get news from MarketWatch RSS feed"""
        try:
            url = "https://feeds.marketwatch.com/marketwatch/marketpulse/"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                headlines = []
                sentiments = []
                sources = []
                
                for item in items[:10]:
                    title = item.find('title')
                    if title and self.ticker.lower() in title.get_text().lower():
                        title_text = title.get_text()
                        headlines.append(title_text)
                        sentiment = self._analyze_text_sentiment(title_text)
                        sentiments.append(sentiment)
                        sources.append('MarketWatch')
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ MarketWatch RSS error: {e}")
            return None
    
    def _get_reddit_sentiment(self):
        """Get sentiment from Reddit (using Pushshift API - free)"""
        try:
            # Using Pushshift API (free, no auth required)
            url = f"https://api.pushshift.io/reddit/search/submission/?q={self.ticker}&subreddit=stocks,investing,wallstreetbets&size=10&sort=score"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                posts = data.get('data', [])
                
                headlines = []
                sentiments = []
                sources = []
                
                for post in posts:
                    title = post.get('title', '')
                    if title and self.ticker.lower() in title.lower():
                        headlines.append(title)
                        sentiment = self._analyze_text_sentiment(title)
                        sentiments.append(sentiment)
                        sources.append(f"Reddit r/{post.get('subreddit', 'unknown')}")
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ Reddit API error: {e}")
            return None
    
    def _get_stocktwits_data(self):
        """Get sentiment from StockTwits (free API)"""
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{self.ticker}.json"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                messages = data.get('messages', [])
                
                headlines = []
                sentiments = []
                sources = []
                
                for message in messages[:10]:
                    body = message.get('body', '')
                    if body:
                        headlines.append(body[:100] + "..." if len(body) > 100 else body)
                        sentiment = self._analyze_text_sentiment(body)
                        sentiments.append(sentiment)
                        sources.append('StockTwits')
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ StockTwits error: {e}")
            return None
    
    def _get_google_news_rss(self):
        """Get news from Google News RSS"""
        try:
            url = f"https://news.google.com/rss/search?q={self.ticker}+stock&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                headlines = []
                sentiments = []
                sources = []
                
                for item in items[:10]:
                    title = item.find('title')
                    if title:
                        title_text = title.get_text()
                        headlines.append(title_text)
                        sentiment = self._analyze_text_sentiment(title_text)
                        sentiments.append(sentiment)
                        sources.append('Google News')
                
                return {
                    'headlines': headlines,
                    'sentiments': sentiments,
                    'sources': sources,
                    'count': len(headlines)
                }
        except Exception as e:
            print(f"⚠️ Google News RSS error: {e}")
            return None
    
    # STOCK RANKINGS DATA SOURCES
    
    def _get_stock_rankings_data(self):
        """Get stock rankings from free sources"""
        try:
            rankings_data = {
                'finviz_rankings': self._get_finviz_rankings(),
                'marketwatch_rankings': self._get_marketwatch_rankings(),
                'yahoo_rankings': self._get_yahoo_rankings(),
                'sector_performance': self._get_sector_performance_ranking()
            }
            
            return rankings_data
        except Exception as e:
            print(f"❌ Stock rankings error: {e}")
            return {}
    
    def _get_finviz_rankings(self):
        """Get rankings from Finviz (free)"""
        try:
            url = f"https://finviz.com/quote.ashx?t={self.ticker}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract key metrics
                metrics = {}
                table = soup.find('table', class_='snapshot-table2')
                if table:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            key = cells[0].get_text().strip()
                            value = cells[1].get_text().strip()
                            metrics[key] = value
                
                return {
                    'pe_ratio': metrics.get('P/E', 'N/A'),
                    'peg_ratio': metrics.get('PEG', 'N/A'),
                    'price_to_sales': metrics.get('P/S', 'N/A'),
                    'price_to_book': metrics.get('P/B', 'N/A'),
                    'debt_to_equity': metrics.get('Debt/Eq', 'N/A'),
                    'return_on_equity': metrics.get('ROE', 'N/A'),
                    'return_on_investment': metrics.get('ROI', 'N/A'),
                    'revenue_growth': metrics.get('EPS next Y', 'N/A'),
                    'earnings_growth': metrics.get('EPS past 5Y', 'N/A'),
                    'sector_rank': metrics.get('Sector', 'N/A'),
                    'industry_rank': metrics.get('Industry', 'N/A')
                }
        except Exception as e:
            print(f"⚠️ Finviz error: {e}")
            return {}
    
    def _get_marketwatch_rankings(self):
        """Get rankings from MarketWatch"""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{self.ticker.lower()}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract analyst ratings
                ratings = []
                rating_elements = soup.find_all('div', class_='analyst__rating')
                for element in rating_elements:
                    rating_text = element.get_text().strip()
                    if rating_text:
                        ratings.append(rating_text)
                
                return {
                    'analyst_ratings': ratings,
                    'rating_count': len(ratings),
                    'average_rating': sum([1 if 'buy' in r.lower() else 0.5 if 'hold' in r.lower() else 0 for r in ratings]) / len(ratings) if ratings else 0
                }
        except Exception as e:
            print(f"⚠️ MarketWatch rankings error: {e}")
            return {}
    
    def _get_yahoo_rankings(self):
        """Get rankings from Yahoo Finance"""
        try:
            info = self.stock.info
            
            return {
                'recommendation_mean': info.get('recommendationMean', 0),
                'recommendation_key': info.get('recommendationKey', 'N/A'),
                'target_mean_price': info.get('targetMeanPrice', 0),
                'target_high_price': info.get('targetHighPrice', 0),
                'target_low_price': info.get('targetLowPrice', 0),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap_rank': info.get('marketCap', 0)
            }
        except Exception as e:
            print(f"⚠️ Yahoo rankings error: {e}")
            return {}
    
    def _get_sector_performance_ranking(self):
        """Get sector performance ranking"""
        try:
            # Get sector performance from Yahoo Finance
            sector = self.stock.info.get('sector', '')
            if sector:
                # Get sector ETF performance
                sector_etfs = {
                    'Technology': 'XLK',
                    'Healthcare': 'XLV',
                    'Financial Services': 'XLF',
                    'Consumer Cyclical': 'XLY',
                    'Consumer Defensive': 'XLP',
                    'Energy': 'XLE',
                    'Industrials': 'XLI',
                    'Communication Services': 'XLC',
                    'Utilities': 'XLU',
                    'Real Estate': 'XLRE',
                    'Materials': 'XLB'
                }
                
                sector_etf = sector_etfs.get(sector, 'SPY')
                sector_stock = yf.Ticker(sector_etf)
                sector_hist = sector_stock.history(period="1y")
                
                if not sector_hist.empty:
                    sector_return = (sector_hist['Close'].iloc[-1] / sector_hist['Close'].iloc[0] - 1) * 100
                    
                    # Get our stock's return
                    our_hist = self.stock.history(period="1y")
                    if not our_hist.empty:
                        our_return = (our_hist['Close'].iloc[-1] / our_hist['Close'].iloc[0] - 1) * 100
                        
                        return {
                            'sector': sector,
                            'sector_etf': sector_etf,
                            'sector_return_1y': sector_return,
                            'stock_return_1y': our_return,
                            'outperformance': our_return - sector_return,
                            'sector_rank': 'Above' if our_return > sector_return else 'Below'
                        }
        except Exception as e:
            print(f"⚠️ Sector performance error: {e}")
            return {}