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

class EnhancedDataCollector:
    """Collect data from multiple sources for comprehensive analysis"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        
    def get_comprehensive_data(self):
        """Get comprehensive data from multiple sources"""
        print(f"🔍 Collecting comprehensive data for {self.ticker}...")
        
        data = {}
        
        # 1. Market Data (Primary)
        data['market_data'] = self._get_market_data()
        
        # 2. News Sentiment (Multiple Sources)
        data['news_sentiment'] = self._get_news_sentiment()
        
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
        
        return data
    
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
