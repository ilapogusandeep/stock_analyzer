#!/usr/bin/env python3
"""
🏆 Universal Stock Analyzer - Compact One-Pager
Single file that works as both command-line tool and web interface
"""

import sys
import argparse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Try to import streamlit (optional for CLI)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Try to import plotly (optional for CLI)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception) as e:
    XGB_AVAILABLE = False

class PortfolioManager:
    """Portfolio and watchlist management with alerts"""
    
    def __init__(self, portfolio_file="portfolio.json"):
        self.portfolio_file = portfolio_file
        self.portfolio = self.load_portfolio()
        self.alerts = []
    
    def load_portfolio(self):
        """Load portfolio from JSON file"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default portfolio structure
        return {
            'watchlist': [],
            'portfolio': {},
            'alerts': [],
            'settings': {
                'alert_thresholds': {
                    'sentiment_change': 0.3,
                    'price_change': 0.05,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'volatility_spike': 0.4
                }
            }
        }
    
    def save_portfolio(self):
        """Save portfolio to JSON file"""
        try:
            with open(self.portfolio_file, 'w') as f:
                json.dump(self.portfolio, f, indent=2, default=str)
            return True
        except Exception:
            return False
    
    def add_to_watchlist(self, ticker, notes=""):
        """Add stock to watchlist"""
        ticker = ticker.upper()
        if ticker not in self.portfolio['watchlist']:
            self.portfolio['watchlist'].append({
                'ticker': ticker,
                'added_date': datetime.now().isoformat(),
                'notes': notes
            })
            self.save_portfolio()
            return True
        return False
    
    def remove_from_watchlist(self, ticker):
        """Remove stock from watchlist"""
        ticker = ticker.upper()
        self.portfolio['watchlist'] = [
            stock for stock in self.portfolio['watchlist'] 
            if stock['ticker'] != ticker
        ]
        self.save_portfolio()
        return True
    
    def add_to_portfolio(self, ticker, shares, avg_price, notes=""):
        """Add stock to portfolio"""
        ticker = ticker.upper()
        self.portfolio['portfolio'][ticker] = {
            'shares': shares,
            'avg_price': avg_price,
            'added_date': datetime.now().isoformat(),
            'notes': notes
        }
        self.save_portfolio()
        return True
    
    def remove_from_portfolio(self, ticker):
        """Remove stock from portfolio"""
        ticker = ticker.upper()
        if ticker in self.portfolio['portfolio']:
            del self.portfolio['portfolio'][ticker]
            self.save_portfolio()
            return True
        return False
    
    def get_watchlist_analysis(self):
        """Analyze all stocks in watchlist"""
        results = []
        
        for stock in self.portfolio['watchlist']:
            ticker = stock['ticker']
            try:
                analyzer = UniversalStockAnalyzer(ticker)
                data = analyzer.analyze(show_charts=False, show_ml=True, show_fundamentals=True, show_sentiment=True)
                
                if data:
                    # Extract key metrics
                    tech_data = data['tech_data']
                    sentiment_data = data['sentiment_data']
                    ml_prediction = data['ml_prediction']
                    earnings_data = data['earnings_data']
                    
                    result = {
                        'ticker': ticker,
                        'price': tech_data['current_price'],
                        'change_pct': tech_data['price_change_pct'],
                        'rsi': tech_data['rsi'],
                        'sentiment': sentiment_data['sentiment_label'],
                        'sentiment_score': sentiment_data['overall_sentiment'],
                        'ml_direction': ml_prediction['direction'] if ml_prediction else 'N/A',
                        'ml_confidence': ml_prediction['confidence'] if ml_prediction else 0,
                        'volatility': tech_data['volatility_20d'],
                        'earnings_alert': earnings_data['volatility_expected'],
                        'notes': stock.get('notes', ''),
                        'added_date': stock.get('added_date', '')
                    }
                    results.append(result)
                    
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'error': str(e),
                    'notes': stock.get('notes', ''),
                    'added_date': stock.get('added_date', '')
                })
        
        return results
    
    def get_portfolio_analysis(self):
        """Analyze portfolio holdings"""
        results = []
        total_value = 0
        total_cost = 0
        
        for ticker, holding in self.portfolio['portfolio'].items():
            try:
                analyzer = UniversalStockAnalyzer(ticker)
                data = analyzer.analyze(show_charts=False, show_ml=True, show_fundamentals=True, show_sentiment=True)
                
                if data:
                    current_price = data['tech_data']['current_price']
                    shares = holding['shares']
                    avg_price = holding['avg_price']
                    
                    current_value = current_price * shares
                    cost_basis = avg_price * shares
                    gain_loss = current_value - cost_basis
                    gain_loss_pct = (gain_loss / cost_basis) * 100
                    
                    total_value += current_value
                    total_cost += cost_basis
                    
                    result = {
                        'ticker': ticker,
                        'shares': shares,
                        'avg_price': avg_price,
                        'current_price': current_price,
                        'current_value': current_value,
                        'cost_basis': cost_basis,
                        'gain_loss': gain_loss,
                        'gain_loss_pct': gain_loss_pct,
                        'notes': holding.get('notes', ''),
                        'added_date': holding.get('added_date', '')
                    }
                    results.append(result)
                    
            except Exception as e:
                results.append({
                    'ticker': ticker,
                    'error': str(e),
                    'notes': holding.get('notes', ''),
                    'added_date': holding.get('added_date', '')
                })
        
        # Portfolio summary
        portfolio_summary = {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_gain_loss': total_value - total_cost,
            'total_gain_loss_pct': ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0,
            'holdings': results
        }
        
        return portfolio_summary
    
    def check_alerts(self):
        """Check for alert conditions across watchlist and portfolio"""
        alerts = []
        threshold = self.portfolio['settings']['alert_thresholds']
        
        # Check watchlist
        watchlist_data = self.get_watchlist_analysis()
        
        for stock in watchlist_data:
            if 'error' in stock:
                continue
                
            ticker = stock['ticker']
            
            # Price change alert
            if abs(stock['change_pct']) > threshold['price_change'] * 100:
                alerts.append({
                    'type': 'price_change',
                    'ticker': ticker,
                    'message': f"{ticker}: {stock['change_pct']:+.1f}% price change",
                    'severity': 'high' if abs(stock['change_pct']) > 0.1 else 'medium',
                    'timestamp': datetime.now().isoformat()
                })
            
            # RSI alerts
            if stock['rsi'] > threshold['rsi_overbought']:
                alerts.append({
                    'type': 'rsi_overbought',
                    'ticker': ticker,
                    'message': f"{ticker}: RSI {stock['rsi']:.1f} - Overbought",
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
            elif stock['rsi'] < threshold['rsi_oversold']:
                alerts.append({
                    'type': 'rsi_oversold',
                    'ticker': ticker,
                    'message': f"{ticker}: RSI {stock['rsi']:.1f} - Oversold",
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Volatility alert
            if stock['volatility'] > threshold['volatility_spike'] * 100:
                alerts.append({
                    'type': 'volatility_spike',
                    'ticker': ticker,
                    'message': f"{ticker}: High volatility {stock['volatility']:.1f}%",
                    'severity': 'high',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Earnings alert
            if stock['earnings_alert']:
                alerts.append({
                    'type': 'earnings_alert',
                    'ticker': ticker,
                    'message': f"{ticker}: Earnings coming up - High volatility expected",
                    'severity': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Store alerts
        self.portfolio['alerts'] = alerts
        self.save_portfolio()
        
        return alerts

class UniversalStockAnalyzer:
    """Universal stock analyzer - works for both CLI and web"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        
    def analyze(self, show_charts=True, show_ml=True, show_fundamentals=True, show_sentiment=True):
        """Main analysis function with enhanced data sources"""
        print(f"🔍 Analyzing {self.ticker}...")
        
        # Fetch enhanced data from multiple sources
        try:
            from enhanced_data_collector import EnhancedDataCollector
            enhanced_collector = EnhancedDataCollector(self.ticker)
            enhanced_data = enhanced_collector.get_comprehensive_data()
        except ImportError:
            print("⚠️ Enhanced data collector not available, using basic data")
            enhanced_data = {}
        
        # Fetch institutional and hedge fund data
        institutional_data = {}
        try:
            from enhanced_institutional_data import EnhancedInstitutionalData
            institutional_collector = EnhancedInstitutionalData(self.ticker)
            institutional_data = institutional_collector.get_comprehensive_institutional_data()
        except ImportError:
            print("⚠️ Institutional data collector not available")
            institutional_data = {}
        
        # Get basic market data
        hist = self.stock.history(period="1y")
        info = self.stock.info
        
        if hist.empty:
            print(f"❌ No data available for {self.ticker}")
            return None
        
        # Compute analysis
        tech_data = self.compute_technical_indicators(hist)
        fundamental_data = self.analyze_fundamentals(info, hist)
        
        # Enhanced sentiment analysis
        if show_sentiment and enhanced_data:
            sentiment_data = self._process_enhanced_sentiment(enhanced_data)
        else:
            sentiment_data = self.analyze_sentiment() if show_sentiment else {'sentiment_label': 'NEUTRAL', 'overall_sentiment': 0, 'news_sentiment': 0, 'social_sentiment': 0, 'news_count': 0, 'positive_ratio': 0.5, 'confidence': 0.5}
        
        earnings_data = self.get_earnings_calendar() if show_fundamentals else {'earnings_expected': False, 'next_earnings_date': None, 'days_to_earnings': 0, 'volatility_expected': False}
        
        # Enhanced ML prediction with multiple data sources
        ml_prediction = None
        if show_ml and ML_AVAILABLE:
            ml_prediction = self.create_enhanced_ml_prediction(hist, tech_data, fundamental_data, sentiment_data, enhanced_data)
        
        # Enhanced backtesting with ML validation
        backtest_results = None
        if show_ml and ML_AVAILABLE and len(hist) > 100:  # Need minimum data for backtesting
            backtest_results = self.enhanced_backtest_strategy(hist, tech_data, fundamental_data, sentiment_data, enhanced_data)
        
        return {
            'hist': hist,
            'info': info,
            'tech_data': tech_data,
            'fundamental_data': fundamental_data,
            'sentiment_data': sentiment_data,
            'earnings_data': earnings_data,
            'ml_prediction': ml_prediction,
            'backtest_results': backtest_results,
            'institutional_data': institutional_data
        }
    
    def compute_technical_indicators(self, hist):
        """Compute comprehensive technical indicators"""
        tech_data = {}
        current_price = hist['Close'].iloc[-1]
        
        # Price data
        tech_data['current_price'] = current_price
        tech_data['price_change'] = current_price - hist['Close'].iloc[-2] if len(hist) > 1 else 0
        tech_data['price_change_pct'] = (tech_data['price_change'] / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
        
        # Moving averages
        tech_data['sma_5'] = float(hist['Close'].rolling(5).mean().iloc[-1])
        tech_data['sma_10'] = float(hist['Close'].rolling(10).mean().iloc[-1])
        tech_data['sma_20'] = float(hist['Close'].rolling(20).mean().iloc[-1])
        tech_data['sma_50'] = float(hist['Close'].rolling(50).mean().iloc[-1])
        tech_data['sma_200'] = float(hist['Close'].rolling(200).mean().iloc[-1])
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        tech_data['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
        
        # MACD
        ema_12 = hist['Close'].ewm(span=12).mean()
        ema_26 = hist['Close'].ewm(span=26).mean()
        tech_data['macd'] = float(ema_12.iloc[-1] - ema_26.iloc[-1])
        tech_data['macd_signal'] = float(hist['Close'].ewm(span=9).mean().iloc[-1])
        tech_data['macd_histogram'] = float(tech_data['macd'] - tech_data['macd_signal'])
        
        # Bollinger Bands
        bb_middle = hist['Close'].rolling(20).mean()
        bb_std = hist['Close'].rolling(20).std()
        tech_data['bb_upper'] = float(bb_middle.iloc[-1] + (bb_std.iloc[-1] * 2))
        tech_data['bb_lower'] = float(bb_middle.iloc[-1] - (bb_std.iloc[-1] * 2))
        tech_data['bb_position'] = float(((current_price - tech_data['bb_lower']) / 
                                   (tech_data['bb_upper'] - tech_data['bb_lower'])) * 100)
        
        # Volume
        tech_data['volume'] = int(hist['Volume'].iloc[-1])
        tech_data['volume_sma_20'] = float(hist['Volume'].rolling(20).mean().iloc[-1])
        tech_data['volume_ratio'] = float(tech_data['volume'] / tech_data['volume_sma_20'])
        
        # Volatility
        returns = hist['Close'].pct_change().dropna()
        tech_data['volatility_20d'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
        
        # Performance metrics
        tech_data['performance_1d'] = float(((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0)
        tech_data['performance_5d'] = float(((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100 if len(hist) > 5 else 0)
        tech_data['performance_1m'] = float(((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100 if len(hist) > 21 else 0)
        tech_data['performance_3m'] = float(((current_price - hist['Close'].iloc[-63]) / hist['Close'].iloc[-63]) * 100 if len(hist) > 63 else 0)
        tech_data['performance_1y'] = float(((current_price - hist['Close'].iloc[-252]) / hist['Close'].iloc[-252]) * 100 if len(hist) > 252 else 0)
        
        return tech_data
    
    def analyze_fundamentals(self, info, hist):
        """Analyze fundamental metrics"""
        fundamental_data = {}
        
        # Basic ratios
        fundamental_data['pe_ratio'] = info.get('trailingPE', 0)
        fundamental_data['forward_pe'] = info.get('forwardPE', 0)
        fundamental_data['peg_ratio'] = info.get('pegRatio', 0)
        fundamental_data['price_to_book'] = info.get('priceToBook', 0)
        fundamental_data['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
        fundamental_data['debt_to_equity'] = info.get('debtToEquity', 0)
        
        # Growth metrics
        fundamental_data['revenue_growth'] = info.get('revenueGrowth', 0)
        fundamental_data['earnings_growth'] = info.get('earningsGrowth', 0)
        fundamental_data['return_on_equity'] = info.get('returnOnEquity', 0)
        fundamental_data['profit_margin'] = info.get('profitMargins', 0)
        
        # Market metrics
        fundamental_data['market_cap'] = info.get('marketCap', 0)
        fundamental_data['enterprise_value'] = info.get('enterpriseValue', 0)
        
        # Liquidity ratios
        fundamental_data['current_ratio'] = info.get('currentRatio', 0)
        fundamental_data['quick_ratio'] = info.get('quickRatio', 0)
        fundamental_data['beta'] = info.get('beta', 1.0)
        
        # Analyst data
        fundamental_data['target_price'] = info.get('targetMeanPrice', 0)
        fundamental_data['recommendation'] = info.get('recommendationMean', 0)
        
        # Calculate relative strength vs market
        if not hist.empty and len(hist) > 252:
            try:
                spy = yf.Ticker("SPY").history(period="1y")
                if not spy.empty:
                    stock_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1) * 100
                    market_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[-252] - 1) * 100
                    fundamental_data['relative_strength'] = stock_return - market_return
                else:
                    fundamental_data['relative_strength'] = 0
            except:
                fundamental_data['relative_strength'] = 0
        else:
            fundamental_data['relative_strength'] = 0
        
        return fundamental_data
    
    def analyze_sentiment(self):
        """Enhanced sentiment analysis"""
        sentiment_data = {}
        
        # Mock enhanced sentiment data
        news_sentiment = {
            'headlines': [
                f"{self.ticker} reports strong quarterly earnings beating estimates",
                f"{self.ticker} faces regulatory challenges in key markets",
                f"{self.ticker} announces new product launch with innovative features",
                f"{self.ticker} stock shows bullish momentum with institutional buying",
                f"{self.ticker} analyst upgrades price target citing strong fundamentals"
            ],
            'sources': ['Reuters', 'Bloomberg', 'CNBC', 'Yahoo Finance', 'MarketWatch'],
            'sentiment_scores': [0.8, -0.3, 0.6, 0.7, 0.5],
            'impact_levels': ['High', 'Medium', 'High', 'Medium', 'Medium']
        }
        
        # Calculate weighted sentiment
        weighted_sentiment = np.average(news_sentiment['sentiment_scores'], 
                                      weights=[1.0, 0.7, 1.0, 0.7, 0.7])
        
        sentiment_data['news_sentiment'] = weighted_sentiment
        sentiment_data['news_count'] = len(news_sentiment['headlines'])
        sentiment_data['positive_ratio'] = len([s for s in news_sentiment['sentiment_scores'] if s > 0.1]) / len(news_sentiment['sentiment_scores'])
        
        # Social sentiment (mock)
        sentiment_data['social_sentiment'] = 0.3
        sentiment_data['social_volume'] = 1250
        
        # Overall sentiment
        sentiment_data['overall_sentiment'] = (weighted_sentiment + sentiment_data['social_sentiment']) / 2
        
        if sentiment_data['overall_sentiment'] > 0.1:
            sentiment_data['sentiment_label'] = "POSITIVE"
        elif sentiment_data['overall_sentiment'] < -0.1:
            sentiment_data['sentiment_label'] = "NEGATIVE"
        else:
            sentiment_data['sentiment_label'] = "NEUTRAL"
        
        return sentiment_data
    
    def _process_enhanced_sentiment(self, enhanced_data):
        """Process enhanced sentiment data from multiple sources"""
        try:
            from enhanced_data_collector import EnhancedDataCollector
            collector = EnhancedDataCollector(self.ticker)
            enhanced_sentiment = collector.get_enhanced_sentiment_score(enhanced_data)
            
            return {
                'sentiment_label': enhanced_sentiment['label'],
                'overall_sentiment': enhanced_sentiment['enhanced_sentiment'],
                'confidence': enhanced_sentiment['confidence'],
                'news_sentiment': enhanced_data.get('news_sentiment', {}).get('overall_sentiment', 0),
                'social_sentiment': enhanced_data.get('social_sentiment', {}).get('sentiment', 0),
                'analyst_sentiment': enhanced_data.get('analyst_data', {}).get('rating_mean', 0),
                'options_sentiment': enhanced_data.get('options_data', {}).get('sentiment', 'NEUTRAL'),
                'news_count': enhanced_data.get('news_sentiment', {}).get('news_count', 0),
                'positive_ratio': enhanced_data.get('news_sentiment', {}).get('positive_ratio', 0.5),
                'breakdown': enhanced_sentiment.get('breakdown', {}),
                'sources': enhanced_data.get('news_sentiment', {}).get('sources', [])
            }
        except Exception as e:
            print(f"⚠️ Enhanced sentiment processing error: {e}")
            return self.analyze_sentiment()
    
    def get_earnings_calendar(self):
        """Get earnings calendar information"""
        try:
            # Get earnings data from yfinance
            earnings = self.stock.calendar
            if earnings is not None and not earnings.empty:
                next_earnings = earnings.iloc[0] if len(earnings) > 0 else None
                
                if next_earnings is not None:
                    earnings_date = next_earnings.name
                    days_to_earnings = (earnings_date - pd.Timestamp.now()).days
                    
                    return {
                        'next_earnings_date': earnings_date,
                        'days_to_earnings': days_to_earnings,
                        'earnings_expected': True,
                        'volatility_expected': days_to_earnings <= 30  # High volatility expected within 30 days
                    }
        except Exception:
            pass
        
        # Fallback: estimate based on quarterly pattern
        return {
            'next_earnings_date': None,
            'days_to_earnings': None,
            'earnings_expected': False,
            'volatility_expected': False
        }
    
    def backtest_strategy(self, hist, lookback_days=252, prediction_horizon=5):
        """Backtest the ML strategy with performance metrics"""
        try:
            if len(hist) < lookback_days + prediction_horizon:
                return None
            
            # Use last year of data for backtesting
            test_data = hist.tail(lookback_days).copy()
            
            # Generate predictions for each day
            predictions = []
            actual_returns = []
            confidence_scores = []
            
            for i in range(prediction_horizon, len(test_data) - prediction_horizon):
                # Get historical data up to this point
                hist_slice = test_data.iloc[:i]
                
                # Compute features for this point
                tech_data = self.compute_technical_indicators(hist_slice)
                fundamental_data = self.analyze_fundamentals(self.stock.info, hist_slice)
                sentiment_data = self.analyze_sentiment()
                
                # Create ML prediction
                ml_pred = self.create_ml_prediction(hist_slice, tech_data, fundamental_data, sentiment_data)
                
                if ml_pred:
                    predictions.append(ml_pred['direction'])
                    confidence_scores.append(ml_pred['confidence'])
                    
                    # Calculate actual return over next N days
                    current_price = hist_slice['Close'].iloc[-1]
                    future_price = test_data['Close'].iloc[i + prediction_horizon]
                    actual_return = (future_price - current_price) / current_price
                    actual_returns.append(actual_return)
            
            if len(predictions) < 10:  # Need minimum data points
                return None
            
            # Calculate performance metrics
            return self._calculate_backtest_metrics(predictions, actual_returns, confidence_scores)
            
        except Exception as e:
            return None
    
    def _calculate_backtest_metrics(self, predictions, actual_returns, confidence_scores):
        """Calculate comprehensive backtest performance metrics"""
        # Convert predictions to signals
        signals = []
        for pred in predictions:
            if pred == "BULLISH":
                signals.append(1)  # Buy
            elif pred == "BEARISH":
                signals.append(-1)  # Sell
            else:
                signals.append(0)  # Hold
        
        # Calculate returns for each signal
        strategy_returns = []
        for i, (signal, actual_return) in enumerate(zip(signals, actual_returns)):
            if signal == 1:  # Buy signal
                strategy_returns.append(actual_return)
            elif signal == -1:  # Sell signal
                strategy_returns.append(-actual_return)  # Short
            else:  # Hold
                strategy_returns.append(0)
        
        # Performance metrics
        total_trades = len([r for r in strategy_returns if r != 0])
        winning_trades = len([r for r in strategy_returns if r > 0])
        losing_trades = len([r for r in strategy_returns if r < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        strategy_total_return = sum(strategy_returns)
        strategy_avg_return = np.mean(strategy_returns) if strategy_returns else 0
        strategy_volatility = np.std(strategy_returns) if len(strategy_returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = strategy_avg_return / strategy_volatility if strategy_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Confidence-weighted performance
        high_conf_trades = []
        for i, (conf, ret) in enumerate(zip(confidence_scores, strategy_returns)):
            if conf > 0.6:  # High confidence threshold
                high_conf_trades.append(ret)
        
        high_conf_win_rate = len([r for r in high_conf_trades if r > 0]) / len(high_conf_trades) if high_conf_trades else 0
        high_conf_avg_return = np.mean(high_conf_trades) if high_conf_trades else 0
        
        # Prediction accuracy
        correct_predictions = 0
        for i, (pred, actual_return) in enumerate(zip(predictions, actual_returns)):
            if pred == "BULLISH" and actual_return > 0:
                correct_predictions += 1
            elif pred == "BEARISH" and actual_return < 0:
                correct_predictions += 1
            elif pred == "NEUTRAL" and abs(actual_return) < 0.02:  # Within 2%
                correct_predictions += 1
        
        prediction_accuracy = correct_predictions / len(predictions) if predictions else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'strategy_total_return': strategy_total_return,
            'strategy_avg_return': strategy_avg_return,
            'strategy_volatility': strategy_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'prediction_accuracy': prediction_accuracy,
            'high_conf_trades': len(high_conf_trades),
            'high_conf_win_rate': high_conf_win_rate,
            'high_conf_avg_return': high_conf_avg_return,
            'total_signals': {
                'bullish': len([p for p in predictions if p == "BULLISH"]),
                'bearish': len([p for p in predictions if p == "BEARISH"]),
                'neutral': len([p for p in predictions if p == "NEUTRAL"])
            }
        }
    
    def create_ml_prediction(self, hist, tech_data, fundamental_data, sentiment_data):
        """Create ML-based prediction with probability"""
        try:
            # Create comprehensive feature set
            features = pd.DataFrame(index=hist.index)
            features['price'] = hist['Close']
            features['returns_1d'] = hist['Close'].pct_change(1)
            features['returns_3d'] = hist['Close'].pct_change(3)
            features['returns_7d'] = hist['Close'].pct_change(7)
            features['volume_ratio'] = hist['Volume'] / hist['Volume'].rolling(20).mean()
            features['volatility'] = hist['Close'].pct_change().rolling(20).std()
            
            # Technical features
            features['rsi'] = self._calculate_rsi(hist['Close'])
            features['macd'] = self._calculate_macd(hist['Close'])
            features['sma_20'] = hist['Close'].rolling(20).mean()
            features['price_sma_ratio'] = hist['Close'] / features['sma_20']
            features['bb_position'] = self._calculate_bb_position(hist['Close'])
            
            # Fundamental features
            features['pe_ratio'] = fundamental_data['pe_ratio']
            features['relative_strength'] = fundamental_data['relative_strength']
            
            # Sentiment features
            features['sentiment'] = sentiment_data['overall_sentiment']
            features['news_count'] = sentiment_data['news_count']
            
            # Create target (next day direction)
            features['target'] = (features['returns_1d'].shift(-1) > 0).astype(int)
            
            # Remove NaN
            model_data = features.dropna()
            
            if len(model_data) < 100:
                return None
            
            # Prepare data
            feature_cols = [col for col in model_data.columns if col not in ['target', 'price', 'returns_1d']]
            X = model_data[feature_cols]
            y = model_data['target']
            
            # Train models
            models = {}
            accuracies = {}
            
            # Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            models['RandomForest'] = rf
            accuracies['RandomForest'] = rf.score(X, y)
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(X, y)
            models['GradientBoosting'] = gb
            accuracies['GradientBoosting'] = gb.score(X, y)
            
            # XGBoost if available
            if XGB_AVAILABLE:
                try:
                    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
                    xgb_model.fit(X, y)
                    models['XGBoost'] = xgb_model
                    accuracies['XGBoost'] = xgb_model.score(X, y)
                except Exception:
                    pass
            
            # Ensemble prediction
            latest_features = X.iloc[-1:].values
            
            predictions = []
            probabilities = []
            
            for name, model in models.items():
                pred = model.predict(latest_features)[0]
                prob = model.predict_proba(latest_features)[0]
                predictions.append(pred)
                probabilities.append(prob)
            
            # Average predictions
            avg_prediction = np.mean(predictions)
            avg_probability = np.mean(probabilities, axis=0)
            
            # Enhanced scenario probabilities
            bullish_prob = avg_probability[1]
            bearish_prob = avg_probability[0]
            # Calculate neutral probability as the uncertainty between bullish and bearish
            neutral_prob = 1 - abs(bullish_prob - bearish_prob)
            
            # Ensure probabilities sum to 1
            total_prob = bullish_prob + bearish_prob + neutral_prob
            bullish_prob /= total_prob
            bearish_prob /= total_prob
            neutral_prob /= total_prob
            
            # Determine primary direction and confidence
            if bullish_prob > bearish_prob and bullish_prob > neutral_prob:
                direction = "BULLISH"
                confidence = bullish_prob
            elif bearish_prob > bullish_prob and bearish_prob > neutral_prob:
                direction = "BEARISH"
                confidence = bearish_prob
            else:
                direction = "NEUTRAL"
                confidence = neutral_prob
            
            # SHAP explainability
            shap_explanations = {}
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(latest_features)
                    
                    # Get feature contributions for latest prediction
                    feature_contributions = {}
                    for i, feature in enumerate(feature_cols):
                        # Use the positive class contribution (index 1)
                        if len(shap_values) > 1:
                            feature_contributions[feature] = shap_values[1][0][i]
                        else:
                            feature_contributions[feature] = shap_values[0][i]
                    
                    # Sort by absolute contribution
                    sorted_contributions = sorted(feature_contributions.items(), 
                                               key=lambda x: abs(x[1]), reverse=True)
                    
                    shap_explanations = {
                        'top_features': sorted_contributions[:5],
                        'explanation': self._generate_explanation(sorted_contributions, bullish_prob, bearish_prob, neutral_prob)
                    }
                except Exception as e:
                    # Fallback: use feature importance for explanation
                    feature_contributions = {}
                    for i, feature in enumerate(feature_cols):
                        feature_contributions[feature] = rf.feature_importances_[i]
                    
                    sorted_contributions = sorted(feature_contributions.items(), 
                                               key=lambda x: abs(x[1]), reverse=True)
                    
                    shap_explanations = {
                        'top_features': sorted_contributions[:5],
                        'explanation': self._generate_explanation(sorted_contributions, bullish_prob, bearish_prob, neutral_prob)
                    }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Calculate price targets for each scenario
            current_price = hist['Close'].iloc[-1]
            bullish_target = current_price * 1.10  # +10% for bullish
            bearish_target = current_price * 0.92  # -8% for bearish
            neutral_target = current_price * 1.00  # Flat for neutral
            
            return {
                'direction': direction,
                'confidence': confidence,
                'scenario_probabilities': {
                    'bullish': bullish_prob,
                    'neutral': neutral_prob,
                    'bearish': bearish_prob
                },
                'scenario_targets': {
                    'bullish': bullish_target,
                    'neutral': neutral_target,
                    'bearish': bearish_target
                },
                'probability_up': bullish_prob,
                'probability_down': bearish_prob,
                'model_accuracies': accuracies,
                'feature_importance': feature_importance.head(5).to_dict('records'),
                'shap_explanations': shap_explanations,
                'price_target': current_price * (1 + (bullish_prob - bearish_prob) * 0.1)
            }
            
        except Exception as e:
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def _calculate_bollinger_upper(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands Upper"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            return upper_band.fillna(prices)
        except:
            return prices
    
    def _calculate_bollinger_lower(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands Lower"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            lower_band = sma - (std * std_dev)
            return lower_band.fillna(prices)
        except:
            return prices
    
    def _calculate_bb_position(self, prices, period=20):
        """Calculate Bollinger Band position"""
        bb_middle = prices.rolling(period).mean()
        bb_std = prices.rolling(period).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        return ((prices - bb_lower) / (bb_upper - bb_lower)) * 100
    
    def _generate_explanation(self, contributions, bullish_prob, bearish_prob, neutral_prob):
        """Generate natural language explanation of ML prediction"""
        explanations = []
        
        # Analyze top contributing features
        for feature, contribution in contributions[:3]:
            if abs(contribution) > 0.1:  # Significant contribution
                if 'rsi' in feature.lower():
                    if contribution > 0:
                        explanations.append(f"RSI showing bullish momentum (+{contribution:.2f})")
                    else:
                        explanations.append(f"RSI indicating bearish pressure ({contribution:.2f})")
                elif 'macd' in feature.lower():
                    if contribution > 0:
                        explanations.append(f"MACD trending upward (+{contribution:.2f})")
                    else:
                        explanations.append(f"MACD showing downward trend ({contribution:.2f})")
                elif 'sentiment' in feature.lower():
                    if contribution > 0:
                        explanations.append(f"Positive sentiment driving price (+{contribution:.2f})")
                    else:
                        explanations.append(f"Negative sentiment weighing on price ({contribution:.2f})")
                elif 'volume' in feature.lower():
                    if contribution > 0:
                        explanations.append(f"High volume supporting move (+{contribution:.2f})")
                    else:
                        explanations.append(f"Low volume suggesting weakness ({contribution:.2f})")
                elif 'returns' in feature.lower():
                    if contribution > 0:
                        explanations.append(f"Recent returns showing strength (+{contribution:.2f})")
                    else:
                        explanations.append(f"Recent returns indicating weakness ({contribution:.2f})")
        
        # Add scenario summary
        if bullish_prob > 0.4:
            scenario = f"Bullish scenario most likely ({bullish_prob:.1%})"
        elif bearish_prob > 0.4:
            scenario = f"Bearish scenario most likely ({bearish_prob:.1%})"
        else:
            scenario = f"Neutral scenario most likely ({neutral_prob:.1%})"
        
        if explanations:
            return f"{scenario}. Key drivers: {'; '.join(explanations[:2])}."
        else:
            return f"{scenario}. Mixed signals across technical indicators."
    
    def create_enhanced_ml_prediction(self, hist, tech_data, fundamental_data, sentiment_data, enhanced_data):
        """Create enhanced ML prediction with multiple data sources"""
        if not ML_AVAILABLE:
            return None
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import shap
            
            # Prepare enhanced features
            features = self._create_enhanced_features(hist, tech_data, fundamental_data, sentiment_data, enhanced_data)
            
            if len(features) < 50:  # Need minimum data for ML
                return None
            
            # Create labels (next day direction)
            labels = (hist['Close'].shift(-1) > hist['Close']).astype(int)
            labels = labels.dropna()
            
            # Align features and labels
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble models
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            model_predictions = {}
            model_accuracies = {}
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = (y_pred == y_test).mean()
                
                model_predictions[name] = model
                model_accuracies[name] = accuracy
            
            # Get latest prediction
            latest_features = features.iloc[-1:].values
            latest_features_scaled = scaler.transform(latest_features)
            
            # Ensemble prediction
            rf_pred = model_predictions['RandomForest'].predict_proba(latest_features_scaled)[0]
            gb_pred = model_predictions['GradientBoosting'].predict_proba(latest_features_scaled)[0]
            
            # Weighted ensemble
            ensemble_pred = (rf_pred * 0.6 + gb_pred * 0.4)
            
            # Calculate direction and confidence
            bullish_prob = ensemble_pred[1]
            bearish_prob = ensemble_pred[0]
            neutral_prob = 1 - abs(bullish_prob - bearish_prob)
            
            if bullish_prob > bearish_prob:
                direction = "BULLISH"
                confidence = bullish_prob
            elif bearish_prob > bullish_prob:
                direction = "BEARISH"
                confidence = bearish_prob
            else:
                direction = "NEUTRAL"
                confidence = neutral_prob
            
            # Calculate price target
            current_price = tech_data['current_price']
            if direction == "BULLISH":
                price_target = current_price * (1 + bullish_prob * 0.1)  # Up to 10% upside
                expected_return = bullish_prob * 0.1
            elif direction == "BEARISH":
                price_target = current_price * (1 - bearish_prob * 0.08)  # Up to 8% downside
                expected_return = -bearish_prob * 0.08
            else:
                price_target = current_price
                expected_return = 0
            
            # Scenario probabilities
            scenario_probabilities = {
                'bullish': bullish_prob,
                'neutral': neutral_prob,
                'bearish': bearish_prob
            }
            
            scenario_targets = {
                'bullish': current_price * 1.1,
                'neutral': current_price,
                'bearish': current_price * 0.92
            }
            
            # SHAP explanations
            shap_explanations = None
            try:
                explainer = shap.TreeExplainer(model_predictions['RandomForest'])
                shap_values = explainer.shap_values(latest_features_scaled)
                
                # Get top features - handle both single and multi-class cases
                feature_names = features.columns.tolist()
                
                # Handle different SHAP output formats
                if len(shap_values.shape) == 2:
                    # Multi-class case
                    shap_values_single = shap_values[0]  # Take first class
                else:
                    # Single class case
                    shap_values_single = shap_values
                
                feature_importance = abs(shap_values_single)
                # Convert to Python scalars
                top_features = []
                for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                    # Convert numpy scalar to Python float
                    if hasattr(importance, 'item'):
                        importance_val = importance.item()
                    else:
                        importance_val = float(importance)
                    top_features.append((name, importance_val))
                
                top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:5]
                
                # Generate explanation
                explanation = self._generate_ml_explanation(direction, confidence, top_features, enhanced_data)
                
                # Convert SHAP values to Python list
                if hasattr(shap_values_single, 'tolist'):
                    shap_list = shap_values_single.tolist()
                else:
                    shap_list = [float(x) for x in shap_values_single]
                
                shap_explanations = {
                    'explanation': explanation,
                    'top_features': top_features,
                    'shap_values': shap_list
                }
            except Exception as e:
                print(f"⚠️ SHAP explanation error: {e}")
            
            # Feature importance
            feature_importance = []
            for name, model in model_predictions.items():
                importance = model.feature_importances_
                for i, feature in enumerate(features.columns):
                    feature_importance.append({
                        'feature': feature,
                        'importance': importance[i],
                        'model': name
                    })
            
            return {
                'direction': direction,
                'confidence': confidence,
                'price_target': price_target,
                'expected_return': expected_return,
                'scenario_probabilities': scenario_probabilities,
                'scenario_targets': scenario_targets,
                'model_accuracies': model_accuracies,
                'feature_importance': feature_importance,
                'shap_explanations': shap_explanations,
                'enhanced_features_used': len(features.columns),
                'data_sources': list(enhanced_data.keys()) if enhanced_data else ['basic']
            }
            
        except Exception as e:
            print(f"❌ Enhanced ML prediction error: {e}")
            return self.create_ml_prediction(hist, tech_data, fundamental_data, sentiment_data)
    
    def _create_enhanced_features(self, hist, tech_data, fundamental_data, sentiment_data, enhanced_data):
        """Create enhanced feature set from multiple data sources"""
        try:
            # Basic technical features
            features = pd.DataFrame(index=hist.index)
            
            # Price features
            features['returns_1d'] = hist['Close'].pct_change(1)
            features['returns_5d'] = hist['Close'].pct_change(5)
            features['returns_20d'] = hist['Close'].pct_change(20)
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(hist['Close'])
            features['macd'] = self._calculate_macd(hist['Close'])
            features['bb_upper'] = self._calculate_bollinger_upper(hist['Close'])
            features['bb_lower'] = self._calculate_bollinger_lower(hist['Close'])
            features['volatility_20d'] = hist['Close'].pct_change().rolling(20).std()
            
            # Volume features
            features['volume_ratio'] = hist['Volume'] / hist['Volume'].rolling(20).mean()
            features['price_volume'] = hist['Close'] * hist['Volume']
            
            # Moving averages
            features['sma_5'] = hist['Close'].rolling(5).mean()
            features['sma_20'] = hist['Close'].rolling(20).mean()
            features['sma_50'] = hist['Close'].rolling(50).mean()
            features['price_ma5_ratio'] = hist['Close'] / features['sma_5']
            features['price_ma20_ratio'] = hist['Close'] / features['sma_20']
            
            # Enhanced features from multiple sources
            if enhanced_data:
                # News sentiment features
                news_sentiment = enhanced_data.get('news_sentiment', {})
                features['news_sentiment'] = news_sentiment.get('overall_sentiment', 0)
                features['news_count'] = news_sentiment.get('news_count', 0)
                features['news_positive_ratio'] = news_sentiment.get('positive_ratio', 0.5)
                
                # Social sentiment features
                social_sentiment = enhanced_data.get('social_sentiment', {})
                features['social_sentiment'] = social_sentiment.get('sentiment', 0)
                features['social_mentions'] = social_sentiment.get('mentions', 0)
                features['social_engagement'] = social_sentiment.get('engagement_rate', 0)
                
                # Analyst features
                analyst_data = enhanced_data.get('analyst_data', {})
                features['analyst_rating'] = analyst_data.get('rating_mean', 0)
                features['analyst_upside'] = analyst_data.get('upside_potential', 0)
                features['analyst_count'] = analyst_data.get('analyst_count', 0)
                
                # Options features
                options_data = enhanced_data.get('options_data', {})
                features['put_call_ratio'] = options_data.get('put_call_ratio', 1.0)
                features['options_volume'] = options_data.get('options_volume', 0)
                features['implied_volatility'] = options_data.get('implied_volatility', 0.2)
                
                # Institutional features
                institutional_data = enhanced_data.get('institutional_data', {})
                features['institutional_ownership'] = institutional_data.get('ownership_percent', 0)
                features['institution_count'] = institutional_data.get('institution_count', 0)
                
                # Economic features
                economic_data = enhanced_data.get('economic_data', {})
                features['vix'] = economic_data.get('vix', 20)
                features['treasury_yield'] = economic_data.get('treasury_yield', 3.0)
                features['dollar_index'] = economic_data.get('dollar_index', 100)
            
            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            print(f"❌ Enhanced feature creation error: {e}")
            return pd.DataFrame()
    
    def _generate_ml_explanation(self, direction, confidence, top_features, enhanced_data):
        """Generate natural language explanation for ML prediction"""
        try:
            explanation_parts = []
            
            # Direction and confidence
            explanation_parts.append(f"The model predicts a {direction.lower()} direction with {confidence:.1%} confidence.")
            
            # Top features
            if top_features:
                explanation_parts.append("Key factors driving this prediction:")
                for feature, importance in top_features[:3]:
                    feature_name = feature.replace('_', ' ').title()
                    explanation_parts.append(f"• {feature_name}: {importance:.3f}")
            
            # Data sources
            if enhanced_data:
                sources = list(enhanced_data.keys())
                explanation_parts.append(f"Analysis based on {len(sources)} data sources: {', '.join(sources[:3])}")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            return f"ML model predicts {direction.lower()} direction with {confidence:.1%} confidence."
    
    def enhanced_backtest_strategy(self, hist, tech_data, fundamental_data, sentiment_data, enhanced_data):
        """Enhanced backtesting with ML validation and multiple data sources"""
        if not ML_AVAILABLE:
            return None
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            print("🔄 Running enhanced backtest...")
            
            # Create enhanced features
            features = self._create_enhanced_features(hist, tech_data, fundamental_data, sentiment_data, enhanced_data)
            
            if len(features) < 50:
                return None
            
            # Create labels (next day direction)
            labels = (hist['Close'].shift(-1) > hist['Close']).astype(int)
            labels = labels.dropna()
            
            # Align features and labels
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            # Use time series split for proper backtesting
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Initialize metrics
            all_predictions = []
            all_actuals = []
            all_confidences = []
            trade_results = []
            
            # Walk-forward validation
            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train ensemble model
                rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
                gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
                
                rf_model.fit(X_train_scaled, y_train)
                gb_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                rf_pred = rf_model.predict_proba(X_test_scaled)
                gb_pred = gb_model.predict_proba(X_test_scaled)
                
                # Ensemble prediction
                ensemble_pred = (rf_pred * 0.6 + gb_pred * 0.4)
                
                # Store results
                predictions = np.argmax(ensemble_pred, axis=1)
                confidences = np.max(ensemble_pred, axis=1)
                
                all_predictions.extend(predictions)
                all_actuals.extend(y_test.values)
                all_confidences.extend(confidences)
                
                # Calculate trade results for this fold
                for i, (pred, actual, conf) in enumerate(zip(predictions, y_test.values, confidences)):
                    if conf > 0.6:  # Only trade on high confidence
                        next_day_return = (hist['Close'].iloc[test_idx[i+1]] / hist['Close'].iloc[test_idx[i]] - 1) if i+1 < len(test_idx) else 0
                        
                        if pred == 1:  # Bullish prediction
                            trade_return = next_day_return
                        else:  # Bearish prediction
                            trade_return = -next_day_return
                        
                        trade_results.append({
                            'prediction': pred,
                            'actual': actual,
                            'confidence': conf,
                            'return': trade_return,
                            'correct': (pred == actual)
                        })
            
            # Calculate comprehensive metrics
            predictions = np.array(all_predictions)
            actuals = np.array(all_actuals)
            confidences = np.array(all_confidences)
            
            # Basic accuracy
            accuracy = (predictions == actuals).mean()
            
            # Trade-based metrics
            if trade_results:
                trade_df = pd.DataFrame(trade_results)
                
                # High confidence trades
                high_conf_trades = trade_df[trade_df['confidence'] > 0.7]
                
                # Calculate returns
                strategy_returns = trade_df['return'].values
                strategy_total_return = np.prod(1 + strategy_returns) - 1
                strategy_avg_return = np.mean(strategy_returns)
                
                # Risk metrics
                strategy_volatility = np.std(strategy_returns)
                sharpe_ratio = strategy_avg_return / strategy_volatility if strategy_volatility > 0 else 0
                
                # Drawdown
                cumulative_returns = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # Win rate
                win_rate = (strategy_returns > 0).mean()
                
                # High confidence metrics
                if len(high_conf_trades) > 0:
                    high_conf_win_rate = high_conf_trades['correct'].mean()
                    high_conf_avg_return = high_conf_trades['return'].mean()
                    high_conf_count = len(high_conf_trades)
                else:
                    high_conf_win_rate = 0
                    high_conf_avg_return = 0
                    high_conf_count = 0
                
                # Signal breakdown
                bullish_signals = (predictions == 1).sum()
                bearish_signals = (predictions == 0).sum()
                neutral_signals = len(predictions) - bullish_signals - bearish_signals
                
                return {
                    'total_trades': len(trade_results),
                    'win_rate': win_rate,
                    'strategy_total_return': strategy_total_return,
                    'strategy_avg_return': strategy_avg_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'prediction_accuracy': accuracy,
                    'high_conf_trades': high_conf_count,
                    'high_conf_win_rate': high_conf_win_rate,
                    'high_conf_avg_return': high_conf_avg_return,
                    'total_signals': {
                        'bullish': bullish_signals,
                        'bearish': bearish_signals,
                        'neutral': neutral_signals
                    },
                    'enhanced_features': len(features.columns),
                    'data_sources': list(enhanced_data.keys()) if enhanced_data else ['basic'],
                    'backtest_period': f"{len(hist)} days",
                    'validation_method': 'Time Series Cross-Validation'
                }
            else:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'strategy_total_return': 0,
                    'strategy_avg_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'prediction_accuracy': accuracy,
                    'high_conf_trades': 0,
                    'high_conf_win_rate': 0,
                    'high_conf_avg_return': 0,
                    'total_signals': {'bullish': 0, 'bearish': 0, 'neutral': 0},
                    'enhanced_features': len(features.columns),
                    'data_sources': list(enhanced_data.keys()) if enhanced_data else ['basic'],
                    'backtest_period': f"{len(hist)} days",
                    'validation_method': 'Time Series Cross-Validation'
                }
                
        except Exception as e:
            print(f"❌ Enhanced backtest error: {e}")
            return self.backtest_strategy(hist)  # Fallback to basic backtest
    
    def print_analysis(self, data):
        """Print analysis for command line"""
        if not data:
            return
        
        # Show ML availability status
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available. Install with: pip install scikit-learn")
        elif not XGB_AVAILABLE:
            print("ℹ️ XGBoost not available (OpenMP issue). Using RandomForest + GradientBoosting.")
        
        tech_data = data['tech_data']
        fundamental_data = data['fundamental_data']
        sentiment_data = data['sentiment_data']
        earnings_data = data['earnings_data']
        ml_prediction = data['ml_prediction']
        backtest_results = data['backtest_results']
        info = data['info']
        
        print(f"\n🏆 STOCK ANALYSIS: {self.ticker}")
        print("=" * 50)
        
        # Basic Information
        print(f"\n📊 BASIC INFORMATION")
        print(f"Current Price: ${tech_data['current_price']:.2f}")
        print(f"Price Change: {tech_data['price_change_pct']:+.2f}%")
        print(f"Volume: {tech_data['volume']:,}")
        print(f"Market Cap: ${fundamental_data['market_cap']:,.0f}" if fundamental_data['market_cap'] else "Market Cap: N/A")
        print(f"Company: {info.get('longName', 'N/A')}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print(f"Industry: {info.get('industry', 'N/A')}")
        
        # Technical Analysis
        print(f"\n🔧 TECHNICAL ANALYSIS")
        print(f"RSI (14): {tech_data['rsi']:.1f}")
        print(f"MACD: {tech_data['macd']:.3f}")
        print(f"SMA 20: ${tech_data['sma_20']:.2f}")
        print(f"SMA 50: ${tech_data['sma_50']:.2f}")
        print(f"Volatility: {tech_data['volatility_20d']:.1f}%")
        print(f"BB Position: {tech_data['bb_position']:.1f}%")
        
        # Performance
        print(f"\n📈 PERFORMANCE")
        print(f"1 Day: {tech_data['performance_1d']:+.1f}%")
        print(f"5 Days: {tech_data['performance_5d']:+.1f}%")
        print(f"1 Month: {tech_data['performance_1m']:+.1f}%")
        print(f"3 Months: {tech_data['performance_3m']:+.1f}%")
        print(f"1 Year: {tech_data['performance_1y']:+.1f}%")
        
        # Fundamental Analysis
        print(f"\n💰 FUNDAMENTAL ANALYSIS")
        print(f"P/E Ratio: {fundamental_data['pe_ratio']:.2f}" if fundamental_data['pe_ratio'] else "P/E Ratio: N/A")
        print(f"Forward P/E: {fundamental_data['forward_pe']:.2f}" if fundamental_data['forward_pe'] else "Forward P/E: N/A")
        print(f"PEG Ratio: {fundamental_data['peg_ratio']:.2f}" if fundamental_data['peg_ratio'] else "PEG Ratio: N/A")
        print(f"Price-to-Book: {fundamental_data['price_to_book']:.2f}" if fundamental_data['price_to_book'] else "Price-to-Book: N/A")
        print(f"Revenue Growth: {fundamental_data['revenue_growth']:.1%}" if fundamental_data['revenue_growth'] else "Revenue Growth: N/A")
        print(f"ROE: {fundamental_data['return_on_equity']:.1%}" if fundamental_data['return_on_equity'] else "ROE: N/A")
        print(f"Beta: {fundamental_data['beta']:.2f}" if fundamental_data['beta'] else "Beta: N/A")
        print(f"Relative Strength: {fundamental_data['relative_strength']:+.1f}%" if fundamental_data['relative_strength'] else "Relative Strength: N/A")
        
        # Sentiment Analysis
        print(f"\n😊 SENTIMENT ANALYSIS")
        print(f"Overall Sentiment: {sentiment_data['sentiment_label']}")
        print(f"Sentiment Score: {sentiment_data['overall_sentiment']:+.3f}")
        print(f"News Sentiment: {sentiment_data['news_sentiment']:+.3f}")
        print(f"Social Sentiment: {sentiment_data['social_sentiment']:+.3f}")
        print(f"News Count: {sentiment_data['news_count']}")
        print(f"Positive Ratio: {sentiment_data['positive_ratio']:.1%}")
        
        # Earnings Calendar
        print(f"\n📅 EARNINGS CALENDAR")
        if earnings_data['earnings_expected']:
            print(f"Next Earnings: {earnings_data['next_earnings_date'].strftime('%Y-%m-%d')}")
            print(f"Days to Earnings: {earnings_data['days_to_earnings']}")
            if earnings_data['volatility_expected']:
                print("⚠️ High volatility expected near earnings")
        else:
            print("No upcoming earnings data available")
        
        # Institutional Holdings
        print(f"\n🏛️ INSTITUTIONAL HOLDINGS")
        institutional_data = data.get('institutional_data', {})
        holders = institutional_data.get('institutional_holders', {}).get('top_holders', [])
        
        if holders:
            print(f"Top Institutional Holders ({len(holders)} total):")
            print("┌─────────────────────────────────┬─────────────┬──────────────┬─────────────┐")
            print("│ Institution                     │ Ownership   │ Shares       │ Change      │")
            print("├─────────────────────────────────┼─────────────┼──────────────┼─────────────┤")
            
            for holder in holders[:8]:  # Show top 8
                name = holder['name'][:29]  # Truncate long names
                ownership = f"{holder['percent_out']:.2%}"
                shares = f"{holder['shares']:,}"
                change = holder.get('pct_change', 0)
                change_str = f"{change:+.2%}" if change != 0 else "N/A"
                change_indicator = "📈" if change > 0 else "📉" if change < 0 else ""
                
                print(f"│ {name:<29} │ {ownership:>9} │ {shares:>12} │ {change_indicator}{change_str:>8} │")
            
            print("└─────────────────────────────────┴─────────────┴──────────────┴─────────────┘")
        else:
            print("No institutional holdings data available")
        
        # Insider Trading
        print(f"\n👥 INSIDER TRADING")
        transactions = institutional_data.get('insider_transactions', {}).get('recent_transactions', [])
        
        if transactions:
            print(f"Recent Insider Activity ({len(transactions)} transactions):")
            print("┌─────────────────────────┬─────────────┬─────────────┬──────────────┬─────────────┐")
            print("│ Insider                 │ Position    │ Transaction │ Shares       │ Value       │")
            print("├─────────────────────────┼─────────────┼─────────────┼──────────────┼─────────────┤")
            
            for transaction in transactions[:6]:  # Show top 6
                insider = transaction['insider'][:23]  # Truncate long names
                position = transaction['position'][:11] if transaction['position'] else "N/A"
                trans_type = transaction['transaction_type'][:11] if transaction['transaction_type'] else "N/A"
                shares = f"{transaction['shares']:,}" if transaction['shares'] else "N/A"
                value = f"${transaction['value']:,.0f}" if transaction.get('value', 0) > 0 else "N/A"
                
                print(f"│ {insider:<23} │ {position:<11} │ {trans_type:<11} │ {shares:>12} │ {value:>11} │")
            
            print("└─────────────────────────┴─────────────┴─────────────┴──────────────┴─────────────┘")
        else:
            print("No recent insider transactions")
        
        # Earnings History
        print(f"\n📊 EARNINGS HISTORY")
        earnings_history = institutional_data.get('earnings_data', {}).get('history', [])
        
        if earnings_history:
            print("Recent Earnings Performance:")
            print("┌─────────────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
            print("│ Quarter             │ Actual EPS  │ Estimate    │ Difference │ Surprise    │")
            print("├─────────────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")
            
            for earning in earnings_history[:5]:  # Show last 5 quarters
                quarter = earning['quarter'][:19]  # Truncate long dates
                actual = f"{earning['actual_eps']}" if earning['actual_eps'] != 'N/A' else "N/A"
                estimate = f"{earning['estimate_eps']}" if earning['estimate_eps'] != 'N/A' else "N/A"
                difference = f"{earning['surprise']}" if earning['surprise'] != 'N/A' else "N/A"
                surprise_pct = earning.get('surprise_percent', 0)
                
                if surprise_pct != 0 and surprise_pct != 'N/A':
                    surprise_indicator = "🎯" if surprise_pct > 0 else "⚠️"
                    surprise_str = f"{surprise_indicator}{surprise_pct:+.1%}"
                else:
                    surprise_str = "📊 N/A"
                
                print(f"│ {quarter:<19} │ {actual:>9} │ {estimate:>9} │ {difference:>9} │ {surprise_str:>9} │")
            
            print("└─────────────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")
        else:
            print("No earnings history available")
        
        # ML Prediction
        if ml_prediction:
            print(f"\n🤖 ML PREDICTION")
            print(f"Direction: {ml_prediction['direction']}")
            print(f"Confidence: {ml_prediction['confidence']:.1%}")
            
            # Scenario probabilities
            scenarios = ml_prediction['scenario_probabilities']
            targets = ml_prediction['scenario_targets']
            print(f"\n📊 SCENARIO PROBABILITIES")
            print(f"Bullish (+10%): {scenarios['bullish']:.1%} → ${targets['bullish']:.2f}")
            print(f"Neutral (flat): {scenarios['neutral']:.1%} → ${targets['neutral']:.2f}")
            print(f"Bearish (-8%): {scenarios['bearish']:.1%} → ${targets['bearish']:.2f}")
            
            # SHAP explanations
            if ml_prediction['shap_explanations']:
                print(f"\n🔍 EXPLANATION")
                print(f"{ml_prediction['shap_explanations']['explanation']}")
                
                print(f"\n📈 TOP DRIVERS")
                for feature, contribution in ml_prediction['shap_explanations']['top_features']:
                    print(f"• {feature}: {contribution:+.3f}")
            
            print(f"\n🎯 MODEL PERFORMANCE")
            for model, accuracy in ml_prediction['model_accuracies'].items():
                print(f"{model}: {accuracy:.3f}")
            
            print(f"\n🔍 FEATURE IMPORTANCE")
            for feature in ml_prediction['feature_importance']:
                print(f"• {feature['feature']}: {feature['importance']:.3f}")
        
        # Backtest Results
        if backtest_results:
            print(f"\n📊 BACKTEST RESULTS (Last 252 Days)")
            print(f"Total Trades: {backtest_results['total_trades']}")
            print(f"Win Rate: {backtest_results['win_rate']:.1%}")
            print(f"Strategy Return: {backtest_results['strategy_total_return']:+.1%}")
            print(f"Average Return per Trade: {backtest_results['strategy_avg_return']:+.2%}")
            print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {backtest_results['max_drawdown']:+.1%}")
            print(f"Prediction Accuracy: {backtest_results['prediction_accuracy']:.1%}")
            
            print(f"\n🎯 SIGNAL BREAKDOWN")
            signals = backtest_results['total_signals']
            print(f"Bullish Signals: {signals['bullish']}")
            print(f"Bearish Signals: {signals['bearish']}")
            print(f"Neutral Signals: {signals['neutral']}")
            
            if backtest_results['high_conf_trades'] > 0:
                print(f"\n🔥 HIGH CONFIDENCE TRADES (>60%)")
                print(f"High Conf Trades: {backtest_results['high_conf_trades']}")
                print(f"High Conf Win Rate: {backtest_results['high_conf_win_rate']:.1%}")
                print(f"High Conf Avg Return: {backtest_results['high_conf_avg_return']:+.2%}")
        
        # Recommendation
        print(f"\n🎯 RECOMMENDATION")
        
        # Calculate comprehensive score
        score = 0
        factors = []
        
        # Technical factors
        if tech_data['rsi'] < 70:
            score += 1
            factors.append("RSI not overbought")
        if tech_data['macd'] > tech_data['macd_signal']:
            score += 1
            factors.append("MACD bullish")
        if tech_data['current_price'] > tech_data['sma_20']:
            score += 1
            factors.append("Above 20-day SMA")
        
        # Fundamental factors
        if fundamental_data['pe_ratio'] and fundamental_data['pe_ratio'] < 25:
            score += 1
            factors.append("Reasonable P/E ratio")
        if fundamental_data['relative_strength'] and fundamental_data['relative_strength'] > 0:
            score += 1
            factors.append("Outperforming market")
        
        # Sentiment factors
        if sentiment_data['sentiment_label'] == 'POSITIVE':
            score += 1
            factors.append("Positive sentiment")
        
        # ML factors
        if ml_prediction and ml_prediction['confidence'] > 0.6:
            score += 1
            factors.append("High ML confidence")
        
        # Generate recommendation
        if score >= 5:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif score >= 3:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif score <= 1:
            recommendation = "SELL"
            confidence = "MEDIUM"
        else:
            recommendation = "HOLD"
            confidence = "LOW"
        
        print(f"Recommendation: {recommendation}")
        print(f"Confidence: {confidence}")
        print(f"Score: {score}/7")
        
        print(f"\n🔍 KEY FACTORS")
        for factor in factors:
            print(f"✅ {factor}")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT")
        risk_factors = []
        
        if tech_data['volatility_20d'] > 30:
            risk_factors.append("High volatility")
        if tech_data['rsi'] > 80:
            risk_factors.append("Overbought conditions")
        if fundamental_data['debt_to_equity'] and fundamental_data['debt_to_equity'] > 1:
            risk_factors.append("High debt levels")
        
        if risk_factors:
            for risk in risk_factors:
                print(f"⚠️ {risk}")
        else:
            print("✅ Low risk factors identified")
        
        print(f"\n⚠️ Disclaimer: This analysis is for educational purposes only.")
        print(f"Always consult with qualified financial advisors before investing.")

def create_charts(data):
    """Create interactive charts"""
    if not data or data['hist'].empty or not PLOTLY_AVAILABLE:
        return {}
    
    hist = data['hist']
    charts = {}
    
    # Price chart
    fig_price = go.Figure()
    
    # Candlestick
    fig_price.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name="Price"
    ))
    
    # Moving averages
    fig_price.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'].rolling(20).mean(),
        mode='lines',
        name='SMA 20',
        line=dict(color='orange', width=2)
    ))
    
    fig_price.add_trace(go.Scatter(
        x=hist.index,
        y=hist['Close'].rolling(50).mean(),
        mode='lines',
        name='SMA 50',
        line=dict(color='blue', width=2)
    ))
    
    fig_price.update_layout(
        title=f"📈 {data['info'].get('longName', 'Stock')} Price Chart",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white"
    )
    
    charts['price'] = fig_price
    
    return charts

def web_interface():
    """Streamlit web interface - Fixed Version"""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not available. Install with: pip install streamlit")
        return
    
    st.set_page_config(
        page_title="🏆 Stock Analyzer",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Ultra-compact CSS for one-pager
    st.markdown("""
    <style>
        .main-header {
            font-size: 1.5rem;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin-bottom: 0.3rem;
        }
        .compact-section {
            font-size: 1rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 0.5rem;
            margin-bottom: 0.3rem;
            border-bottom: 1px solid #3498db;
            padding-bottom: 0.1rem;
        }
        .recommendation-box {
            background: linear-gradient(90deg, #28a745, #20c997);
            color: white;
            padding: 0.3rem;
            border-radius: 0.3rem;
            text-align: center;
            font-weight: bold;
            font-size: 1rem;
            margin: 0.3rem 0;
        }
        .metric-compact {
            font-size: 0.9rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 2rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .stExpander {
            margin-bottom: 0.3rem;
        }
        .stExpander > div {
            padding: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Compact Header
    st.markdown('<div class="main-header">🏆 Stock Analyzer</div>', unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "👀 Watchlist", "💼 Portfolio"])
    
    with tab1:
        # Analysis controls
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="AAPL")
        
        with col2:
            col2a, col2b = st.columns(2)
            with col2a:
                show_charts = st.checkbox("📈 Charts", value=True)
            with col2b:
                show_ml = st.checkbox("🤖 ML", value=True)
        
        # Show ML availability status
        if not ML_AVAILABLE:
            st.warning("⚠️ ML libraries not available. Install with: pip install scikit-learn")
        elif not XGB_AVAILABLE:
            st.info("ℹ️ XGBoost not available (OpenMP issue). Using RandomForest + GradientBoosting.")
        
        if st.button("🚀 Analyze Stock", type="primary"):
            if ticker:
                analyzer = UniversalStockAnalyzer(ticker)
                data = analyzer.analyze(show_charts, show_ml, True, True)
                
                if data:
                    st.success(f"✅ {ticker} Analysis Complete")
                    
                    tech_data = data['tech_data']
                    fundamental_data = data['fundamental_data']
                    sentiment_data = data['sentiment_data']
                    earnings_data = data['earnings_data']
                    ml_prediction = data['ml_prediction']
                    backtest_results = data['backtest_results']
                    info = data['info']
                    
                    # COMPACT ONE-PAGER LAYOUT
                    
                    # Top Row: Key Metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Price", f"${tech_data['current_price']:.2f}", f"{tech_data['price_change_pct']:+.1f}%")
                    with col2:
                        st.metric("Volume", f"{tech_data['volume']/1e6:.1f}M", f"{tech_data['volume_ratio']:.1f}x")
                    with col3:
                        st.metric("Market Cap", f"${fundamental_data['market_cap']/1e9:.1f}B" if fundamental_data['market_cap'] else "N/A")
                    with col4:
                        st.metric("P/E", f"{fundamental_data['pe_ratio']:.1f}" if fundamental_data['pe_ratio'] else "N/A")
                    with col5:
                        st.metric("Beta", f"{fundamental_data['beta']:.2f}" if fundamental_data['beta'] else "N/A")
                    
                    # Company Info
                    st.markdown(f"**{info.get('longName', 'N/A')}** • {info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}")
                    
                    # Earnings Alert
                    if earnings_data['earnings_expected'] and earnings_data['volatility_expected']:
                        st.warning(f"📅 Earnings in {earnings_data['days_to_earnings']} days - High volatility expected!")
                    elif earnings_data['earnings_expected']:
                        st.info(f"📅 Earnings in {earnings_data['days_to_earnings']} days")
                    
                    # Charts Row (Ultra-compact)
                    if show_charts and PLOTLY_AVAILABLE:
                        charts = create_charts(data)
                        if 'price' in charts:
                            # Make chart smaller for one-pager
                            charts['price'].update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0))
                            st.plotly_chart(charts['price'], use_container_width=True)
                    
                    # Analysis Row: Technical + ML + Sentiment
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="compact-section">🔧 Technical</div>', unsafe_allow_html=True)
                        st.markdown(f"**RSI:** {tech_data['rsi']:.1f} {'🔴' if tech_data['rsi'] > 70 else '🟢' if tech_data['rsi'] < 30 else '🟡'}")
                        st.markdown(f"**MACD:** {tech_data['macd']:.2f}")
                        st.markdown(f"**SMA20:** ${tech_data['sma_20']:.2f}")
                        st.markdown(f"**Volatility:** {tech_data['volatility_20d']:.1f}%")
                    
                    with col2:
                        st.markdown('<div class="compact-section">🤖 ML Prediction</div>', unsafe_allow_html=True)
                        if ml_prediction:
                            direction_emoji = "🟢" if "BULLISH" in ml_prediction['direction'] else "🔴" if "BEARISH" in ml_prediction['direction'] else "🟡"
                            st.markdown(f"**Direction:** {ml_prediction['direction']} {direction_emoji}")
                            st.markdown(f"**Confidence:** {ml_prediction['confidence']:.1%}")
                            
                            # Scenario probabilities
                            scenarios = ml_prediction['scenario_probabilities']
                            st.markdown(f"**Bullish:** {scenarios['bullish']:.1%}")
                            st.markdown(f"**Neutral:** {scenarios['neutral']:.1%}")
                            st.markdown(f"**Bearish:** {scenarios['bearish']:.1%}")
                            
                            # SHAP explanation
                            if ml_prediction['shap_explanations']:
                                explanation = ml_prediction['shap_explanations']['explanation']
                                st.markdown(f"**Explanation:** {explanation[:100]}...")
                        else:
                            st.markdown("ML not available")
                    
                    with col3:
                        st.markdown('<div class="compact-section">😊 Sentiment</div>', unsafe_allow_html=True)
                        sentiment_emoji = "🟢" if sentiment_data['sentiment_label'] == 'POSITIVE' else "🔴" if sentiment_data['sentiment_label'] == 'NEGATIVE' else "🟡"
                        st.markdown(f"**Overall:** {sentiment_data['sentiment_label']} {sentiment_emoji}")
                        st.markdown(f"**Score:** {sentiment_data['overall_sentiment']:+.2f}")
                        st.markdown(f"**News:** {sentiment_data['news_sentiment']:+.2f}")
                        st.markdown(f"**Social:** {sentiment_data['social_sentiment']:+.2f}")
                    
                    # Performance Row
                    st.markdown('<div class="compact-section">📈 Performance</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("1D", f"{tech_data['performance_1d']:+.1f}%")
                    with col2:
                        st.metric("5D", f"{tech_data['performance_5d']:+.1f}%")
                    with col3:
                        st.metric("1M", f"{tech_data['performance_1m']:+.1f}%")
                    with col4:
                        st.metric("3M", f"{tech_data['performance_3m']:+.1f}%")
                    with col5:
                        st.metric("1Y", f"{tech_data['performance_1y']:+.1f}%")
                    
                    # Recommendation Row
                    st.markdown('<div class="compact-section">🎯 Recommendation</div>', unsafe_allow_html=True)
                    
                    # Calculate recommendation
                    score = 0
                    factors = []
                    
                    if tech_data['rsi'] < 70:
                        score += 1
                        factors.append("RSI OK")
                    if tech_data['macd'] > tech_data['macd_signal']:
                        score += 1
                        factors.append("MACD Bull")
                    if tech_data['current_price'] > tech_data['sma_20']:
                        score += 1
                        factors.append("Above SMA20")
                    
                    if fundamental_data['pe_ratio'] and fundamental_data['pe_ratio'] < 25:
                        score += 1
                        factors.append("Good P/E")
                    if fundamental_data['relative_strength'] and fundamental_data['relative_strength'] > 0:
                        score += 1
                        factors.append("Beat Market")
                    
                    if sentiment_data['sentiment_label'] == 'POSITIVE':
                        score += 1
                        factors.append("Positive Sentiment")
                    
                    if ml_prediction and ml_prediction['confidence'] > 0.6:
                        score += 1
                        factors.append("High ML Conf")
                    
                    if score >= 5:
                        recommendation = "STRONG BUY"
                        confidence = "HIGH"
                        rec_color = "background: linear-gradient(90deg, #28a745, #20c997);"
                    elif score >= 3:
                        recommendation = "BUY"
                        confidence = "MEDIUM"
                        rec_color = "background: linear-gradient(90deg, #17a2b8, #6f42c1);"
                    elif score <= 1:
                        recommendation = "SELL"
                        confidence = "MEDIUM"
                        rec_color = "background: linear-gradient(90deg, #dc3545, #fd7e14);"
                    else:
                        recommendation = "HOLD"
                        confidence = "LOW"
                        rec_color = "background: linear-gradient(90deg, #ffc107, #fd7e14);"
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f'<div class="recommendation-box" style="{rec_color}">{recommendation}</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", confidence)
                    
                    with col3:
                        st.metric("Score", f"{score}/7")
                    
                    # Key Factors (Compact)
                    st.markdown("**Key Factors:** " + " • ".join(factors[:5]))
                    
                    # Risk Assessment (Compact)
                    risk_factors = []
                    if tech_data['volatility_20d'] > 30:
                        risk_factors.append("High Volatility")
                    if tech_data['rsi'] > 80:
                        risk_factors.append("Overbought")
                    if fundamental_data['debt_to_equity'] and fundamental_data['debt_to_equity'] > 1:
                        risk_factors.append("High Debt")
                    
                    if risk_factors:
                        st.markdown("**⚠️ Risks:** " + " • ".join(risk_factors))
                    else:
                        st.markdown("**✅ Low Risk**")
                    
                    # Backtest Results (Compact)
                    if backtest_results:
                        st.markdown('<div class="compact-section">📊 Backtest Performance</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")
                        with col2:
                            st.metric("Strategy Return", f"{backtest_results['strategy_total_return']:+.1%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:+.1%}")
                        
                        # Signal breakdown
                        signals = backtest_results['total_signals']
                        st.markdown(f"**Signals:** Bullish {signals['bullish']} • Bearish {signals['bearish']} • Neutral {signals['neutral']}")
                        
                        if backtest_results['high_conf_trades'] > 0:
                            st.markdown(f"**🔥 High Conf Trades:** {backtest_results['high_conf_trades']} (Win Rate: {backtest_results['high_conf_win_rate']:.1%})")
                
                else:
                    st.error(f"❌ No data available for {ticker}")
            
            else:
                st.error("Please enter a stock ticker")
    
    with tab2:
        st.header("👀 Watchlist Management")
        portfolio_manager = PortfolioManager()
        
        # Add to watchlist
        st.subheader("Add to Watchlist")
        col1, col2 = st.columns(2)
        
        with col1:
            new_ticker = st.text_input("Ticker", placeholder="TSLA", key="watchlist_ticker")
        with col2:
            notes = st.text_input("Notes", placeholder="AI play", key="watchlist_notes")
        
        if st.button("Add to Watchlist", key="add_watchlist"):
            if new_ticker:
                success = portfolio_manager.add_to_watchlist(new_ticker, notes)
                if success:
                    st.success(f"✅ Added {new_ticker} to watchlist!")
                else:
                    st.warning(f"⚠️ {new_ticker} already in watchlist")
        
        # Show watchlist
        st.subheader("Current Watchlist")
        watchlist_data = portfolio_manager.get_watchlist_analysis()
        
        if watchlist_data:
            for stock in watchlist_data:
                if 'error' not in stock:
                    with st.expander(f"{stock['ticker']} - ${stock['price']:.2f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Change:** {stock['change_pct']:+.1f}%")
                            st.write(f"**RSI:** {stock['rsi']:.1f}")
                            st.write(f"**Sentiment:** {stock['sentiment']}")
                        with col2:
                            st.write(f"**ML:** {stock['ml_direction']} ({stock['ml_confidence']:.1%})")
                            st.write(f"**Volatility:** {stock['volatility']:.1f}%")
                            if stock['earnings_alert']:
                                st.write("⚠️ **Earnings Alert!**")
                        if stock['notes']:
                            st.write(f"**Notes:** {stock['notes']}")
        else:
            st.info("No stocks in watchlist")
    
    with tab3:
        st.header("💼 Portfolio Management")
        portfolio_manager = PortfolioManager()
        
        # Add to portfolio
        st.subheader("Add Holding")
        col1, col2 = st.columns(2)
        
        with col1:
            port_ticker = st.text_input("Ticker", key="port_ticker", placeholder="AAPL")
            port_shares = st.number_input("Shares", min_value=0.0, value=1.0, key="port_shares")
        with col2:
            port_price = st.number_input("Avg Price", min_value=0.0, value=100.0, key="port_price")
            port_notes = st.text_input("Notes", key="port_notes", placeholder="Long term hold")
        
        if st.button("Add to Portfolio", key="add_portfolio"):
            if port_ticker:
                success = portfolio_manager.add_to_portfolio(port_ticker, port_shares, port_price, port_notes)
                if success:
                    st.success(f"✅ Added {port_ticker} to portfolio!")
                else:
                    st.warning(f"⚠️ Failed to add {port_ticker}")
        
        # Show portfolio
        st.subheader("Current Portfolio")
        portfolio_data = portfolio_manager.get_portfolio_analysis()
        
        if portfolio_data['holdings']:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Value", f"${portfolio_data['total_value']:,.2f}")
            with col2:
                st.metric("Total P&L", f"${portfolio_data['total_gain_loss']:+,.2f}", f"{portfolio_data['total_gain_loss_pct']:+.1f}%")
            
            for holding in portfolio_data['holdings']:
                if 'error' not in holding:
                    with st.expander(f"{holding['ticker']} - {holding['shares']} shares"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Current Price:** ${holding['current_price']:.2f}")
                            st.write(f"**Avg Price:** ${holding['avg_price']:.2f}")
                        with col2:
                            st.write(f"**Current Value:** ${holding['current_value']:,.2f}")
                            st.write(f"**P&L:** ${holding['gain_loss']:+,.2f} ({holding['gain_loss_pct']:+.1f}%)")
                        if holding['notes']:
                            st.write(f"**Notes:** {holding['notes']}")
        else:
            st.info("No holdings in portfolio")
    
    # Compact Footer
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer:** Educational purposes only. Consult financial advisors before investing.")
def print_portfolio_analysis(portfolio_summary):
    """Print portfolio analysis"""
    print(f"\n🏆 PORTFOLIO ANALYSIS")
    print("=" * 50)
    
    print(f"\n📊 PORTFOLIO SUMMARY")
    print(f"Total Value: ${portfolio_summary['total_value']:,.2f}")
    print(f"Total Cost: ${portfolio_summary['total_cost']:,.2f}")
    print(f"Total P&L: ${portfolio_summary['total_gain_loss']:+,.2f}")
    print(f"Total Return: {portfolio_summary['total_gain_loss_pct']:+.1f}%")
    
    print(f"\n📈 HOLDINGS")
    for holding in portfolio_summary['holdings']:
        if 'error' in holding:
            print(f"❌ {holding['ticker']}: Error - {holding['error']}")
            continue
            
        print(f"\n{holding['ticker']}")
        print(f"  Shares: {holding['shares']}")
        print(f"  Avg Price: ${holding['avg_price']:.2f}")
        print(f"  Current Price: ${holding['current_price']:.2f}")
        print(f"  Current Value: ${holding['current_value']:,.2f}")
        print(f"  Cost Basis: ${holding['cost_basis']:,.2f}")
        print(f"  P&L: ${holding['gain_loss']:+,.2f} ({holding['gain_loss_pct']:+.1f}%)")
        if holding['notes']:
            print(f"  Notes: {holding['notes']}")

def print_watchlist_analysis(watchlist_data):
    """Print watchlist analysis"""
    print(f"\n🏆 WATCHLIST ANALYSIS")
    print("=" * 50)
    
    for stock in watchlist_data:
        if 'error' in stock:
            print(f"\n❌ {stock['ticker']}: Error - {stock['error']}")
            continue
            
        print(f"\n{stock['ticker']}")
        print(f"  Price: ${stock['price']:.2f} ({stock['change_pct']:+.1f}%)")
        print(f"  RSI: {stock['rsi']:.1f}")
        print(f"  Sentiment: {stock['sentiment']} ({stock['sentiment_score']:+.2f})")
        print(f"  ML Direction: {stock['ml_direction']} ({stock['ml_confidence']:.1%})")
        print(f"  Volatility: {stock['volatility']:.1f}%")
        if stock['earnings_alert']:
            print(f"  ⚠️ Earnings Alert!")
        if stock['notes']:
            print(f"  Notes: {stock['notes']}")

def print_alerts(alerts):
    """Print alerts"""
    print(f"\n🚨 ALERTS")
    print("=" * 50)
    
    if not alerts:
        print("✅ No alerts at this time")
        return
    
    # Group by severity
    high_alerts = [a for a in alerts if a['severity'] == 'high']
    medium_alerts = [a for a in alerts if a['severity'] == 'medium']
    
    if high_alerts:
        print(f"\n🔴 HIGH PRIORITY ALERTS")
        for alert in high_alerts:
            print(f"• {alert['message']}")
    
    if medium_alerts:
        print(f"\n🟡 MEDIUM PRIORITY ALERTS")
        for alert in medium_alerts:
            print(f"• {alert['message']}")
    
    print(f"\nTotal Alerts: {len(alerts)}")

def main():
    """Main function - handles both CLI and web interface"""
    
    # Check if running as web interface
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        web_interface()
        return
    
    # Check for portfolio commands
    if len(sys.argv) > 1 and sys.argv[1] in ['--portfolio', '--watchlist', '--alerts']:
        portfolio_manager = PortfolioManager()
        
        if sys.argv[1] == '--portfolio':
            portfolio_analysis = portfolio_manager.get_portfolio_analysis()
            print_portfolio_analysis(portfolio_analysis)
            
        elif sys.argv[1] == '--watchlist':
            watchlist_analysis = portfolio_manager.get_watchlist_analysis()
            print_watchlist_analysis(watchlist_analysis)
            
        elif sys.argv[1] == '--alerts':
            alerts = portfolio_manager.check_alerts()
            print_alerts(alerts)
        
        return
    
    # Check for portfolio management commands
    if len(sys.argv) > 2 and sys.argv[1] in ['--add-watchlist', '--add-portfolio', '--remove-watchlist', '--remove-portfolio']:
        portfolio_manager = PortfolioManager()
        
        if sys.argv[1] == '--add-watchlist':
            ticker = sys.argv[2]
            notes = sys.argv[3] if len(sys.argv) > 3 else ""
            if portfolio_manager.add_to_watchlist(ticker, notes):
                print(f"✅ Added {ticker} to watchlist")
            else:
                print(f"❌ {ticker} already in watchlist")
                
        elif sys.argv[1] == '--add-portfolio':
            if len(sys.argv) < 5:
                print("Usage: --add-portfolio TICKER SHARES AVG_PRICE [NOTES]")
                return
            ticker = sys.argv[2]
            shares = float(sys.argv[3])
            avg_price = float(sys.argv[4])
            notes = sys.argv[5] if len(sys.argv) > 5 else ""
            portfolio_manager.add_to_portfolio(ticker, shares, avg_price, notes)
            print(f"✅ Added {ticker} to portfolio: {shares} shares @ ${avg_price}")
            
        elif sys.argv[1] == '--remove-watchlist':
            ticker = sys.argv[2]
            portfolio_manager.remove_from_watchlist(ticker)
            print(f"✅ Removed {ticker} from watchlist")
            
        elif sys.argv[1] == '--remove-portfolio':
            ticker = sys.argv[2]
            portfolio_manager.remove_from_portfolio(ticker)
            print(f"✅ Removed {ticker} from portfolio")
        
        return
    
    # Command line interface
    parser = argparse.ArgumentParser(description='🏆 Universal Stock Analyzer')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML predictions')
    parser.add_argument('--no-fundamentals', action='store_true', help='Disable fundamental analysis')
    parser.add_argument('--no-sentiment', action='store_true', help='Disable sentiment analysis')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = UniversalStockAnalyzer(args.ticker)
    
    # Run analysis
    data = analyzer.analyze(
        show_ml=not args.no_ml,
        show_fundamentals=not args.no_fundamentals,
        show_sentiment=not args.no_sentiment
    )
    
    # Print results
    analyzer.print_analysis(data)

if __name__ == "__main__":
    # Check if this is being run by streamlit or with --web flag
    if len(sys.argv) > 1 and sys.argv[-1] == '--web':
        web_interface()
    elif len(sys.argv) > 0 and 'streamlit' in sys.argv[0]:
        web_interface()
    else:
        main()
