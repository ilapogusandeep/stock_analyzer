#!/usr/bin/env python3
"""
Enhanced Price Prediction System
Combines multiple data sources for comprehensive price forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EnhancedPricePredictor:
    """Advanced price prediction using multiple data sources"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(ticker)
        
    def get_comprehensive_prediction_data(self):
        """Collect all data sources for price prediction"""
        try:
            # Get enhanced data
            from enhanced_data_collector import EnhancedDataCollector
            from enhanced_institutional_data import EnhancedInstitutionalData
            
            enhanced_collector = EnhancedDataCollector(self.ticker)
            enhanced_data = enhanced_collector.get_comprehensive_data()
            
            institutional_collector = EnhancedInstitutionalData(self.ticker)
            institutional_data = institutional_collector.get_comprehensive_institutional_data()
            
            # Get market data
            hist = self.stock.history(period="2y")
            info = self.stock.info
            
            return {
                'enhanced_data': enhanced_data,
                'institutional_data': institutional_data,
                'hist': hist,
                'info': info
            }
        except Exception as e:
            print(f"❌ Error collecting prediction data: {e}")
            return None
    
    def create_price_prediction_features(self, data):
        """Create comprehensive features for price prediction"""
        try:
            hist = data.get('hist')
            info = data.get('info', {})
            enhanced_data = data.get('enhanced_data', {})
            institutional_data = data.get('institutional_data', {})
            
            if hist is None or hist.empty:
                print("❌ No historical data available")
                return {}
            
            features = {}
            
            # 1. TECHNICAL INDICATORS
            current_price = hist['Close'].iloc[-1]
            
            # Price momentum
            features['price_momentum_5d'] = (current_price / hist['Close'].iloc[-6] - 1) * 100 if len(hist) > 5 else 0
            features['price_momentum_20d'] = (current_price / hist['Close'].iloc[-21] - 1) * 100 if len(hist) > 21 else 0
            features['price_momentum_50d'] = (current_price / hist['Close'].iloc[-51] - 1) * 100 if len(hist) > 51 else 0
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(hist['Close'])
            features['macd'] = self._calculate_macd(hist['Close'])
            features['bb_position'] = self._calculate_bb_position(hist['Close'])
            features['volume_ratio'] = hist['Volume'].iloc[-1] / hist['Volume'].rolling(20).mean().iloc[-1]
            features['volatility_20d'] = hist['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Moving averages
            features['price_vs_sma20'] = (current_price / hist['Close'].rolling(20).mean().iloc[-1] - 1) * 100
            features['price_vs_sma50'] = (current_price / hist['Close'].rolling(50).mean().iloc[-1] - 1) * 100
            features['sma20_vs_sma50'] = (hist['Close'].rolling(20).mean().iloc[-1] / hist['Close'].rolling(50).mean().iloc[-1] - 1) * 100
            
            # 2. FUNDAMENTAL METRICS
            features['pe_ratio'] = info.get('trailingPE', 0)
            features['forward_pe'] = info.get('forwardPE', 0)
            features['peg_ratio'] = info.get('pegRatio', 0)
            features['price_to_book'] = info.get('priceToBook', 0)
            features['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
            features['debt_to_equity'] = info.get('debtToEquity', 0)
            features['return_on_equity'] = info.get('returnOnEquity', 0)
            features['profit_margin'] = info.get('profitMargins', 0)
            features['revenue_growth'] = info.get('revenueGrowth', 0)
            features['earnings_growth'] = info.get('earningsGrowth', 0)
            features['beta'] = info.get('beta', 1.0)
            
            # Market cap and enterprise value
            features['market_cap'] = info.get('marketCap', 0) / 1e9  # Convert to billions
            features['enterprise_value'] = info.get('enterpriseValue', 0) / 1e9
            
            # 3. ANALYST SENTIMENT & TARGETS
            analyst_data = enhanced_data.get('analyst_data', {}) if enhanced_data else {}
            features['analyst_rating_mean'] = analyst_data.get('rating_mean', 0) if analyst_data else 0
            features['analyst_upside_potential'] = analyst_data.get('upside_potential', 0) if analyst_data else 0
            features['analyst_count'] = analyst_data.get('analyst_count', 0) if analyst_data else 0
            
            # Yahoo Finance analyst data
            features['yahoo_target_mean'] = info.get('targetMeanPrice', 0)
            features['yahoo_recommendation_mean'] = info.get('recommendationMean', 0)
            features['yahoo_analyst_opinions'] = info.get('numberOfAnalystOpinions', 0)
            
            # 4. NEWS & SENTIMENT ANALYSIS
            news_sentiment = enhanced_data.get('news_sentiment', {}) if enhanced_data else {}
            features['news_sentiment_score'] = news_sentiment.get('overall_sentiment', 0) if news_sentiment else 0
            features['news_confidence'] = news_sentiment.get('confidence', 0) if news_sentiment else 0
            features['news_count'] = news_sentiment.get('news_count', 0) if news_sentiment else 0
            features['news_positive_ratio'] = news_sentiment.get('positive_ratio', 0.5) if news_sentiment else 0.5
            
            # Social sentiment
            social_sentiment = enhanced_data.get('social_sentiment', {}) if enhanced_data else {}
            features['social_sentiment_score'] = social_sentiment.get('sentiment', 0) if social_sentiment else 0
            features['social_mentions'] = social_sentiment.get('mentions', 0) if social_sentiment else 0
            features['social_engagement'] = social_sentiment.get('engagement_rate', 0) if social_sentiment else 0
            
            # 5. STOCK RANKINGS & PERFORMANCE
            stock_rankings = enhanced_data.get('stock_rankings', {}) if enhanced_data else {}
            
            # Finviz rankings
            finviz_data = stock_rankings.get('finviz_rankings', {}) if stock_rankings else {}
            features['finviz_pe_rank'] = self._safe_float(finviz_data.get('pe_ratio', 0)) if finviz_data else 0
            features['finviz_peg_rank'] = self._safe_float(finviz_data.get('peg_ratio', 0)) if finviz_data else 0
            features['finviz_price_to_sales_rank'] = self._safe_float(finviz_data.get('price_to_sales', 0)) if finviz_data else 0
            features['finviz_debt_to_equity_rank'] = self._safe_float(finviz_data.get('debt_to_equity', 0)) if finviz_data else 0
            features['finviz_roe_rank'] = self._safe_float(finviz_data.get('return_on_equity', 0)) if finviz_data else 0
            
            # MarketWatch rankings
            mw_data = stock_rankings.get('marketwatch_rankings', {}) if stock_rankings else {}
            features['mw_average_rating'] = mw_data.get('average_rating', 0) if mw_data else 0
            features['mw_rating_count'] = mw_data.get('rating_count', 0) if mw_data else 0
            
            # Sector performance
            sector_data = stock_rankings.get('sector_performance', {}) if stock_rankings else {}
            features['sector_outperformance'] = sector_data.get('outperformance', 0) if sector_data else 0
            features['sector_rank_above'] = 1 if sector_data.get('sector_rank') == 'Above' else 0
            
            # 6. INSTITUTIONAL DATA
            institutional_holders = institutional_data.get('institutional_holders', {}) if institutional_data else {}
            features['institutional_ownership'] = institutional_holders.get('ownership_percent', 0) if institutional_holders else 0
            features['institution_count'] = institutional_holders.get('institution_count', 0) if institutional_holders else 0
            
            # Insider trading sentiment
            insider_data = institutional_data.get('insider_transactions', {}) if institutional_data else {}
            features['insider_buy_ratio'] = insider_data.get('buy_ratio', 0.5) if insider_data else 0.5
            features['insider_net_shares'] = insider_data.get('net_shares_change', 0) if insider_data else 0
            
            # 7. EARNINGS DATA
            earnings_data = institutional_data.get('earnings_data', {}) if institutional_data else {}
            features['earnings_surprise_avg'] = earnings_data.get('avg_surprise_percent', 0) if earnings_data else 0
            features['earnings_beat_ratio'] = earnings_data.get('beat_ratio', 0.5) if earnings_data else 0.5
            
            # Revenue growth
            revenue_growth = earnings_data.get('revenue_growth', {}) if earnings_data else {}
            features['yoy_revenue_growth'] = revenue_growth.get('yoy_growth', 0) if revenue_growth else 0
            features['ytd_revenue_growth'] = revenue_growth.get('ytd_growth', 0) if revenue_growth else 0
            
            # 8. ECONOMIC & MARKET CONDITIONS
            economic_data = enhanced_data.get('economic_data', {}) if enhanced_data else {}
            features['vix'] = economic_data.get('vix', 20) if economic_data else 20
            features['treasury_yield'] = economic_data.get('treasury_yield', 3.0) if economic_data else 3.0
            features['dollar_index'] = economic_data.get('dollar_index', 100) if economic_data else 100
            
            # 9. OPTIONS DATA
            options_data = enhanced_data.get('options_data', {}) if enhanced_data else {}
            features['put_call_ratio'] = options_data.get('put_call_ratio', 1.0) if options_data else 1.0
            features['options_volume'] = options_data.get('options_volume', 0) if options_data else 0
            features['implied_volatility'] = options_data.get('implied_volatility', 0.2) if options_data else 0.2
            
            # 10. RELATIVE PERFORMANCE
            features['relative_strength_vs_spy'] = self._calculate_relative_strength(hist)
            features['relative_strength_vs_sector'] = self._calculate_sector_relative_strength(hist, info.get('sector', ''))
            
            return features
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            return {}
    
    def predict_price_targets(self, features, current_price, prediction_horizons=[7, 30, 90, 365]):
        """Predict price targets for different time horizons"""
        try:
            predictions = {}
            
            # Create feature DataFrame
            feature_df = pd.DataFrame([features])
            
            # Remove any infinite or NaN values
            feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # For each prediction horizon
            for horizon_days in prediction_horizons:
                # Create training data with historical price changes
                hist = self.stock.history(period="2y")
                if len(hist) < horizon_days + 30:
                    continue
                
                # Create target variable (future price change)
                hist['future_price'] = hist['Close'].shift(-horizon_days)
                hist['price_change_pct'] = (hist['future_price'] / hist['Close'] - 1) * 100
                
                # Create features for historical data
                hist_features = []
                hist_targets = []
                
                for i in range(30, len(hist) - horizon_days):
                    # Get features for this point in time
                    hist_slice = hist.iloc[:i+1]
                    hist_info = self.stock.info  # Use current info as proxy
                    
                    # Create features (simplified version)
                    hist_feature = {
                        'rsi': self._calculate_rsi(hist_slice['Close']),
                        'macd': self._calculate_macd(hist_slice['Close']),
                        'price_momentum_5d': (hist_slice['Close'].iloc[-1] / hist_slice['Close'].iloc[-6] - 1) * 100 if len(hist_slice) > 5 else 0,
                        'price_momentum_20d': (hist_slice['Close'].iloc[-1] / hist_slice['Close'].iloc[-21] - 1) * 100 if len(hist_slice) > 21 else 0,
                        'volume_ratio': hist_slice['Volume'].iloc[-1] / hist_slice['Volume'].rolling(20).mean().iloc[-1] if len(hist_slice) > 20 else 1,
                        'volatility_20d': hist_slice['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(hist_slice) > 20 else 0,
                        'pe_ratio': features.get('pe_ratio', 0),
                        'beta': features.get('beta', 1.0),
                        'news_sentiment_score': features.get('news_sentiment_score', 0),
                        'analyst_rating_mean': features.get('analyst_rating_mean', 0),
                        'institutional_ownership': features.get('institutional_ownership', 0)
                    }
                    
                    hist_features.append(list(hist_feature.values()))
                    hist_targets.append(hist.iloc[i]['price_change_pct'])
                
                if len(hist_features) < 10:
                    continue
                
                # Train model
                X = np.array(hist_features)
                y = np.array(hist_targets)
                
                # Remove NaN values
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                
                if len(X) < 10:
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train ensemble model
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
                gb_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                
                rf_model.fit(X_train, y_train)
                gb_model.fit(X_train, y_train)
                
                # Make prediction
                current_features = np.array([list(feature_df.iloc[0][list(hist_feature.keys())])])
                
                rf_pred = rf_model.predict(current_features)[0]
                gb_pred = gb_model.predict(current_features)[0]
                
                # Ensemble prediction
                ensemble_pred = (rf_pred * 0.6 + gb_pred * 0.4)
                
                # Calculate price target
                price_target = current_price * (1 + ensemble_pred / 100)
                
                # Calculate confidence based on model performance
                rf_score = rf_model.score(X_test, y_test)
                gb_score = gb_model.score(X_test, y_test)
                confidence = (rf_score * 0.6 + gb_score * 0.4)
                
                predictions[f'{horizon_days}d'] = {
                    'price_target': price_target,
                    'price_change_pct': ensemble_pred,
                    'confidence': max(0, min(1, confidence)),
                    'model_performance': {
                        'rf_score': rf_score,
                        'gb_score': gb_score,
                        'ensemble_score': confidence
                    }
                }
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error in price prediction: {e}")
            return {}
    
    def generate_price_scenarios(self, features, current_price):
        """Generate bullish, bearish, and neutral price scenarios"""
        try:
            scenarios = {}
            
            # Calculate scenario probabilities based on features
            bullish_score = 0
            bearish_score = 0
            
            # Technical factors
            if features.get('rsi', 50) < 70:
                bullish_score += 1
            else:
                bearish_score += 1
                
            if features.get('macd', 0) > 0:
                bullish_score += 1
            else:
                bearish_score += 1
                
            if features.get('price_momentum_20d', 0) > 0:
                bullish_score += 1
            else:
                bearish_score += 1
            
            # Fundamental factors
            if features.get('pe_ratio', 0) < 25 and features.get('pe_ratio', 0) > 0:
                bullish_score += 1
            else:
                bearish_score += 1
                
            if features.get('revenue_growth', 0) > 0:
                bullish_score += 1
            else:
                bearish_score += 1
            
            # Sentiment factors
            if features.get('news_sentiment_score', 0) > 0.1:
                bullish_score += 1
            else:
                bearish_score += 1
                
            if features.get('analyst_rating_mean', 0) > 2.5:
                bullish_score += 1
            else:
                bearish_score += 1
            
            # Calculate probabilities
            total_score = bullish_score + bearish_score
            bullish_prob = bullish_score / total_score if total_score > 0 else 0.5
            bearish_prob = bearish_score / total_score if total_score > 0 else 0.5
            neutral_prob = 1 - abs(bullish_prob - bearish_prob)
            
            # Generate price targets for each scenario
            scenarios['bullish'] = {
                'probability': bullish_prob,
                'price_target': current_price * 1.15,  # +15% upside
                'price_change_pct': 15,
                'description': 'Strong positive momentum with favorable fundamentals'
            }
            
            scenarios['neutral'] = {
                'probability': neutral_prob,
                'price_target': current_price * 1.02,  # +2% modest upside
                'price_change_pct': 2,
                'description': 'Mixed signals with moderate expectations'
            }
            
            scenarios['bearish'] = {
                'probability': bearish_prob,
                'price_target': current_price * 0.88,  # -12% downside
                'price_change_pct': -12,
                'description': 'Negative momentum with concerning fundamentals'
            }
            
            return scenarios
            
        except Exception as e:
            print(f"❌ Error generating scenarios: {e}")
            return {}
    
    def get_analyst_price_targets(self, data):
        """Extract analyst price targets from multiple sources"""
        try:
            targets = {}
            
            # Yahoo Finance targets
            info = data['info']
            yahoo_target = info.get('targetMeanPrice', 0)
            if yahoo_target > 0:
                targets['yahoo_mean'] = yahoo_target
                targets['yahoo_high'] = info.get('targetHighPrice', yahoo_target * 1.1)
                targets['yahoo_low'] = info.get('targetLowPrice', yahoo_target * 0.9)
            
            # Enhanced analyst data
            enhanced_data = data['enhanced_data']
            analyst_data = enhanced_data.get('analyst_data', {})
            
            if analyst_data.get('target_price_mean', 0) > 0:
                targets['enhanced_mean'] = analyst_data['target_price_mean']
                targets['enhanced_high'] = analyst_data.get('target_price_high', targets['enhanced_mean'] * 1.1)
                targets['enhanced_low'] = analyst_data.get('target_price_low', targets['enhanced_mean'] * 0.9)
            
            # Calculate consensus
            all_targets = [v for v in targets.values() if v > 0]
            if all_targets:
                targets['consensus_mean'] = np.mean(all_targets)
                targets['consensus_high'] = np.max(all_targets)
                targets['consensus_low'] = np.min(all_targets)
                targets['target_count'] = len(all_targets)
            
            return targets
            
        except Exception as e:
            print(f"❌ Error getting analyst targets: {e}")
            return {}
    
    def generate_comprehensive_prediction(self):
        """Generate comprehensive price prediction with all factors"""
        try:
            print(f"🔮 Generating comprehensive price prediction for {self.ticker}...")
            
            # Get all data
            data = self.get_comprehensive_prediction_data()
            if not data:
                return None
            
            current_price = data['hist']['Close'].iloc[-1]
            
            # Create features
            features = self.create_price_prediction_features(data)
            
            # Generate predictions
            predictions = self.predict_price_targets(features, current_price)
            scenarios = self.generate_price_scenarios(features, current_price)
            analyst_targets = self.get_analyst_price_targets(data)
            
            # Calculate overall recommendation
            overall_score = self._calculate_overall_score(features)
            
            return {
                'ticker': self.ticker,
                'current_price': current_price,
                'prediction_horizons': predictions,
                'scenarios': scenarios,
                'analyst_targets': analyst_targets,
                'overall_score': overall_score,
                'features_used': len(features),
                'prediction_date': datetime.now().isoformat(),
                'data_sources': list(data['enhanced_data'].keys()) if data['enhanced_data'] else ['basic']
            }
            
        except Exception as e:
            print(f"❌ Error in comprehensive prediction: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return float(100 - (100 / (1 + rs)).iloc[-1])
        except:
            return 50
    
    def _calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            return float((ema_fast - ema_slow).iloc[-1])
        except:
            return 0
    
    def _calculate_bb_position(self, prices, period=20):
        """Calculate Bollinger Band position"""
        try:
            bb_middle = prices.rolling(period).mean()
            bb_std = prices.rolling(period).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            return float(((prices.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) * 100)
        except:
            return 50
    
    def _calculate_relative_strength(self, hist):
        """Calculate relative strength vs SPY"""
        try:
            spy = yf.Ticker("SPY").history(period="1y")
            if len(spy) > 0 and len(hist) > 0:
                stock_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1) * 100 if len(hist) > 252 else 0
                spy_return = (spy['Close'].iloc[-1] / spy['Close'].iloc[-252] - 1) * 100 if len(spy) > 252 else 0
                return stock_return - spy_return
            return 0
        except:
            return 0
    
    def _calculate_sector_relative_strength(self, hist, sector):
        """Calculate relative strength vs sector"""
        try:
            # This would require sector ETF data - simplified for now
            return 0
        except:
            return 0
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            if value == 'N/A' or value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_overall_score(self, features):
        """Calculate overall bullish/bearish score"""
        try:
            score = 0
            factors = []
            
            # Technical factors (0-3 points)
            if features.get('rsi', 50) < 70:
                score += 1
                factors.append("RSI not overbought")
            if features.get('macd', 0) > 0:
                score += 1
                factors.append("MACD bullish")
            if features.get('price_momentum_20d', 0) > 0:
                score += 1
                factors.append("Positive momentum")
            
            # Fundamental factors (0-3 points)
            if features.get('pe_ratio', 0) < 25 and features.get('pe_ratio', 0) > 0:
                score += 1
                factors.append("Reasonable valuation")
            if features.get('revenue_growth', 0) > 0:
                score += 1
                factors.append("Revenue growth")
            if features.get('return_on_equity', 0) > 0.15:
                score += 1
                factors.append("Strong ROE")
            
            # Sentiment factors (0-2 points)
            if features.get('news_sentiment_score', 0) > 0.1:
                score += 1
                factors.append("Positive sentiment")
            if features.get('analyst_rating_mean', 0) > 2.5:
                score += 1
                factors.append("Analyst support")
            
            # Institutional factors (0-2 points)
            if features.get('institutional_ownership', 0) > 0.5:
                score += 1
                factors.append("Institutional support")
            if features.get('insider_buy_ratio', 0.5) > 0.6:
                score += 1
                factors.append("Insider buying")
            
            return {
                'score': score,
                'max_score': 10,
                'percentage': (score / 10) * 100,
                'factors': factors,
                'recommendation': 'STRONG BUY' if score >= 8 else 'BUY' if score >= 6 else 'HOLD' if score >= 4 else 'SELL'
            }
            
        except Exception as e:
            print(f"❌ Error calculating overall score: {e}")
            return {'score': 5, 'max_score': 10, 'percentage': 50, 'factors': [], 'recommendation': 'HOLD'}

def main():
    """Test the enhanced price predictor"""
    ticker = "AAPL"
    predictor = EnhancedPricePredictor(ticker)
    
    result = predictor.generate_comprehensive_prediction()
    
    if result:
        print(f"\n🏆 COMPREHENSIVE PRICE PREDICTION: {ticker}")
        print("=" * 60)
        
        print(f"\n📊 Current Price: ${result['current_price']:.2f}")
        print(f"📅 Prediction Date: {result['prediction_date']}")
        print(f"🔍 Features Used: {result['features_used']}")
        print(f"📈 Data Sources: {', '.join(result['data_sources'])}")
        
        # Prediction horizons
        print(f"\n🎯 PRICE PREDICTIONS BY TIMELINE")
        for horizon, pred in result['prediction_horizons'].items():
            print(f"{horizon}: ${pred['price_target']:.2f} ({pred['price_change_pct']:+.1f}%) - Confidence: {pred['confidence']:.1%}")
        
        # Scenarios
        print(f"\n📊 PRICE SCENARIOS")
        for scenario, data in result['scenarios'].items():
            print(f"{scenario.upper()}: ${data['price_target']:.2f} ({data['price_change_pct']:+.1f}%) - Probability: {data['probability']:.1%}")
            print(f"  Description: {data['description']}")
        
        # Analyst targets
        if result['analyst_targets']:
            print(f"\n👔 ANALYST PRICE TARGETS")
            for source, target in result['analyst_targets'].items():
                if isinstance(target, (int, float)) and target > 0:
                    print(f"{source}: ${target:.2f}")
        
        # Overall score
        score = result['overall_score']
        print(f"\n🎯 OVERALL RECOMMENDATION")
        print(f"Score: {score['score']}/{score['max_score']} ({score['percentage']:.1f}%)")
        print(f"Recommendation: {score['recommendation']}")
        print(f"Key Factors: {', '.join(score['factors'][:5])}")
        
    else:
        print(f"❌ Failed to generate prediction for {ticker}")

if __name__ == "__main__":
    main()
