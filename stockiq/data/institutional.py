#!/usr/bin/env python3
"""
Enhanced Institutional and Hedge Fund Data Collection
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json

class EnhancedInstitutionalData:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
    
    def get_top_institutional_holders(self):
        """Get top institutional holders from Yahoo Finance"""
        try:
            # Get institutional holders
            institutional_holders = self.stock.institutional_holders
            major_holders = self.stock.major_holders
            
            if institutional_holders is not None and not institutional_holders.empty:
                top_holders = institutional_holders.head(10)
                
                holders_data = []
                for _, holder in top_holders.iterrows():
                    holders_data.append({
                        'name': holder['Holder'],
                        'shares': holder['Shares'],
                        'date_reported': holder['Date Reported'],
                        'percent_out': holder['pctHeld'],
                        'value': holder['Value'],
                        'pct_change': holder['pctChange']
                    })
                
                return {
                    'top_holders': holders_data,
                    'total_institutions': len(institutional_holders),
                    'data_freshness': 'Latest 13F filings'
                }
            else:
                return {'top_holders': [], 'total_institutions': 0, 'data_freshness': 'No data available'}
                
        except Exception as e:
            print(f"❌ Error fetching institutional holders: {e}")
            return {'top_holders': [], 'total_institutions': 0, 'data_freshness': 'Error'}
    
    def get_insider_transactions(self):
        """Get recent insider transactions"""
        try:
            insider_transactions = self.stock.insider_transactions
            
            if insider_transactions is not None and not insider_transactions.empty:
                recent_transactions = insider_transactions.head(10)
                
                transactions_data = []
                for _, transaction in recent_transactions.iterrows():
                    transactions_data.append({
                        'insider': transaction['Insider'],
                        'position': transaction['Position'],
                        'transaction_date': transaction['Start Date'],
                        'transaction_type': transaction['Transaction'],
                        'shares': transaction['Shares'],
                        'value': transaction['Value'],
                        'ownership': transaction['Ownership']
                    })
                
                return {
                    'recent_transactions': transactions_data,
                    'total_transactions': len(insider_transactions),
                    'data_freshness': 'Latest insider filings'
                }
            else:
                return {'recent_transactions': [], 'total_transactions': 0, 'data_freshness': 'No data available'}
                
        except Exception as e:
            print(f"❌ Error fetching insider transactions: {e}")
            return {'recent_transactions': [], 'total_transactions': 0, 'data_freshness': 'Error'}
    
    def get_earnings_history(self):
        """Get earnings history and estimates with revenue growth"""
        try:
            # Get earnings history
            earnings_history = self.stock.earnings_history
            
            # Get earnings estimates
            earnings_estimates = self.stock.earnings_estimate
            
            # Get financials for revenue growth calculation
            financials = self.stock.financials
            
            earnings_data = {
                'history': [],
                'estimates': [],
                'revenue_growth': {},
                'data_freshness': 'Latest earnings data'
            }
            
            if earnings_history is not None and not earnings_history.empty:
                # Sort by report date descending so the UI shows the latest
                # quarter first rather than the oldest one in the window.
                earnings_history = earnings_history.sort_index(ascending=False)
                for date, row in earnings_history.head(5).iterrows():
                    # Convert date to proper quarter format
                    year = date.year
                    month = date.month
                    if month in [1, 2, 3]:
                        quarter = f"{year} Q1"
                    elif month in [4, 5, 6]:
                        quarter = f"{year} Q2"
                    elif month in [7, 8, 9]:
                        quarter = f"{year} Q3"
                    else:  # month in [10, 11, 12]
                        quarter = f"{year} Q4"
                    
                    earnings_data['history'].append({
                        'quarter': quarter,
                        'actual_eps': row.get('epsActual', 'N/A'),
                        'estimate_eps': row.get('epsEstimate', 'N/A'),
                        'surprise': row.get('epsDifference', 'N/A'),
                        'surprise_percent': row.get('surprisePercent', 'N/A')
                    })
            
            if earnings_estimates is not None and not earnings_estimates.empty:
                for period, row in earnings_estimates.head(4).iterrows():
                    # Convert period to readable format
                    if period == '0q':
                        quarter = "Current Q"
                    elif period == '+1q':
                        quarter = "Next Q"
                    elif period == '0y':
                        quarter = "Current Year"
                    elif period == '+1y':
                        quarter = "Next Year"
                    else:
                        quarter = str(period)
                    
                    earnings_data['estimates'].append({
                        'quarter': quarter,
                        'estimate_eps': row.get('avg', 'N/A'),
                        'low_estimate': row.get('low', 'N/A'),
                        'high_estimate': row.get('high', 'N/A'),
                        'year_ago_eps': row.get('yearAgoEps', 'N/A'),
                        'analysts': row.get('numberOfAnalysts', 'N/A'),
                        'growth': row.get('growth', 'N/A')
                    })
            
            # Calculate revenue growth metrics
            if financials is not None and not financials.empty:
                try:
                    # Get revenue data (Total Revenue)
                    revenue_data = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else None
                    
                    if revenue_data is not None and len(revenue_data) >= 2:
                        # Year-over-Year revenue growth
                        current_year_revenue = revenue_data.iloc[0]  # Most recent year
                        previous_year_revenue = revenue_data.iloc[1]  # Previous year
                        
                        if previous_year_revenue != 0:
                            yoy_revenue_growth = ((current_year_revenue - previous_year_revenue) / previous_year_revenue) * 100
                        else:
                            yoy_revenue_growth = 0
                        
                        # Calculate YTD growth (if we have quarterly data)
                        quarterly_financials = self.stock.quarterly_financials
                        if quarterly_financials is not None and not quarterly_financials.empty:
                            try:
                                quarterly_revenue = quarterly_financials.loc['Total Revenue']
                                if len(quarterly_revenue) >= 4:
                                    # Sum last 4 quarters for current year
                                    current_ytd_revenue = quarterly_revenue.iloc[:4].sum()
                                    # Sum quarters 4-7 for previous year
                                    previous_ytd_revenue = quarterly_revenue.iloc[4:8].sum() if len(quarterly_revenue) >= 8 else previous_year_revenue
                                    
                                    if previous_ytd_revenue != 0:
                                        ytd_revenue_growth = ((current_ytd_revenue - previous_ytd_revenue) / previous_ytd_revenue) * 100
                                    else:
                                        ytd_revenue_growth = 0
                                else:
                                    ytd_revenue_growth = yoy_revenue_growth  # Fallback to YoY
                            except:
                                ytd_revenue_growth = yoy_revenue_growth  # Fallback to YoY
                        else:
                            ytd_revenue_growth = yoy_revenue_growth  # Fallback to YoY
                        
                        earnings_data['revenue_growth'] = {
                            'yoy_growth': round(yoy_revenue_growth, 2),
                            'ytd_growth': round(ytd_revenue_growth, 2),
                            'current_year_revenue': current_year_revenue,
                            'previous_year_revenue': previous_year_revenue,
                            'growth_trend': 'positive' if yoy_revenue_growth > 0 else 'negative' if yoy_revenue_growth < 0 else 'stable'
                        }
                    else:
                        earnings_data['revenue_growth'] = {
                            'yoy_growth': 'N/A',
                            'ytd_growth': 'N/A',
                            'current_year_revenue': 'N/A',
                            'previous_year_revenue': 'N/A',
                            'growth_trend': 'N/A'
                        }
                except Exception as e:
                    print(f"⚠️ Revenue growth calculation error: {e}")
                    earnings_data['revenue_growth'] = {
                        'yoy_growth': 'N/A',
                        'ytd_growth': 'N/A',
                        'current_year_revenue': 'N/A',
                        'previous_year_revenue': 'N/A',
                        'growth_trend': 'N/A'
                    }
            else:
                earnings_data['revenue_growth'] = {
                    'yoy_growth': 'N/A',
                    'ytd_growth': 'N/A',
                    'current_year_revenue': 'N/A',
                    'previous_year_revenue': 'N/A',
                    'growth_trend': 'N/A'
                }
            
            return earnings_data
            
        except Exception as e:
            print(f"❌ Error fetching earnings data: {e}")
            return {'history': [], 'estimates': [], 'data_freshness': 'Error'}
    
    def get_analyst_recommendations_detailed(self):
        """Get detailed analyst recommendations"""
        try:
            recommendations = self.stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                recent_recommendations = recommendations.head(10)
                
                rec_data = []
                for _, rec in recent_recommendations.iterrows():
                    rec_data.append({
                        'firm': rec.get('Firm', 'N/A'),
                        'to_grade': rec.get('To Grade', 'N/A'),
                        'from_grade': rec.get('From Grade', 'N/A'),
                        'action': rec.get('Action', 'N/A'),
                        'date': rec.get('Date', 'N/A')
                    })
                
                return {
                    'recent_recommendations': rec_data,
                    'total_recommendations': len(recommendations),
                    'data_freshness': 'Latest analyst reports'
                }
            else:
                return {'recent_recommendations': [], 'total_recommendations': 0, 'data_freshness': 'No data available'}
                
        except Exception as e:
            print(f"❌ Error fetching detailed recommendations: {e}")
            return {'recent_recommendations': [], 'total_recommendations': 0, 'data_freshness': 'Error'}
    
    def get_comprehensive_institutional_data(self):
        """Get all institutional and hedge fund data"""
        print(f"🔍 Collecting comprehensive institutional data for {self.ticker}...")
        
        data = {
            'ticker': self.ticker,
            'timestamp': datetime.now().isoformat(),
            'institutional_holders': self.get_top_institutional_holders(),
            'insider_transactions': self.get_insider_transactions(),
            'earnings_data': self.get_earnings_history(),
            'analyst_recommendations': self.get_analyst_recommendations_detailed()
        }
        
        return data

def main():
    """Test the enhanced institutional data collection"""
    ticker = "AAPL"  # Test with Apple
    collector = EnhancedInstitutionalData(ticker)
    
    print(f"🏆 Enhanced Institutional Data Collection for {ticker}")
    print("=" * 60)
    
    data = collector.get_comprehensive_institutional_data()
    
    # Display results
    print(f"\n📊 Top Institutional Holders:")
    holders = data['institutional_holders']['top_holders']
    for i, holder in enumerate(holders[:5], 1):
        print(f"  {i}. {holder['name']}: {holder['percent_out']:.2f}% ({holder['shares']:,} shares)")
    
    print(f"\n👥 Recent Insider Transactions:")
    transactions = data['insider_transactions']['recent_transactions']
    for i, transaction in enumerate(transactions[:3], 1):
        print(f"  {i}. {transaction['insider']}: {transaction['transaction_type']} {transaction['shares']:,} shares")
    
    print(f"\n📈 Recent Earnings History:")
    earnings = data['earnings_data']['history']
    for i, earning in enumerate(earnings[:3], 1):
        print(f"  {i}. {earning['quarter']}: Actual {earning['actual_eps']} vs Est. {earning['estimate_eps']}")
    
    print(f"\n🎯 Recent Analyst Recommendations:")
    recommendations = data['analyst_recommendations']['recent_recommendations']
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec['firm']}: {rec['action']} to {rec['to_grade']}")

if __name__ == "__main__":
    main()
