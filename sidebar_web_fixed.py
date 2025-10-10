#!/usr/bin/env python3
"""
Sidebar layout web interface with detailed analysis on the right
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="🏆 Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 0.8rem;
        margin-bottom: 0.3rem;
        padding: 0.4rem;
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        border-radius: 4px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 0.6rem;
        border-radius: 6px;
        border-left: 3px solid #3498db;
        margin: 0.3rem 0;
    }
    .compact-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 6px;
        margin: 4px 0;
    }
    .compact-metric {
        background: #f8f9fa;
        padding: 4px 6px;
        border-radius: 3px;
        border-left: 2px solid #3498db;
        font-size: 11px;
    }
    .inline-metric {
        display: inline-block;
        margin-right: 12px;
        font-size: 11px;
    }
    .compact-text {
        font-size: 11px;
        margin: 1px 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .risk-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<div class="main-header">🏆 Professional Stock Analyzer</div>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    
    # Main analysis controls
    st.markdown("### 📊 Stock Analysis")
    ticker = st.text_input("Stock Ticker", value="AAPL", placeholder="AAPL", help="Enter stock symbol")
    
    # Analysis options
    st.markdown("### ⚙️ Analysis Options")
    show_charts = st.checkbox("📈 Show Charts", value=True, help="Display interactive price charts")
    show_ml = st.checkbox("🤖 ML Predictions", value=True, help="Include machine learning predictions")
    show_fundamentals = st.checkbox("💰 Fundamentals", value=True, help="Show fundamental analysis")
    show_sentiment = st.checkbox("😊 Sentiment", value=True, help="Include sentiment analysis")
    show_backtest = st.checkbox("📊 Backtest", value=True, help="Show backtesting results")
    
    # Analyze button
    analyze_clicked = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Quick Analysis Options
    st.markdown("### ⚡ Quick Analysis")
    quick_mode = st.checkbox("🚀 Fast Mode", value=False, help="Skip ML and backtest for faster analysis")
    
    if quick_mode:
        st.info("Fast mode will skip ML predictions and backtesting for quicker results")

# Main content area
if analyze_clicked and ticker:
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            from stock_analyzer import UniversalStockAnalyzer
            analyzer = UniversalStockAnalyzer(ticker)
            
            # Use quick mode settings if enabled
            if quick_mode:
                data = analyzer.analyze(show_charts, False, show_fundamentals, show_sentiment)  # Skip ML
                if data and 'backtest_results' in data:
                    data['backtest_results'] = None  # Skip backtest
            else:
                data = analyzer.analyze(show_charts, show_ml, show_fundamentals, show_sentiment)
            
            if data:
                st.success(f"✅ {ticker} Analysis Complete")
                
                tech_data = data['tech_data']
                fundamental_data = data['fundamental_data']
                sentiment_data = data['sentiment_data']
                earnings_data = data['earnings_data']
                ml_prediction = data['ml_prediction']
                backtest_results = data['backtest_results']
                info = data['info']
                
                # DETAILED ANALYSIS LAYOUT
                
                # Combined Company Overview, Key Metrics & Price Chart
                with st.expander("📊 Company Overview & Price Chart", expanded=True):
                    # Create two main columns: left for company info, right for chart
                    left_col, right_col = st.columns([1, 1.5])
                    
                    with left_col:
                        st.markdown("#### 🏢 Company Information")
                        st.markdown(f"**Company:** {info.get('longName', 'N/A')}")
                        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                        st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
                        st.markdown(f"**Website:** {info.get('website', 'N/A')}")
                        st.markdown(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Employees:** N/A")
                        
                        # Earnings Calendar (if available)
                        if earnings_data['earnings_expected']:
                            st.markdown("#### 📅 Earnings Calendar")
                            st.markdown(f"**Next Earnings:** {earnings_data['next_earnings_date'].strftime('%Y-%m-%d')}")
                            st.markdown(f"**Days to Earnings:** {earnings_data['days_to_earnings']}")
                            if earnings_data['volatility_expected']:
                                st.markdown('<div class="risk-card">⚠️ **High Volatility Expected**</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success-card">✅ **Normal Volatility Expected**</div>', unsafe_allow_html=True)
                    
                    with right_col:
                        st.markdown("### 📊 Price Chart")
                        # Charts
                        if show_charts:
                            try:
                                from stock_analyzer import create_charts
                                charts = create_charts(data)
                                if 'price' in charts:
                                    st.plotly_chart(charts['price'], use_container_width=True)
                            except Exception as e:
                                st.warning(f"Charts not available: {str(e)}")
                        else:
                            st.info("Enable 'Charts' option in sidebar to view price chart")
                    
                    # Key Metrics - Full width horizontal line under the chart
                    st.markdown("#### 📈 Key Metrics")
                    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
                    with col1:
                        st.metric("Current Price", f"${tech_data['current_price']:.2f}", f"{tech_data['price_change_pct']:+.1f}%")
                    with col2:
                        st.metric("Volume", f"{tech_data['volume']/1e6:.1f}M", f"{tech_data['volume_ratio']:.1f}x avg")
                    with col3:
                        st.metric("Market Cap", f"${fundamental_data['market_cap']/1e9:.1f}B" if fundamental_data['market_cap'] else "N/A")
                    with col4:
                        st.metric("P/E Ratio", f"{fundamental_data['pe_ratio']:.1f}" if fundamental_data['pe_ratio'] else "N/A")
                    with col5:
                        st.metric("Beta", f"{fundamental_data['beta']:.2f}" if fundamental_data['beta'] else "N/A")
                    with col6:
                        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
                    with col7:
                        st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else "N/A")
                    with col8:
                        st.metric("Avg Volume", f"{tech_data['volume_sma_20']/1e6:.1f}M")
                
                # Technical Analysis - Collapsible
                with st.expander("🔧 Technical Analysis", expanded=True):
                    # Compact metrics in a single row
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        rsi_status = "🔴" if tech_data['rsi'] > 70 else "🟢" if tech_data['rsi'] < 30 else "🟡"
                        st.metric("RSI", f"{tech_data['rsi']:.1f}", help=f"{rsi_status} {'Overbought' if tech_data['rsi'] > 70 else 'Oversold' if tech_data['rsi'] < 30 else 'Neutral'}")
                    with col2:
                        st.metric("MACD", f"{tech_data['macd']:.2f}")
                    with col3:
                        st.metric("SMA 20", f"${tech_data['sma_20']:.2f}")
                    with col4:
                        st.metric("SMA 50", f"${tech_data['sma_50']:.2f}")
                    with col5:
                        st.metric("Volatility", f"{tech_data['volatility_20d']:.1f}%")
                    with col6:
                        sma20_change = ((tech_data['current_price'] - tech_data['sma_20']) / tech_data['sma_20'] * 100)
                        st.metric("vs SMA20", f"{sma20_change:+.1f}%")
                
                # Performance Metrics - Collapsible
                with st.expander("📈 Performance Metrics", expanded=True):
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
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
                    with col6:
                        st.metric("Volume", f"{tech_data['volume']:,}")
                
                # Fundamental Analysis - Collapsible
                if show_fundamentals:
                    with st.expander("💰 Fundamental Analysis", expanded=False):
                        # Compact metrics in two rows
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("P/E", f"{fundamental_data['pe_ratio']:.1f}" if fundamental_data['pe_ratio'] else "N/A")
                        with col2:
                            st.metric("Forward P/E", f"{fundamental_data['forward_pe']:.1f}" if fundamental_data['forward_pe'] else "N/A")
                        with col3:
                            st.metric("PEG", f"{fundamental_data['peg_ratio']:.1f}" if fundamental_data['peg_ratio'] else "N/A")
                        with col4:
                            st.metric("P/B", f"{fundamental_data['price_to_book']:.1f}" if fundamental_data['price_to_book'] else "N/A")
                        with col5:
                            st.metric("P/S", f"{fundamental_data['price_to_sales']:.1f}" if fundamental_data['price_to_sales'] else "N/A")
                        with col6:
                            st.metric("Beta", f"{fundamental_data['beta']:.1f}" if fundamental_data['beta'] else "N/A")
                        
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("Current Ratio", f"{fundamental_data['current_ratio']:.1f}" if fundamental_data['current_ratio'] else "N/A")
                        with col2:
                            st.metric("D/E", f"{fundamental_data['debt_to_equity']:.1f}" if fundamental_data['debt_to_equity'] else "N/A")
                        with col3:
                            st.metric("ROE", f"{fundamental_data['return_on_equity']:.1%}" if fundamental_data['return_on_equity'] else "N/A")
                        with col4:
                            st.metric("Profit Margin", f"{fundamental_data['profit_margin']:.1%}" if fundamental_data['profit_margin'] else "N/A")
                        with col5:
                            st.metric("Revenue Growth", f"{fundamental_data['revenue_growth']:.1%}" if fundamental_data['revenue_growth'] else "N/A")
                        with col6:
                            st.metric("Market Cap", f"${fundamental_data['market_cap']/1e9:.1f}B" if fundamental_data['market_cap'] else "N/A")
                
                # Sentiment Analysis - Collapsible
                if show_sentiment:
                    with st.expander("😊 Advanced Sentiment Analysis", expanded=False):
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            sentiment_emoji = "😊" if sentiment_data['sentiment_label'] == 'POSITIVE' else "😞" if sentiment_data['sentiment_label'] == 'NEGATIVE' else "😐"
                            st.markdown(f"**Sentiment:** {sentiment_data['sentiment_label']} {sentiment_emoji}")
                        with col2:
                            st.metric("Score", f"{sentiment_data['overall_sentiment']:+.3f}")
                        with col3:
                            st.metric("Confidence", f"{sentiment_data['confidence']:.1%}")
                        with col4:
                            st.metric("News Sentiment", f"{sentiment_data['news_sentiment']:+.3f}")
                        with col5:
                            st.metric("Social Sentiment", f"{sentiment_data['social_sentiment']:+.3f}")
                        with col6:
                            st.metric("News Count", f"{sentiment_data['news_count']}")
                        
                        # Advanced sentiment breakdown if available
                        if 'advanced_sentiment' in sentiment_data:
                            advanced = sentiment_data['advanced_sentiment']
                            st.markdown("#### 🔬 Advanced Analysis Breakdown")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Overall Label", advanced.get('label', 'N/A'))
                            with col2:
                                st.metric("Overall Score", f"{advanced.get('score', 0):+.3f}")
                            with col3:
                                st.metric("Overall Confidence", f"{advanced.get('confidence', 0):.1%}")
                            with col4:
                                st.metric("Texts Analyzed", advanced.get('text_count', 0))
                            
                            # Score distribution
                            if 'score_distribution' in advanced:
                                dist = advanced['score_distribution']
                                st.markdown("#### 📊 Sentiment Distribution")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Positive", dist.get('positive', 0))
                                with col2:
                                    st.metric("Neutral", dist.get('neutral', 0))
                                with col3:
                                    st.metric("Negative", dist.get('negative', 0))
                            
                            # Individual results breakdown
                            if 'individual_results' in advanced and advanced['individual_results']:
                                st.markdown("#### 📰 Individual Article Analysis")
                                for i, result in enumerate(advanced['individual_results'][:5], 1):
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    with col1:
                                        st.markdown(f"**{i}.** {result.get('label', 'N/A')} ({result.get('score', 0):+.2f})")
                                    with col2:
                                        st.markdown(f"{result.get('confidence', 0):.0%}")
                                    with col3:
                                        st.markdown(f"{result.get('score', 0):+.2f}")
                        
                        # Sentiment explanation if available
                        if 'sentiment_explanation' in sentiment_data:
                            st.markdown("#### 💡 Sentiment Explanation")
                            st.info(sentiment_data['sentiment_explanation'])
                
                # Institutional Holdings - Collapsible
                institutional_data = data.get('institutional_data', {})
                if institutional_data:
                    with st.expander("🏛️ Institutional Holdings", expanded=False):
                        # Top Institutional Holders Table
                        holders = institutional_data.get('institutional_holders', {}).get('top_holders', [])
                        if holders:
                            st.markdown("#### 🏛️ Top Institutional Holders")
                            
                            # Create table data
                            table_data = []
                            for holder in holders[:8]:
                                change = holder.get('pct_change', 0)
                                change_str = f"{change:+.2%}" if change != 0 else "N/A"
                                change_indicator = "📈" if change > 0 else "📉" if change < 0 else ""
                                
                                table_data.append({
                                    'Institution': holder['name'],
                                    'Ownership': f"{holder['percent_out']:.2%}",
                                    'Shares': f"{holder['shares']:,}",
                                    'Change': f"{change_indicator}{change_str}"
                                })
                            
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                        
                        # Insider Trading Table
                        transactions = institutional_data.get('insider_transactions', {}).get('recent_transactions', [])
                        if transactions:
                            st.markdown("#### 👥 Recent Insider Activity")
                            
                            # Create table data
                            insider_data = []
                            for transaction in transactions[:6]:
                                insider_data.append({
                                    'Insider': transaction['insider'],
                                    'Position': transaction['position'] or 'N/A',
                                    'Transaction': transaction['transaction_type'] or 'N/A',
                                    'Shares': f"{transaction['shares']:,}" if transaction['shares'] else 'N/A',
                                    'Value': f"${transaction['value']:,.0f}" if transaction.get('value', 0) > 0 else 'N/A'
                                })
                            
                            st.dataframe(insider_data, use_container_width=True, hide_index=True)
                
                # Earnings History - Separate Section
                if institutional_data:
                    earnings_history = institutional_data.get('earnings_data', {}).get('history', [])
                    revenue_growth = institutional_data.get('earnings_data', {}).get('revenue_growth', {})
                    
                    if earnings_history or revenue_growth:
                        with st.expander("📊 Earnings History & Revenue Growth", expanded=False):
                            # Revenue Growth Section
                            if revenue_growth and revenue_growth.get('yoy_growth') != 'N/A':
                                st.markdown("#### 📈 Revenue Growth Analysis")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    yoy_growth = revenue_growth.get('yoy_growth', 'N/A')
                                    if yoy_growth != 'N/A':
                                        growth_color = "🟢" if yoy_growth > 0 else "🔴" if yoy_growth < 0 else "🟡"
                                        st.metric("YoY Revenue Growth", f"{growth_color}{yoy_growth:+.1f}%")
                                    else:
                                        st.metric("YoY Revenue Growth", "N/A")
                                
                                with col2:
                                    ytd_growth = revenue_growth.get('ytd_growth', 'N/A')
                                    if ytd_growth != 'N/A':
                                        growth_color = "🟢" if ytd_growth > 0 else "🔴" if ytd_growth < 0 else "🟡"
                                        st.metric("YTD Revenue Growth", f"{growth_color}{ytd_growth:+.1f}%")
                                    else:
                                        st.metric("YTD Revenue Growth", "N/A")
                                
                                with col3:
                                    current_revenue = revenue_growth.get('current_year_revenue', 'N/A')
                                    if current_revenue != 'N/A':
                                        st.metric("Current Year Revenue", f"${current_revenue/1e9:.1f}B")
                                    else:
                                        st.metric("Current Year Revenue", "N/A")
                                
                                with col4:
                                    growth_trend = revenue_growth.get('growth_trend', 'N/A')
                                    if growth_trend == 'positive':
                                        st.metric("Growth Trend", "🟢 Positive")
                                    elif growth_trend == 'negative':
                                        st.metric("Growth Trend", "🔴 Negative")
                                    elif growth_trend == 'stable':
                                        st.metric("Growth Trend", "🟡 Stable")
                                    else:
                                        st.metric("Growth Trend", "N/A")
                                
                                st.markdown("---")
                            
                            # Earnings History Section
                            if earnings_history:
                                st.markdown("#### 📊 Recent Earnings Performance")
                                
                                # Create table data - reverse order to show latest first
                                earnings_data = []
                                for earning in reversed(earnings_history[:5]):  # Reverse to show latest first
                                    surprise_pct = earning.get('surprise_percent', 0)
                                    if surprise_pct != 0 and surprise_pct != 'N/A':
                                        surprise_indicator = "🎯" if surprise_pct > 0 else "⚠️"
                                        surprise_str = f"{surprise_indicator}{surprise_pct:+.1%}"
                                    else:
                                        surprise_str = "📊 N/A"
                                    
                                    earnings_data.append({
                                        'Quarter': earning['quarter'],
                                        'Actual EPS': f"{earning['actual_eps']}" if earning['actual_eps'] != 'N/A' else 'N/A',
                                        'Estimate': f"{earning['estimate_eps']}" if earning['estimate_eps'] != 'N/A' else 'N/A',
                                        'Difference': f"{earning['surprise']}" if earning['surprise'] != 'N/A' else 'N/A',
                                        'Surprise': surprise_str
                                    })
                                
                                # Use st.dataframe instead of st.table to remove row numbers
                                st.dataframe(earnings_data, use_container_width=True, hide_index=True)
                
                # Enhanced Price Prediction - Collapsible
                if show_ml:
                    with st.expander("🔮 Enhanced Price Prediction", expanded=True):
                        try:
                            from enhanced_price_predictor import EnhancedPricePredictor
                            predictor = EnhancedPricePredictor(ticker)
                            prediction_result = predictor.generate_comprehensive_prediction()
                            
                            if prediction_result:
                                current_price = prediction_result['current_price']
                                
                                # Main prediction metrics
                                col1, col2, col3, col4, col5, col6 = st.columns(6)
                                with col1:
                                    st.metric("Current Price", f"${current_price:.2f}")
                                with col2:
                                    overall_score = prediction_result['overall_score']
                                    st.metric("Overall Score", f"{overall_score['score']}/{overall_score['max_score']}")
                                with col3:
                                    st.metric("Recommendation", overall_score['recommendation'])
                                with col4:
                                    st.metric("Features Used", prediction_result['features_used'])
                                with col5:
                                    st.metric("Data Sources", len(prediction_result['data_sources']))
                                with col6:
                                    st.metric("Prediction Date", prediction_result['prediction_date'][:10])
                                
                                # Price predictions by timeline
                                st.markdown("#### 🎯 Price Predictions by Timeline")
                                if prediction_result['prediction_horizons']:
                                    col1, col2, col3, col4 = st.columns(4)
                                    horizons = list(prediction_result['prediction_horizons'].keys())
                                    for i, horizon in enumerate(horizons[:4]):
                                        pred = prediction_result['prediction_horizons'][horizon]
                                        with [col1, col2, col3, col4][i]:
                                            st.metric(
                                                f"{horizon} Target", 
                                                f"${pred['price_target']:.2f}",
                                                f"{pred['price_change_pct']:+.1f}%"
                                            )
                                            st.caption(f"Confidence: {pred['confidence']:.1%}")
                                
                                # Price scenarios
                                st.markdown("#### 📊 Price Scenarios")
                                scenarios = prediction_result['scenarios']
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    bullish = scenarios['bullish']
                                    st.metric(
                                        "🟢 Bullish Scenario",
                                        f"${bullish['price_target']:.2f}",
                                        f"{bullish['price_change_pct']:+.1f}%"
                                    )
                                    st.caption(f"Probability: {bullish['probability']:.1%}")
                                    st.caption(bullish['description'])
                                
                                with col2:
                                    neutral = scenarios['neutral']
                                    st.metric(
                                        "🟡 Neutral Scenario",
                                        f"${neutral['price_target']:.2f}",
                                        f"{neutral['price_change_pct']:+.1f}%"
                                    )
                                    st.caption(f"Probability: {neutral['probability']:.1%}")
                                    st.caption(neutral['description'])
                                
                                with col3:
                                    bearish = scenarios['bearish']
                                    st.metric(
                                        "🔴 Bearish Scenario",
                                        f"${bearish['price_target']:.2f}",
                                        f"{bearish['price_change_pct']:+.1f}%"
                                    )
                                    st.caption(f"Probability: {bearish['probability']:.1%}")
                                    st.caption(bearish['description'])
                                
                                # Analyst targets
                                analyst_targets = prediction_result['analyst_targets']
                                if analyst_targets:
                                    st.markdown("#### 👔 Analyst Price Targets")
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    target_sources = ['yahoo_mean', 'enhanced_mean', 'consensus_mean']
                                    for i, source in enumerate(target_sources):
                                        if source in analyst_targets and analyst_targets[source] > 0:
                                            with [col1, col2, col3, col4][i]:
                                                target = analyst_targets[source]
                                                st.metric(
                                                    source.replace('_', ' ').title(),
                                                    f"${target:.2f}",
                                                    f"{((target - current_price) / current_price * 100):+.1f}%"
                                                )
                                
                                # Key factors
                                st.markdown("#### 🔍 Key Prediction Factors")
                                factors = overall_score['factors']
                                if factors:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**✅ Positive Factors:**")
                                        for factor in factors[:5]:
                                            st.markdown(f"• {factor}")
                                    with col2:
                                        st.markdown("**📊 Prediction Details:**")
                                        st.markdown(f"• Features Analyzed: {prediction_result['features_used']}")
                                        st.markdown(f"• Data Sources: {', '.join(prediction_result['data_sources'][:3])}")
                                        st.markdown(f"• Overall Confidence: {overall_score['percentage']:.1f}%")
                                
                            else:
                                st.warning("Enhanced price prediction not available")
                                
                        except Exception as e:
                            st.warning(f"Enhanced price prediction error: {str(e)}")
                            # Fallback to original ML prediction
                            if ml_prediction:
                                st.markdown("#### 🤖 Basic ML Prediction (Fallback)")
                                col1, col2, col3, col4, col5, col6 = st.columns(6)
                                with col1:
                                    direction_emoji = "🟢" if "BULLISH" in ml_prediction['direction'] else "🔴" if "BEARISH" in ml_prediction['direction'] else "🟡"
                                    st.markdown(f"**Direction:** {ml_prediction['direction']} {direction_emoji}")
                                with col2:
                                    st.metric("Confidence", f"{ml_prediction['confidence']:.1%}")
                                with col3:
                                    price_target = ml_prediction.get('price_target', 0)
                                    st.metric("Price Target", f"${price_target:.2f}")
                                with col4:
                                    expected_return = ml_prediction.get('expected_return', 0)
                                    st.metric("Expected Return", f"{expected_return:+.1%}")
                                with col5:
                                    scenarios = ml_prediction.get('scenario_probabilities', {'bullish': 0, 'bearish': 0})
                                    st.metric("Bullish Prob", f"{scenarios['bullish']:.1%}")
                                with col6:
                                    st.metric("Bearish Prob", f"{scenarios['bearish']:.1%}")
                
                # Enhanced Backtest Results - Collapsible
                if show_backtest and backtest_results:
                    with st.expander("📊 Enhanced Backtest Results", expanded=False):
                        # Compact info
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("Period", backtest_results.get('backtest_period', 'N/A'))
                        with col2:
                            st.metric("Features", backtest_results.get('enhanced_features', 0))
                        with col3:
                            st.metric("Win Rate", f"{backtest_results['win_rate']:.1%}")
                        with col4:
                            st.metric("Strategy Return", f"{backtest_results['strategy_total_return']:+.1%}")
                        with col5:
                            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                        with col6:
                            st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:+.1%}")
                        
                        # Additional metrics
                        col1, col2, col3, col4, col5, col6 = st.columns(6)
                        with col1:
                            st.metric("Prediction Accuracy", f"{backtest_results['prediction_accuracy']:.1%}")
                        with col2:
                            st.metric("Total Trades", f"{backtest_results['total_trades']}")
                        with col3:
                            st.metric("Avg Return/Trade", f"{backtest_results['strategy_avg_return']:+.2%}")
                        with col4:
                            signals = backtest_results['total_signals']
                            st.metric("Bullish Signals", signals['bullish'])
                        with col5:
                            st.metric("Bearish Signals", signals['bearish'])
                        with col6:
                            st.metric("High Conf Trades", backtest_results['high_conf_trades'])
                
                # Recommendation - Collapsible
                with st.expander("🎯 Investment Recommendation", expanded=True):
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
                        rec_color = "background: linear-gradient(135deg, #28a745, #20c997);"
                    elif score >= 3:
                        recommendation = "BUY"
                        confidence = "MEDIUM"
                        rec_color = "background: linear-gradient(135deg, #17a2b8, #6f42c1);"
                    elif score <= 1:
                        recommendation = "SELL"
                        confidence = "MEDIUM"
                        rec_color = "background: linear-gradient(135deg, #dc3545, #fd7e14);"
                    else:
                        recommendation = "HOLD"
                        confidence = "LOW"
                        rec_color = "background: linear-gradient(135deg, #6c757d, #495057);"
                    
                    # Compact recommendation display
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1:
                        # Set color based on recommendation
                        if "BUY" in recommendation:
                            rec_color = "#28a745"  # Green
                        elif "SELL" in recommendation:
                            rec_color = "#dc3545"  # Red
                        else:
                            rec_color = "#ffc107"  # Yellow/Orange for HOLD
                        
                        st.markdown(f"**Recommendation:** <span style='color: {rec_color}; font-weight: bold;'>{recommendation}</span>", unsafe_allow_html=True)
                    with col2:
                        st.metric("Confidence", confidence)
                    with col3:
                        st.metric("Score", f"{score}/7")
                    with col4:
                        st.metric("Factors", len(factors))
                    with col5:
                        st.metric("Risk Level", "🟢 Low" if score >= 4 else "🟡 Medium" if score >= 2 else "🔴 High")
                    with col6:
                        st.metric("Overall", "🟢" if score >= 4 else "🟡" if score >= 2 else "🔴")
                    
                    # Compact factors display
                    st.markdown(f"**✅ Key Factors:** {', '.join(factors[:4])}{'...' if len(factors) > 4 else ''}")
                    
                    # Risk factors
                    risk_factors = []
                    if tech_data['volatility_20d'] > 30:
                        risk_factors.append("High volatility")
                    if tech_data['rsi'] > 80:
                        risk_factors.append("Overbought")
                    if fundamental_data['debt_to_equity'] and fundamental_data['debt_to_equity'] > 1:
                        risk_factors.append("High debt")
                    
                    if risk_factors:
                        st.markdown(f"**⚠️ Risk Factors:** {', '.join(risk_factors)}")
                    else:
                        st.markdown("**⚠️ Risk Factors:** Low risk identified")
                
                # Data Sources - Collapsible
                with st.expander("📊 Data Sources", expanded=False):
                    # Collect all data sources with links
                    sources_with_links = []
                    
                    # Market data sources
                    sources_with_links.append({
                        'name': "📈 Yahoo Finance (Market Data)",
                        'url': "https://finance.yahoo.com",
                        'description': "Real-time market data, financials, and analyst ratings"
                    })
                    
                    # Enhanced news sources
                    if show_sentiment and sentiment_data.get('sources'):
                        news_sources = sentiment_data['sources']
                        for source in news_sources:
                            sources_with_links.append({
                                'name': f"📰 {source}",
                                'url': "#",
                                'description': "News sentiment analysis"
                            })
                    else:
                        sources_with_links.extend([
                            {
                                'name': "📰 NewsAPI (1000 requests/month free)",
                                'url': "https://newsapi.org",
                                'description': "Global news headlines and articles"
                            },
                            {
                                'name': "📰 Alpha Vantage News (500 calls/day free)",
                                'url': "https://www.alphavantage.co",
                                'description': "Financial news and market sentiment"
                            },
                            {
                                'name': "📰 Yahoo Finance RSS (Free)",
                                'url': f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
                                'description': "Real-time financial news feed"
                            },
                            {
                                'name': "📰 MarketWatch RSS (Free)",
                                'url': f"https://feeds.marketwatch.com/marketwatch/marketpulse/",
                                'description': "Market news and analysis"
                            },
                            {
                                'name': "📰 Google News RSS (Free)",
                                'url': f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
                                'description': "Comprehensive news coverage"
                            }
                        ])
                    
                    # Social media sources
                    if show_sentiment:
                        sources_with_links.extend([
                            {
                                'name': "🐦 Reddit API (Pushshift - Free)",
                                'url': "https://www.reddit.com",
                                'description': "Social sentiment from Reddit discussions"
                            },
                            {
                                'name': "🐦 StockTwits API (Free)",
                                'url': f"https://stocktwits.com/symbol/{ticker}",
                                'description': "Real-time social trading sentiment"
                            },
                            {
                                'name': "🐦 Social Media Sentiment Analysis",
                                'url': "#",
                                'description': "Advanced sentiment analysis from social platforms"
                            }
                        ])
                    
                    # Analyst data
                    if show_fundamentals:
                        sources_with_links.extend([
                            {
                                'name': "👔 Yahoo Finance Analyst Ratings",
                                'url': f"https://finance.yahoo.com/quote/{ticker}/analysis",
                                'description': "Professional analyst recommendations and targets"
                            },
                            {
                                'name': "👔 MarketWatch Analyst Ratings",
                                'url': f"https://www.marketwatch.com/investing/stock/{ticker}/analystestimates",
                                'description': "Analyst estimates and ratings"
                            },
                            {
                                'name': "👔 Finviz Rankings (Free)",
                                'url': f"https://finviz.com/quote.ashx?t={ticker}",
                                'description': "Financial metrics and rankings"
                            }
                        ])
                    
                    # Options data
                    if show_ml and ml_prediction:
                        sources_with_links.append({
                            'name': "📊 Options Flow Data",
                            'url': f"https://finance.yahoo.com/quote/{ticker}/options",
                            'description': "Options trading activity and sentiment"
                        })
                    
                    # Institutional data
                    if show_ml and ml_prediction:
                        sources_with_links.append({
                            'name': "🏛️ Institutional Holdings",
                            'url': f"https://finance.yahoo.com/quote/{ticker}/holders",
                            'description': "Institutional ownership and insider trading"
                        })
                    
                    # Economic data
                    if show_ml and ml_prediction:
                        sources_with_links.append({
                            'name': "🌍 Economic Indicators (VIX, Treasury Yields, Dollar Index)",
                            'url': "https://finance.yahoo.com/markets/indices",
                            'description': "Macroeconomic indicators and market sentiment"
                        })
                    
                    # Stock rankings data
                    if show_ml and ml_prediction:
                        sources_with_links.extend([
                            {
                                'name': "🏆 Finviz Stock Rankings (Free)",
                                'url': f"https://finviz.com/quote.ashx?t={ticker}",
                                'description': "Comprehensive stock rankings and metrics"
                            },
                            {
                                'name': "🏆 MarketWatch Rankings (Free)",
                                'url': f"https://www.marketwatch.com/investing/stock/{ticker}",
                                'description': "Market performance rankings"
                            },
                            {
                                'name': "🏆 Sector Performance Analysis",
                                'url': f"https://finance.yahoo.com/sector/technology",
                                'description': "Sector relative performance analysis"
                            },
                            {
                                'name': "🏆 Relative Strength Rankings",
                                'url': f"https://finviz.com/screener.ashx?v=111&f=cap_largeover",
                                'description': "Relative strength vs market and peers"
                            }
                        ])
                    
                    # Alternative data
                    if show_ml and ml_prediction:
                        sources_with_links.append({
                            'name': "🔍 Alternative Data Sources",
                            'url': "#",
                            'description': "Additional market intelligence and alternative data"
                        })
                    
                    # Display sources with clickable links
                    st.markdown("### 🔗 Data Sources & Links")
                    
                    # Create expandable sections for different categories
                    categories = {
                        "📈 Market Data": [s for s in sources_with_links if "Yahoo Finance" in s['name'] or "Market Data" in s['name']],
                        "📰 News Sources": [s for s in sources_with_links if "📰" in s['name']],
                        "🐦 Social Media": [s for s in sources_with_links if "🐦" in s['name']],
                        "👔 Analyst Data": [s for s in sources_with_links if "👔" in s['name']],
                        "📊 Advanced Data": [s for s in sources_with_links if any(x in s['name'] for x in ["📊", "🏛️", "🌍", "🏆", "🔍"])]
                    }
                    
                    for category, sources in categories.items():
                        if sources:
                            with st.expander(f"{category} ({len(sources)} sources)", expanded=False):
                                for source in sources:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        if source['url'] != "#":
                                            st.markdown(f"**{source['name']}** - {source['description']}")
                                        else:
                                            st.markdown(f"**{source['name']}** - {source['description']}")
                                    with col2:
                                        if source['url'] != "#":
                                            st.markdown(f"[🔗 Visit]({source['url']})")
                                        else:
                                            st.markdown("🔗 N/A")
                    
                    # Quick access section
                    st.markdown("### ⚡ Quick Access")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**[📈 {ticker} on Yahoo Finance](https://finance.yahoo.com/quote/{ticker})**")
                        st.markdown(f"**[📊 {ticker} Analysis](https://finance.yahoo.com/quote/{ticker}/analysis)**")
                    
                    with col2:
                        st.markdown(f"**[🏆 {ticker} on Finviz](https://finviz.com/quote.ashx?t={ticker})**")
                        st.markdown(f"**[📰 {ticker} News](https://news.google.com/rss/search?q={ticker}+stock)**")
                    
                    with col3:
                        st.markdown(f"**[🐦 {ticker} on StockTwits](https://stocktwits.com/symbol/{ticker})**")
                        st.markdown(f"**[📊 {ticker} Options](https://finance.yahoo.com/quote/{ticker}/options)**")
                    
                    # Data freshness
                    st.markdown("---")
                    st.markdown("**📅 Data Freshness:** Real-time market data, Latest news sentiment, Current analyst ratings")
                    st.markdown("**🔄 Update Frequency:** Market data updates continuously, News sentiment refreshed on analysis")
                
            else:
                st.error(f"❌ No data available for {ticker}")
                
        except Exception as e:
            st.error(f"❌ Error analyzing {ticker}: {str(e)}")

else:
    # Welcome message
    st.markdown("""
    ## Welcome to the Professional Stock Analyzer! 🏆
    
    ### How to Use:
    1. **Enter a stock ticker** in the sidebar (e.g., AAPL, TSLA, MSFT)
    2. **Select analysis options** (charts, ML, fundamentals, sentiment, backtest)
    3. **Enable Fast Mode** for quicker analysis (skips ML and backtest)
    4. **Click "Analyze Stock"** to run comprehensive analysis
    
    ### Features:
    - 📊 **Real-time market data** and technical indicators
    - 🤖 **Machine learning predictions** with SHAP explanations
    - 💰 **Fundamental analysis** with valuation metrics
    - 😊 **Sentiment analysis** from news and social media
    - 📈 **Interactive charts** with technical overlays
    - 📊 **Backtesting** with performance metrics
    - ⚡ **Fast Mode** for quick analysis without ML/backtest
    
    ### Getting Started:
    Enter a stock ticker in the sidebar and click "Analyze Stock" to begin!
    """)

# Footer
st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This analysis is for educational purposes only. Always consult with qualified financial advisors before making investment decisions.")
