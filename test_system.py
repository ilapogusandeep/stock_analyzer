#!/usr/bin/env python3
"""
Test script for Universal Stock Analyzer
"""

import sys
import subprocess

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False
    
    try:
        import shap
        print("✅ shap imported successfully")
    except ImportError as e:
        print(f"❌ shap import failed: {e}")
        return False
    
    return True

def test_data_fetch():
    """Test that we can fetch stock data"""
    print("\n📊 Testing data fetch...")
    
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d")
        
        if hist.empty:
            print("❌ No data returned for AAPL")
            return False
        
        print(f"✅ Successfully fetched {len(hist)} days of data for AAPL")
        print(f"   Latest price: ${hist['Close'].iloc[-1]:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        return False

def test_analyzer():
    """Test the main analyzer"""
    print("\n🏆 Testing analyzer...")
    
    try:
        # Import the analyzer
        from stock_analyzer import UniversalStockAnalyzer
        
        # Create analyzer instance
        analyzer = UniversalStockAnalyzer("AAPL")
        
        # Test basic functionality
        hist = analyzer.stock.history(period="5d")
        if hist.empty:
            print("❌ No historical data available")
            return False
        
        print("✅ Analyzer created successfully")
        print(f"   Data points: {len(hist)}")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏆 Universal Stock Analyzer - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Fetch Test", test_data_fetch),
        ("Analyzer Test", test_analyzer)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Quick Start:")
        print("  Command line: ./analyze AAPL")
        print("  Web interface: ./analyze --web")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
