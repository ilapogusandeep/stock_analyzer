"""Smoke test: package + compat shims all load."""


def test_package_imports():
    import stockiq
    from stockiq.core.analyzer import UniversalStockAnalyzer, PortfolioManager, create_charts
    from stockiq.data.collector import EnhancedDataCollector
    from stockiq.data.institutional import EnhancedInstitutionalData
    from stockiq.models.predictor import EnhancedPricePredictor
    from stockiq.models.sentiment import AdvancedSentimentAnalyzer

    assert stockiq.__version__
    assert UniversalStockAnalyzer is not None
    assert PortfolioManager is not None
    assert create_charts is not None
    assert EnhancedDataCollector is not None
    assert EnhancedInstitutionalData is not None
    assert EnhancedPricePredictor is not None
    assert AdvancedSentimentAnalyzer is not None


def test_backward_compat_shims():
    from stock_analyzer import UniversalStockAnalyzer
    from enhanced_data_collector import EnhancedDataCollector
    from enhanced_institutional_data import EnhancedInstitutionalData
    from enhanced_price_predictor import EnhancedPricePredictor
    from advanced_sentiment_analyzer import AdvancedSentimentAnalyzer

    assert UniversalStockAnalyzer is not None
    assert EnhancedDataCollector is not None
    assert EnhancedInstitutionalData is not None
    assert EnhancedPricePredictor is not None
    assert AdvancedSentimentAnalyzer is not None
