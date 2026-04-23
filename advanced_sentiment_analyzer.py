#!/usr/bin/env python3
"""
Advanced Sentiment Analysis with Multiple Techniques
"""

import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis using multiple techniques"""
    
    def __init__(self):
        self.setup_sentiment_models()
    
    def setup_sentiment_models(self):
        """Setup various sentiment analysis models"""
        # Enhanced keyword-based sentiment
        self.positive_keywords = {
            # Strong positive
            'strong': 0.8, 'excellent': 0.9, 'outstanding': 0.9, 'exceptional': 0.9,
            'beat': 0.8, 'exceed': 0.8, 'surpass': 0.8, 'outperform': 0.8,
            'bullish': 0.7, 'optimistic': 0.7, 'confident': 0.6, 'positive': 0.6,
            'growth': 0.6, 'increase': 0.6, 'rise': 0.6, 'gain': 0.6,
            'profit': 0.7, 'earnings': 0.6, 'revenue': 0.5, 'success': 0.7,
            'breakthrough': 0.8, 'innovation': 0.6, 'upgrade': 0.7, 'upgrade': 0.7,
            'buy': 0.8, 'strong buy': 0.9, 'outperform': 0.8, 'overweight': 0.7,
            'momentum': 0.6, 'rally': 0.7, 'surge': 0.7, 'soar': 0.8,
            'record': 0.7, 'milestone': 0.6, 'achievement': 0.6, 'win': 0.7,
            'leading': 0.6, 'dominant': 0.7, 'premium': 0.6, 'quality': 0.5,
            'robust': 0.6, 'solid': 0.5, 'stable': 0.4, 'resilient': 0.6,
            'promising': 0.6, 'potential': 0.5, 'opportunity': 0.5, 'upside': 0.6,
            'expansion': 0.6, 'acquisition': 0.5, 'partnership': 0.5, 'deal': 0.4,
            'dividend': 0.5, 'yield': 0.4, 'return': 0.5, 'value': 0.4,
            'undervalued': 0.7, 'cheap': 0.6, 'bargain': 0.6, 'attractive': 0.6
        }
        
        self.negative_keywords = {
            # Strong negative
            'weak': -0.8, 'poor': -0.8, 'terrible': -0.9, 'awful': -0.9,
            'miss': -0.8, 'disappoint': -0.8, 'fail': -0.8, 'crash': -0.9,
            'bearish': -0.7, 'pessimistic': -0.7, 'concern': -0.6, 'worry': -0.6,
            'decline': -0.6, 'decrease': -0.6, 'fall': -0.6, 'drop': -0.6,
            'loss': -0.7, 'deficit': -0.7, 'debt': -0.5, 'risk': -0.5,
            'problem': -0.6, 'issue': -0.5, 'challenge': -0.4, 'headwind': -0.5,
            'sell': -0.8, 'strong sell': -0.9, 'underperform': -0.8, 'underweight': -0.7,
            'volatility': -0.4, 'uncertainty': -0.5, 'instability': -0.6, 'turbulence': -0.5,
            'crisis': -0.8, 'recession': -0.7, 'downturn': -0.6, 'correction': -0.5,
            'overvalued': -0.7, 'expensive': -0.6, 'bubble': -0.8, 'speculative': -0.5,
            'regulatory': -0.4, 'litigation': -0.6, 'investigation': -0.6, 'penalty': -0.7,
            'competition': -0.3, 'threat': -0.5, 'pressure': -0.4, 'headwind': -0.5,
            'cut': -0.6, 'reduce': -0.5, 'lower': -0.5, 'downgrade': -0.7,
            'warning': -0.6, 'caution': -0.5, 'concern': -0.5, 'risk': -0.5,
            'volatile': -0.5, 'unstable': -0.6, 'uncertain': -0.5, 'unpredictable': -0.5
        }
        
        # Context modifiers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.5, 'significantly': 1.5,
            'substantially': 1.5, 'dramatically': 2.0, 'massively': 2.0,
            'slightly': 0.5, 'somewhat': 0.7, 'moderately': 0.8, 'fairly': 0.8,
            'quite': 1.2, 'rather': 1.1, 'pretty': 1.1, 'really': 1.3
        }
        
        # Negation words
        self.negations = {'not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
        
        # Financial context words
        self.financial_context = {
            'earnings': 1.2, 'revenue': 1.1, 'profit': 1.3, 'margin': 1.1,
            'guidance': 1.2, 'forecast': 1.1, 'outlook': 1.1, 'target': 1.1,
            'analyst': 1.1, 'rating': 1.1, 'recommendation': 1.2, 'price': 1.0,
            'stock': 1.0, 'shares': 1.0, 'market': 1.0, 'trading': 1.0,
            'investor': 1.1, 'shareholder': 1.1, 'dividend': 1.1, 'yield': 1.0
        }
    
    def analyze_text_advanced(self, text: str) -> Dict:
        """Advanced sentiment analysis of text"""
        if not text or not isinstance(text, str):
            return self._default_sentiment()
        
        text = text.lower().strip()
        if len(text) < 3:
            return self._default_sentiment()
        
        # Multiple analysis techniques
        keyword_score = self._keyword_analysis(text)
        contextual_score = self._contextual_analysis(text)
        intensity_score = self._intensity_analysis(text)
        financial_score = self._financial_context_analysis(text)
        
        # Weighted combination
        final_score = (
            keyword_score * 0.4 +
            contextual_score * 0.3 +
            intensity_score * 0.2 +
            financial_score * 0.1
        )
        
        # Determine sentiment label and confidence
        sentiment_label, confidence = self._determine_sentiment(final_score)
        
        return {
            'score': final_score,
            'label': sentiment_label,
            'confidence': confidence,
            'breakdown': {
                'keyword': keyword_score,
                'contextual': contextual_score,
                'intensity': intensity_score,
                'financial': financial_score
            },
            'techniques_used': ['keyword', 'contextual', 'intensity', 'financial']
        }
    
    def _keyword_analysis(self, text: str) -> float:
        """Keyword-based sentiment analysis"""
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
        
        total_score = 0.0
        word_count = 0
        
        for i, word in enumerate(words):
            word_score = 0.0
            
            # Check positive keywords
            if word in self.positive_keywords:
                word_score = self.positive_keywords[word]
            # Check negative keywords
            elif word in self.negative_keywords:
                word_score = self.negative_keywords[word]
            
            # Apply negation
            if word_score != 0 and i > 0:
                prev_word = words[i-1]
                if prev_word in self.negations:
                    word_score = -word_score
            
            # Apply intensifiers
            if word_score != 0 and i > 0:
                prev_word = words[i-1]
                if prev_word in self.intensifiers:
                    word_score *= self.intensifiers[prev_word]
            
            total_score += word_score
            if word_score != 0:
                word_count += 1
        
        return total_score / max(word_count, 1)
    
    def _contextual_analysis(self, text: str) -> float:
        """Contextual sentiment analysis"""
        # Look for patterns and phrases
        patterns = {
            # Positive patterns
            r'beat\s+(?:expectations?|estimates?)': 0.8,
            r'exceed\s+(?:expectations?|estimates?)': 0.8,
            r'strong\s+(?:growth|performance|results?)': 0.7,
            r'record\s+(?:high|revenue|earnings?)': 0.8,
            r'positive\s+(?:outlook|guidance|trend)': 0.7,
            r'bullish\s+(?:sentiment|outlook)': 0.8,
            r'upgrade\s+(?:to|from)': 0.7,
            r'buy\s+(?:rating|recommendation)': 0.8,
            r'strong\s+buy': 0.9,
            r'outperform\s+(?:rating|recommendation)': 0.8,
            
            # Negative patterns
            r'miss\s+(?:expectations?|estimates?)': -0.8,
            r'disappoint\s+(?:investors?|market)': -0.7,
            r'weak\s+(?:performance|results?)': -0.7,
            r'decline\s+(?:in|of)': -0.6,
            r'concern\s+(?:about|over)': -0.5,
            r'risk\s+(?:of|to)': -0.5,
            r'sell\s+(?:rating|recommendation)': -0.8,
            r'strong\s+sell': -0.9,
            r'underperform\s+(?:rating|recommendation)': -0.8,
            r'downgrade\s+(?:to|from)': -0.7,
            r'volatility\s+(?:concerns?|risks?)': -0.5,
            r'uncertainty\s+(?:about|over)': -0.5
        }
        
        total_score = 0.0
        pattern_count = 0
        
        for pattern, score in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                total_score += score * len(matches)
                pattern_count += len(matches)
        
        return total_score / max(pattern_count, 1) if pattern_count > 0 else 0.0
    
    def _intensity_analysis(self, text: str) -> float:
        """Analyze intensity and emotional content"""
        # Exclamation marks
        exclamations = text.count('!')
        exclamation_score = min(exclamations * 0.1, 0.5)
        
        # Caps lock words (indicating emphasis)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        caps_score = min(len(caps_words) * 0.05, 0.3)
        
        # Question marks (uncertainty)
        questions = text.count('?')
        question_score = -min(questions * 0.05, 0.2)
        
        # Length of text (more detailed = more confident)
        length_score = min(len(text) / 1000, 0.2)
        
        return exclamation_score + caps_score + question_score + length_score
    
    def _financial_context_analysis(self, text: str) -> float:
        """Analyze financial context relevance"""
        words = re.findall(r'\b\w+\b', text)
        financial_words = [word for word in words if word in self.financial_context]
        
        if not financial_words:
            return 0.0
        
        # Calculate financial relevance score
        relevance_score = len(financial_words) / len(words)
        
        # Weight by financial context importance
        weighted_score = sum(self.financial_context[word] for word in financial_words) / len(financial_words)
        
        return relevance_score * (weighted_score - 1.0)  # Normalize around 0
    
    def _determine_sentiment(self, score: float) -> Tuple[str, float]:
        """Determine sentiment label and confidence"""
        abs_score = abs(score)
        
        if score > 0.3:
            label = "STRONGLY_POSITIVE"
            confidence = min(abs_score * 2, 0.95)
        elif score > 0.1:
            label = "POSITIVE"
            confidence = min(abs_score * 2, 0.85)
        elif score > -0.1:
            label = "NEUTRAL"
            confidence = max(0.3, 1.0 - abs_score * 2)
        elif score > -0.3:
            label = "NEGATIVE"
            confidence = min(abs_score * 2, 0.85)
        else:
            label = "STRONGLY_NEGATIVE"
            confidence = min(abs_score * 2, 0.95)
        
        return label, confidence
    
    def _default_sentiment(self) -> Dict:
        """Default sentiment when analysis fails"""
        return {
            'score': 0.0,
            'label': 'NEUTRAL',
            'confidence': 0.3,
            'breakdown': {'keyword': 0.0, 'contextual': 0.0, 'intensity': 0.0, 'financial': 0.0},
            'techniques_used': []
        }
    
    def analyze_multiple_texts(self, texts: List[str]) -> Dict:
        """Analyze multiple texts and aggregate results"""
        if not texts:
            return self._default_sentiment()
        
        results = []
        for text in texts:
            result = self.analyze_text_advanced(text)
            results.append(result)
        
        # Aggregate scores
        scores = [r['score'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Weighted average by confidence
        if confidences:
            weights = [c for c in confidences]
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            weighted_score = 0.0
            avg_confidence = 0.3
        
        # Determine overall sentiment
        overall_label, overall_confidence = self._determine_sentiment(weighted_score)
        
        return {
            'score': weighted_score,
            'label': overall_label,
            'confidence': avg_confidence,
            'overall_confidence': overall_confidence,
            'text_count': len(texts),
            'individual_results': results,
            'score_distribution': {
                'positive': len([s for s in scores if s > 0.1]),
                'neutral': len([s for s in scores if -0.1 <= s <= 0.1]),
                'negative': len([s for s in scores if s < -0.1])
            }
        }
    
    def get_sentiment_explanation(self, result: Dict) -> str:
        """Generate human-readable explanation of sentiment analysis"""
        score = result['score']
        label = result['label']
        confidence = result['confidence']
        
        if label == "STRONGLY_POSITIVE":
            explanation = f"Very positive sentiment ({score:+.2f}) with high confidence ({confidence:.1%})"
        elif label == "POSITIVE":
            explanation = f"Positive sentiment ({score:+.2f}) with moderate confidence ({confidence:.1%})"
        elif label == "NEUTRAL":
            explanation = f"Neutral sentiment ({score:+.2f}) with low confidence ({confidence:.1%})"
        elif label == "NEGATIVE":
            explanation = f"Negative sentiment ({score:+.2f}) with moderate confidence ({confidence:.1%})"
        else:  # STRONGLY_NEGATIVE
            explanation = f"Very negative sentiment ({score:+.2f}) with high confidence ({confidence:.1%})"
        
        # Add breakdown details
        breakdown = result.get('breakdown', {})
        if breakdown:
            techniques = []
            if breakdown.get('keyword', 0) != 0:
                techniques.append(f"keyword analysis ({breakdown['keyword']:+.2f})")
            if breakdown.get('contextual', 0) != 0:
                techniques.append(f"contextual patterns ({breakdown['contextual']:+.2f})")
            if breakdown.get('intensity', 0) != 0:
                techniques.append(f"intensity analysis ({breakdown['intensity']:+.2f})")
            if breakdown.get('financial', 0) != 0:
                techniques.append(f"financial context ({breakdown['financial']:+.2f})")
            
            if techniques:
                explanation += f". Based on: {', '.join(techniques)}"
        
        return explanation

def main():
    """Test the advanced sentiment analyzer"""
    analyzer = AdvancedSentimentAnalyzer()
    
    # Test cases
    test_texts = [
        "AAPL reports strong quarterly earnings beating estimates by 15%",
        "The stock shows bullish momentum with institutional buying",
        "Analyst upgrades price target citing strong fundamentals",
        "Company faces regulatory challenges in key markets",
        "Weak performance disappoints investors",
        "The stock is trading at fair value with neutral outlook",
        "BREAKTHROUGH innovation drives record revenue growth!",
        "Concerns about market volatility and uncertainty",
        "Strong buy recommendation with 25% upside potential",
        "Sell rating due to overvaluation and competitive pressure"
    ]
    
    print("🏆 Advanced Sentiment Analysis Test")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text_advanced(text)
        explanation = analyzer.get_sentiment_explanation(result)
        
        print(f"\n{i}. Text: {text}")
        print(f"   Result: {result['label']} ({result['score']:+.2f}) - {result['confidence']:.1%} confidence")
        print(f"   Explanation: {explanation}")
    
    # Test multiple texts
    print(f"\n{'='*50}")
    print("📊 Multiple Texts Analysis")
    
    multi_result = analyzer.analyze_multiple_texts(test_texts)
    print(f"Overall: {multi_result['label']} ({multi_result['score']:+.2f})")
    print(f"Confidence: {multi_result['confidence']:.1%}")
    print(f"Texts analyzed: {multi_result['text_count']}")
    print(f"Distribution: {multi_result['score_distribution']}")

if __name__ == "__main__":
    main()
