"""
Financial Sentiment Analyzer using FinBERT
Analyzes sentiment of earnings call excerpts and returns confidence scores.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import Dict, List, Tuple
import json

class FinancialSentimentAnalyzer:
    """Sentiment analysis for financial text using FinBERT."""
    
    def __init__(self):
        # Load FinBERT model (pre-trained on financial text)
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Label mapping: 0 = negative, 1 = neutral, 2 = positive
        self.labels = ['negative', 'neutral', 'positive']
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a text passage.
        
        Args:
            text: Text to analyze (e.g., earnings call excerpt)
            
        Returns:
            Dictionary with sentiment scores and confidence
        """
        # Truncate to model's max length
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Convert to 0-1 scale where 0 = very negative/cautious, 0.5 = neutral, 1 = very positive/confident
        sentiment_score = self._to_confidence_score(predicted_class, confidence)
        
        return {
            'sentiment_label': self.labels[predicted_class],
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'is_confident': confidence > 0.7,
            'is_neutral': predicted_class == 1
        }
    
    def _to_confidence_score(self, predicted_class: int, confidence: float) -> float:
        """
        Convert FinBERT output to 0-1 confidence scale.
        - negative: 0.0 to 0.33
        - neutral: 0.33 to 0.66
        - positive: 0.66 to 1.0
        """
        if predicted_class == 0:  # negative
            return confidence * 0.33
        elif predicted_class == 1:  # neutral
            return 0.33 + (confidence * 0.33)
        else:  # positive
            return 0.66 + (confidence * 0.34)
    
    def analyze_topic(self, text: str, topic: str) -> Dict:
        """
        Analyze sentiment for a specific topic (e.g., "margins", "guidance").
        
        Args:
            text: Text passage
            topic: Topic keyword to focus on
            
        Returns:
            Sentiment analysis with topic context
        """
        # Extract sentences containing the topic
        sentences = re.split(r'[.!?]+', text)
        topic_sentences = [s for s in sentences if topic.lower() in s.lower()]
        
        if not topic_sentences:
            return {
                'topic': topic,
                'found': False,
                'message': f"No sentences containing '{topic}' found"
            }
        
        # Analyze each relevant sentence
        analyses = [self.analyze(s) for s in topic_sentences if len(s.strip()) > 20]
        
        if not analyses:
            return {'topic': topic, 'found': False, 'message': 'No substantial content found'}
        
        # Aggregate scores
        avg_score = sum(a['sentiment_score'] for a in analyses) / len(analyses)
        
        return {
            'topic': topic,
            'found': True,
            'sentence_count': len(analyses),
            'average_sentiment_score': round(avg_score, 3),
            'sentiment_label': 'positive' if avg_score > 0.66 else ('neutral' if avg_score > 0.33 else 'negative'),
            'sample_sentences': topic_sentences[:2]
        }
    
    def compare_sentiment(self, current_text: str, previous_text: str, topic: str) -> Dict:
        """
        Compare sentiment on a specific topic between two transcripts.
        
        Returns:
            Sentiment shift analysis
        """
        current_analysis = self.analyze_topic(current_text, topic)
        previous_analysis = self.analyze_topic(previous_text, topic)
        
        if not current_analysis.get('found') or not previous_analysis.get('found'):
            return {
                'topic': topic,
                'error': 'Topic not found in one or both transcripts',
                'current_found': current_analysis.get('found', False),
                'previous_found': previous_analysis.get('found', False)
            }
        
        current_score = current_analysis['average_sentiment_score']
        previous_score = previous_analysis['average_sentiment_score']
        shift = current_score - previous_score
        
        return {
            'topic': topic,
            'current_sentiment': current_score,
            'previous_sentiment': previous_score,
            'sentiment_shift': round(shift, 3),
            'direction': 'more confident' if shift > 0 else ('less confident' if shift < 0 else 'unchanged'),
            'magnitude': 'high' if abs(shift) > 0.15 else ('moderate' if abs(shift) > 0.08 else 'low'),
            'explanation': self._generate_explanation(topic, shift, current_score, previous_score)
        }
    
    def _generate_explanation(self, topic: str, shift: float, current: float, previous: float) -> str:
        """Generate human-readable explanation of sentiment shift."""
        if abs(shift) < 0.05:
            return f"Management tone on {topic} remained stable quarter over quarter."
        
        direction = "improved" if shift > 0 else "became more cautious"
        magnitude = "significantly" if abs(shift) > 0.15 else "moderately"
        
        return f"Management tone on {topic} {magnitude} {direction} (shift of {abs(shift):.2f}). " \
               f"Sentiment moved from {previous:.2f} to {current:.2f}."
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts in batch for efficiency.
        
        Args:
            texts: List of text passages to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results


# Example usage
if __name__ == "__main__":
    analyzer = FinancialSentimentAnalyzer()
    
    # Example earnings call excerpts
    current_excerpt = "We expect revenue to grow 15% next quarter. Margins remain strong at 45%. The competitive environment is intense but we are well positioned."
    previous_excerpt = "Revenue growth is expected to be modest. Margins are under pressure from input costs. Competition is increasing."
    
    # Analyze single text
    print("=== Single Text Analysis ===")
    result = analyzer.analyze(current_excerpt)
    print(json.dumps(result, indent=2))
    
    # Analyze by topic
    print("\n=== Topic Analysis ===")
    topic_result = analyzer.analyze_topic(current_excerpt, "margin")
    print(json.dumps(topic_result, indent=2))
    
    # Compare sentiment between two transcripts
    print("\n=== Sentiment Comparison ===")
    comparison = analyzer.compare_sentiment(current_excerpt, previous_excerpt, "margin")
    print(json.dumps(comparison, indent=2))
