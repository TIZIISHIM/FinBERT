# Financial Sentiment Analyzer (FinBERT)

A production-ready sentiment analysis tool for financial text, specifically designed for earnings call transcripts and corporate communications. Uses FinBERT, a BERT model fine-tuned on financial documents.

## Features

- Single Text Analysis: Analyze sentiment of any financial text passage
- Topic-Specific Analysis: Focus sentiment analysis on specific topics (margins, guidance, revenue, etc.)
- Sentiment Comparison: Compare sentiment between two transcripts (quarter-over-quarter)
- Confidence Scoring: Returns confidence scores for every prediction
- Batch Processing: Analyze multiple texts efficiently

## Installation

```bash
pip install -r requirements.txt
