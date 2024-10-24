# Google Play Reviews Sentiment Analysis

## Overview

This project is designed to scrape Google Play Store reviews for a given app, analyze the sentiment of those reviews, and perform topic modeling to generate insights. The application uses Gradio to provide a simple and intuitive user interface, allowing users to input the app ID and receive detailed feedback on sentiment distribution, topics, and more.

## Key Features

- **Scraping Reviews**: Automatically fetches reviews from the Google Play Store for a specified app ID.
- **Sentiment Analysis**: Classifies reviews as positive, neutral, or negative using a pre-trained transformer model.
- **Topic Modeling**: Identifies key topics in reviews using `Top2Vec`.
- **Visualizations**: Generates bar charts for sentiment distribution and word clouds for top topics.
- **Gradio Interface**: Simple and easy-to-use interface where users can input an app ID and view results.

## How it Works

1. **Scraping Reviews**: 
   - Fetches reviews using the `google_play_scraper` library for a specific app based on the app ID.
   
2. **Sentiment Analysis**:
   - Uses the `cardiffnlp/twitter-roberta-base-sentiment-latest` model to classify each review's sentiment.
   - Displays sentiment distribution as a bar chart.

3. **Topic Modeling**:
   - Implements `Top2Vec` to uncover the main themes in the reviews.
   - Generates word clouds for the top terms in each topic.

4. **Gradio Interface**:
   - Users input the app ID to get review insights, including:
     - Scraping status
     - Sentiment distribution
     - Word clouds for top topics
     - Sample reviews for reference
     - The app's developer name

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/google-play-reviews-sentiment-analysis.git
   cd google-play-reviews-sentiment-analysis
