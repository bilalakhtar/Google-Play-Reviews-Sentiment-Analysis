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

2. **Install the Dependencies**:
   Make sure you have Python 3.7 or higher installed. Install the required libraries using:
   ```bash
   pip install -r requirements.txt

3. **Run the Application**:
   Start the Gradio interface with:
   ```bash
   python app.py

## Usage
   -Enter the Google Play app ID (for example, com.whatsapp for WhatsApp) in the Gradio interface.
   -The app will:
      1. Scrape the reviews.
      2. Run sentiment analysis and topic modeling.
      3. Display the sentiment bar chart, word clouds for topics, and a preview of sample reviews.

## Requirements
   You need the following libraries to run this project:
   ```txt
   gradio
   google-play-scraper
   pandas
   matplotlib
   transformers
   torch
   top2vec
   wordcloud
   ```

   To install these, you can run:
   ```bash
   pip install gradio google-play-scraper pandas matplotlib transformers torch top2vec wordcloud
   ```

## Project Structure
   ```txt
   google-play-reviews-sentiment-analysis/
   │
   ├── app.py                  # Main application code
   ├── requirements.txt         # Dependencies for the project
   ├── output/                  # Folder where generated images (bar charts, word clouds) are saved
   └── README.md                # Project documentation
   ```
## Functionality Description
   1. Scrape Reviews
      - Function: scrape_reviews(app_id)
      - Description: Scrapes reviews from Google Play for a given app using its app ID.
   2. Sentiment Analysis
      - Function: sentiment_analysis(df)
      - Description: Analyzes the sentiment of the reviews and generates a sentiment distribution bar chart.
   3. Topic Modeling
      - Function: topic_modeling(data_list)
      - Description: Uses Top2Vec to identify topics in the reviews and generate word clouds for the top topics.
   4. Generate Word Cloud
      - Function: generate_word_cloud(text, output_path)
      - Description: Creates a word cloud from the most frequent words in the reviews and saves the result as an image.
   5. Gradio Interface
      - File: app.py
      - Function: analyze_app_reviews(app_id)
      - Description: Integrates all functionalities (scraping, sentiment analysis, topic modeling) into a user-friendly Gradio interface.

## Known Issues and Limitations
   - Google Play Restrictions: Excessive scraping might be throttled by Google Play. Handle requests responsibly.
   - Sentiment Misclassification: Some reviews may be incorrectly classified due to the limitations of the sentiment model.
   - Performance: Topic modeling can be slow on large datasets. Ensure you have adequate system resources for large-scale scraping and processing.

## Future Improvements
   - Add an option to filter reviews by rating or date.
   - Implement caching to store reviews and avoid repeated scraping for the same app.
   - Enhance error handling for scenarios like network failures or invalid app IDs.

## License
   - This project is licensed under the MIT License. See the LICENSE file for details.








