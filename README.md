# Google-Play-Reviews-Sentiment-Analysis
Project Overview
This project is a web-based tool for analyzing Google Play app reviews. It scrapes app reviews from the Google Play Store using the app's ID, performs sentiment analysis, generates word clouds for key topics, and visualizes the sentiment distribution through a bar chart. The user interface is built with Gradio, allowing users to input an app's ID and get immediate insights into customer sentiment and review topics.

Features
Scraping Reviews: Retrieves all the reviews for a given app ID from the Google Play Store. It uses the google_play_scraper library to collect the most recent reviews.
Sentiment Analysis: Uses a fine-tuned sentiment analysis model from Hugging Face to classify reviews as positive, neutral, or negative.
Word Cloud Generation: Utilizes the WordCloud library to create visual representations of key terms found in the app's reviews.
Topic Modeling: Applies Top2Vec for unsupervised topic modeling to identify main themes across the reviews.
Visualization: Generates bar charts to show the distribution of sentiments and displays word clouds for the top topics.
Gradio Interface: The project uses Gradio to create a simple, interactive interface for users to input the app ID and view results.
How it Works
Scrape Reviews:

Reviews are collected using the app ID inputted by the user. The app reviews are scraped in English and sorted by the newest first.
Sentiment Analysis:

The project uses a pre-trained transformer model (cardiffnlp/twitter-roberta-base-sentiment-latest) to classify each review into positive, neutral, or negative sentiments.
The sentiment distribution is visualized as a bar chart and saved as an image.
Word Clouds and Topic Modeling:

Reviews are cleaned and processed to generate word clouds based on the top terms and topics discovered using Top2Vec.
Word clouds are created for up to 5 identified topics and saved as images.
Gradio Interface:

Users can input the Google Play app ID to start the analysis.
The output includes:
Scraping status (indicating success or failure).
A sentiment distribution bar chart.
Word clouds of the top words from topics found using topic modeling.
A list of sample reviews for reference.
The name of the app's developer.
Prerequisites
To run this project, you need the following installed:

Python 3.7 or higher
Required libraries listed in requirements.txt (see below for installation instructions)
