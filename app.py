import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from google_play_scraper import reviews_all, app, Sort
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from top2vec import Top2Vec
from wordcloud import WordCloud
import traceback
import os

# Ensure the directory exists
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to scrape reviews and get app details from Google Play
def scrape_reviews(app_id):
    try:
        # Scraping reviews for the given app ID
        reviews = reviews_all(
            app_id,
            sleep_milliseconds=50,  # Prevent overwhelming the server
            lang='en',
            country='us',
            sort=Sort.NEWEST  # Fetch reviews in the newest order
        )
        
        # Check if reviews are returned
        if reviews:
            print(f"Collected {len(reviews)} reviews for the app: {app_id}")
            # Extract the review content for display
            review_contents = [review['content'] for review in reviews]
            return f"Scraped {len(reviews)} reviews successfully.", review_contents
        else:
            print("No reviews retrieved.")
            return "No reviews found.", []
    except Exception as error:
        print(f"Failed to scrape reviews due to: {error}")
        traceback.print_exc()
        return f"Error occurred while scraping reviews: {error}", []

# Function to get developer name from Google Play
def get_developer_name(app_id):
    try:
        app_info = app(app_id)
        developer_name = app_info.get('developer', 'Unknown Developer')
        return developer_name
    except Exception as error:
        print(f"Error retrieving developer name: {error}")
        return "Error retrieving developer name"

# Function to clean and process the list of reviews
def clean_reviews(data_list):
    try:
        # Remove empty or non-string entries from the review list
        valid_reviews = [review for review in data_list if isinstance(review, str) and review.strip()]
        
        # Convert the cleaned list into a DataFrame
        review_df = pd.DataFrame(valid_reviews, columns=["Reviews"])
        return review_df
    except Exception as error:
        print(f"An error occurred while cleaning reviews: {error}")
        return pd.DataFrame()  # Return an empty DataFrame if something goes wrong

# Function to generate bar chart for sentiment distribution
def generate_bar_chart(sentiment_dist, output_path):
    try:
        labels = list(sentiment_dist.keys())
        sizes = list(sentiment_dist.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=['blue', 'green', 'red'])  # Adjust colors as needed
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.title('Sentiment Distribution')
        plt.xticks(rotation=45)  # Rotate labels for better readability
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path
    except Exception as error:
        print(f"Error generating bar chart: {error}")
        return None

# Sentiment analysis function
def sentiment_analysis(df):
    try:
        # Load model and tokenizer
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        labels = []
        for review in df['Reviews']:
            if isinstance(review, str) and review.strip():  # Check if review is valid
                sentiment = sentiment_task(review)[0]['label']
                labels.append(sentiment)
            else:
                labels.append("INVALID_REVIEW")

        df['Sentiment'] = labels
        label_counts = df['Sentiment'].value_counts(normalize=True) * 100

        # Generate bar chart
        ensure_directory("output")  # Ensure the output directory exists
        bar_chart_path = os.path.join("output", "sentiment_bar_chart.png")
        generate_bar_chart(label_counts.to_dict(), bar_chart_path)

        return df, label_counts.to_dict(), bar_chart_path  # Return DataFrame, sentiment distribution, and bar chart path
    except Exception as error:
        print(f"Error in sentiment analysis: {error}")
        traceback.print_exc()
        return None, None, None

# Function to create a word cloud from text and save the image
def generate_word_cloud(text, output_path):
    try:
        # Generate the word cloud with specified dimensions and background color
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        
        # Plot the word cloud using matplotlib
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Hide the axes for a cleaner look
        
        # Save the image to the specified file path
        plt.savefig(output_path)
        plt.close()  # Close the plot to free memory
        return output_path
    except Exception as error:
        print(f"An error occurred while generating the word cloud: {error}")
        return None

# Function for topic modeling using Top2Vec
def topic_modeling(data_list):
    try:
        # Filter out empty or non-string data
        valid_reviews = [review for review in data_list if isinstance(review, str) and review.strip()]

        # Check if there is sufficient data for modeling
        if len(valid_reviews) < 50:
            return "Insufficient data for topic modeling.", []

        # Initialize and run the Top2Vec model
        print("Starting topic modeling with Top2Vec...")
        model = Top2Vec(documents=valid_reviews, speed="deep-learn", workers=4, min_count=1)

        # Extract topics from the model
        topic_words, word_scores, topic_nums = model.get_topics()

        # Check if topics were generated
        if not len(topic_words):
            return "No topics were identified.", []

        print(f"First 3 topics: {topic_words[:3]}")  # Log first 3 topics for review

        wordcloud_paths = []
        ensure_directory("output")  # Ensure the output directory is available
        for i in range(min(5, len(topic_words))):  # Limit to 5 topics for word clouds
            # Create a string from the words in each topic
            topic_content = " ".join(topic_words[i])
            output_image = os.path.join("output", f"wordcloud_topic_{i+1}.png")

            # Generate word cloud for the topic
            generated_image = generate_word_cloud(topic_content, output_image)
            if generated_image:
                wordcloud_paths.append(generated_image)
            else:
                print(f"Failed to generate word cloud for topic {i+1}")

        return "Word clouds created successfully.", wordcloud_paths
    except Exception as error:
        print(f"An error occurred during topic modeling: {error}")
        traceback.print_exc()
        return f"Topic modeling error: {str(error)}", []

# Main function to process and analyze app reviews
def analyze_app_reviews(app_id):
    try:
        print(f"Starting analysis for App ID: {app_id}")
        status, reviews_list = scrape_reviews(app_id)

        # Return if no reviews are found or an error occurs
        if not reviews_list:
            return status, None, None, None, None, None

        # Get developer name
        developer_name = get_developer_name(app_id)

        # Clean the reviews data
        cleaned_df = clean_reviews(reviews_list)
        if cleaned_df.empty:
            return "No valid reviews after cleaning.", None, None, None, None, developer_name

        # Conduct sentiment analysis
        cleaned_df, sentiment_dist, bar_chart_path = sentiment_analysis(cleaned_df)
        if sentiment_dist is None:
            return "Error encountered during sentiment analysis.", None, None, None, None, developer_name

        # Perform topic modeling
        topic_status, wordcloud_images = topic_modeling(reviews_list)

        # Return status, sentiment chart, word cloud images, developer name, and first 5 reviews for reference
        sample_reviews = "\n".join(cleaned_df['Reviews'].head(5))  # Preview first 5 reviews
        return status, bar_chart_path, topic_status, wordcloud_images, sample_reviews, developer_name
    except Exception as error:
        print(f"An error occurred while analyzing reviews: {error}")
        traceback.print_exc()
        return "Error during review analysis.", None, None, None, None, "Unknown Developer"

# Gradio Interface
iface = gr.Interface(
    fn=analyze_app_reviews,
    inputs=gr.Textbox(label="App ID"),
    outputs=[
        gr.Textbox(label="Scraping Status"),  # Display status of scraping
        gr.Image(label="Sentiment Bar Chart"),  # Display the generated bar chart only
        gr.Textbox(label="Topic Modeling Status"),  # Display status of topic modeling
        gr.Gallery(label="Top Words Word Cloud"),  # Display word clouds in a gallery
        gr.Textbox(label="Sample Reviews"),  # Display sample reviews for reference
        gr.Textbox(label="Developer Name")  # Display the developer's name
    ],
    title="Google Play Reviews Sentiment Analysis",
    description="Enter the app ID to scrape reviews from Google Play Store, perform sentiment analysis and topic modeling."
)

iface.launch()