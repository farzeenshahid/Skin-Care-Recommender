
from flask import Flask, request, jsonify
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from bson import ObjectId
app = Flask(__name__)

# Initialize MongoDB client and database
client = MongoClient("mongodb://localhost:27017")  # Replace with your MongoDB connection string
db = client['recommendation_system_database']  # Replace with your DB name
reviews_collection = db['reviews']  # Replace with your collection name

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mishitm/Sentiment-Classifier-Skincare")
model = AutoModelForSequenceClassification.from_pretrained("mishitm/Sentiment-Classifier-Skincare")
sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API. Use /analyze_review?review_id=<id> to analyze a review."

# Endpoint to get a review and perform sentiment analysis
@app.route("/analyze_review", methods=["GET"])
def analyze_review():
    review_id = request.args.get('review_id')  # Review ID from the query parameters
    try:
        # Convert review_id to ObjectId
        review_id = ObjectId(review_id)
    except Exception as e:
        return jsonify({"error": "Invalid review ID format"}), 400
    if not review_id:
        return jsonify({"error": "Review ID is required"}), 400

    # Fetch the review from the database
    review = reviews_collection.find_one({"_id": review_id})
    
    if not review:
        return jsonify({"error": "Review not found"}), 404
    
    # Perform sentiment analysis
    text = review['ReviewText']  # Assuming review text is stored in 'ReviewText' field
    sentiment_result = sentiment_model(text)

    # Update the review with the sentiment and confidence score
    sentiment = sentiment_result[0]['label']
    confidence_score = sentiment_result[0]['score']
    
    reviews_collection.update_one(
        {"_id": review_id},
        {"$set": {"Sentiment": sentiment, "Confidence_score": confidence_score}}
    )

    return jsonify({
        "Review": review['ReviewText'],
        "Sentiment": sentiment,
        "Confidence_score": confidence_score
    })
# Endpoint to analyze all reviews in the database

# @app.route("/analyze_reviews_with_no_sentiment", methods=["GET"])
# def analyze_reviews_with_no_sentiment():
#     batch_size = 100  # Set the batch size for processing
#     page = int(request.args.get('page', 1))  # Get the page number, default is 1

#     # Skip to the correct page based on batch size and page number
#     reviews_cursor = reviews_collection.find({"Sentiment": None}).skip((page - 1) * batch_size).limit(batch_size)
    
#     updated_reviews = []
#     for review in reviews_cursor:
#         text = review['ReviewText']  # Assuming review text is stored in 'ReviewText' field
#         sentiment_result = sentiment_model(text)

#         # Perform sentiment analysis and extract sentiment and confidence score
#         sentiment = sentiment_result[0]['label']
#         confidence_score = sentiment_result[0]['score']

#         # Update the review with sentiment and confidence score
#         reviews_collection.update_one(
#             {"_id": review["_id"]},
#             {"$set": {"Sentiment": sentiment, "Confidence_score": confidence_score}}
#         )
        
#         updated_reviews.append({
#             "Review": review['ReviewText'],
#             "Sentiment": sentiment,
#             "Confidence_score": confidence_score
#         })

#     # Return the updated reviews for the current batch
#     return jsonify({
#         "updated_reviews": updated_reviews,
#         "next_page": page + 1
#     })

# Endpoint to analyze all reviews with no sentiment and update them
@app.route("/analyze_reviews_with_no_sentiment", methods=["GET"])
def analyze_reviews_with_no_sentiment():
    # Fetch reviews where Sentiment is None
    reviews = reviews_collection.find({"Sentiment": None})  # Filter reviews with no sentiment
    
    updated_reviews = []

    for review in reviews:
        text = review['ReviewText']  # Assuming review text is stored in 'ReviewText' field
         # Truncate the text to fit the model's maximum token limit
        encoded_text = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Decode the truncated text back to string for the sentiment model
        truncated_text = tokenizer.decode(encoded_text["input_ids"][0], skip_special_tokens=True)
        
        sentiment_result = sentiment_model(truncated_text)

        # Perform sentiment analysis and extract sentiment and confidence score
        sentiment = sentiment_result[0]['label']
        confidence_score = sentiment_result[0]['score']

        # Update the review with sentiment and confidence score
        reviews_collection.update_one(
            {"_id": review["_id"]},
            {"$set": {"Sentiment": sentiment, "Confidence_score": confidence_score}}
        )
        
        updated_reviews.append({
            "Review": review['ReviewText'],
            "Sentiment": sentiment,
            "Confidence_score": confidence_score
        })

    return jsonify({"updated_reviews": updated_reviews})

if __name__ == "__main__":
    app.run(debug=True)
