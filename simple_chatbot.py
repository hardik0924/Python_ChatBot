import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_excel('D:\\VS Code\\pythonchatbot\\chatbot_data.xlsx')


# Prepare the vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Question'])

def chatbot_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, X)
    best_match_index = similarities.argmax()
    best_match_score = similarities[0, best_match_index]

    if best_match_score > 0.2:  # Threshold to determine if the response is relevant
        return data['Answer'].iloc[best_match_index]
    else:
        return "I'm sorry, I don't understand the question."

# Simple loop to interact with the chatbot
print("Chatbot is ready to talk! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("Chatbot:", response)
