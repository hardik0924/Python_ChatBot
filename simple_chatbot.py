import telebot
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from Excel file
data = pd.read_excel('chatbot_data.xlsx')

questions = data['Question'].tolist()
answers = data['Answer'].tolist()

# Initialize the vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Function to find the best match for the user's question
def get_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, X)
    best_match_index = similarities.argmax()
    return answers[best_match_index]

# Telegram bot token (replace 'YOUR_TELEGRAM_BOT_TOKEN' with your actual token)
bot = telebot.TeleBot('7434832021:AAHrthIcyd-UIa0jnch50kwEJzhKrKPZjK0')

# Handle the '/start' command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! I'm your AI chatbot. How can I assist you today?")

# Handle messages from the user
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    response = get_response(message.text)
    bot.reply_to(message, response)

# Start the bot
bot.polling()
