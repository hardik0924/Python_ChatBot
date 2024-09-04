import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load chatbot data
data = pd.read_excel('chatbot_data.xlsx')
question = data['Question'].tolist()
answer = data['Answer'].tolist()


# Tokenization
def tokenize(text):
    tokens = text.lower().split()
    print(f"Tokenization: {tokens}")
    return tokens


# Convert to Embedding with fixed length for each question
def embed_question(tokens, max_tokens=10):
    embeddings = []
    for token in tokens:
        token_embedding = np.array([ord(char) for char in token])
        if len(token_embedding) < max_tokens:
            token_embedding = np.pad(token_embedding,
                                     (0, max_tokens - len(token_embedding)),
                                     'constant')
        else:
            token_embedding = token_embedding[:max_tokens]
        embeddings.append(token_embedding)

    # Flatten and pad/truncate to ensure fixed length for each question embedding
    embeddings = np.array(embeddings).flatten()
    if len(embeddings) < max_tokens * max_tokens:
        embeddings = np.pad(embeddings,
                            (0, max_tokens * max_tokens - len(embeddings)),
                            'constant')
    else:
        embeddings = embeddings[:max_tokens * max_tokens]
    print(f"Converted to Embedding: {embeddings}")
    return embeddings


# Find the best match using cosine similarity
def find_best_match(user_embedding, question_embeddings):
    similarities = cosine_similarity(user_embedding.reshape(1, -1),
                                     question_embeddings)
    best_match_idx = np.argmax(similarities)
    return best_match_idx


# Chatbot response function
def chatbot_response(update, context):
    user_input = update.message.text
    tokens = tokenize(user_input)
    user_embedding = embed_question(tokens)

    # Embed all questions with fixed length
    question_embeddings = np.array(
        [embed_question(tokenize(question)) for question in question])

    # Find the best match
    best_match_idx = find_best_match(user_embedding, question_embeddings)
    response = answer[best_match_idx]

    # Display the flow
    update.message.reply_text(f"Tokenization: {tokens}")
    update.message.reply_text(
        f"Converted to Embedding: {user_embedding.tolist()}")
    update.message.reply_text(
        f"Finding the best match using cosine similarity...")
    update.message.reply_text(f"Response: {response}")


# Main function to run the bot
def main():
    bot_token = 'bot token'  # Replace with your bot's token
    updater = Updater(bot_token, use_context=True)
    dp = updater.dispatcher

    # Handle normal messages
    dp.add_handler(
        MessageHandler(Filters.text & ~Filters.command, chatbot_response))

    # Start the bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
