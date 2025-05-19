import pandas as pd
from textblob import TextBlob

# -----------------------------
# 1. Simulated Social Media Dataset
# -----------------------------
data = {
    "username": ["user1", "user2", "user3", "user4", "user5"],
    "text": [
        "I love this new movie! It's amazing 😍",
        "Feeling so sad and depressed today...",
        "Just won a hackathon, I’m so excited!!!",
        "Traffic was horrible. Worst day ever.",
        "What a peaceful and beautiful morning 🌅"
    ]
}
df = pd.DataFrame(data)

# -----------------------------
# 2. Sentiment Analysis using TextBlob
# -----------------------------
df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# -----------------------------
# 3. Basic Rule-Based Emotion Mapping
# -----------------------------
def map_emotion(polarity):
    if polarity > 0.5:
        return 'joy'
    elif 0 < polarity <= 0.5:
        return 'trust'
    elif polarity == 0:
        return 'neutral'
    elif -0.5 < polarity < 0:
        return 'sadness'
    else:  # polarity <= -0.5
        return 'anger'

df['emotion'] = df['polarity'].apply(map_emotion)

# -----------------------------
# 4. Output Results
# -----------------------------
print(df[['username', 'text', 'polarity', 'subjectivity', 'emotion']])
