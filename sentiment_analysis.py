import ktrain
from ktrain.text.sentiment import SentimentAnalyzer

classifier = SentimentAnalyzer()

def predict_sentiments(sentences):
    sentiments = classifier.predict(sentences, max_length=512, truncation=True)
    sentiments_final = []
    for i in range(0, len(sentiments)):
        if str(sentiments[i]).startswith("{'NEU"):
            sentiments_final.append("Neutral")
        elif str(sentiments[i]).startswith("{'POS"):
            sentiments_final.append("Positive")
        else:
            sentiments_final.append("Negative")
    return sentiments_final
