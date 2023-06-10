import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

analyser = SentimentIntensityAnalyzer()
def vader_sentiment_result(text):
  scores = analyser.polarity_scores(text)
  if scores["neg"] > scores["pos"]:
    return 0
  return 1

def sentiment_analyse(comments_df):
    """
    This function analyses the sentiment of a given dataset of comments.

    Parameters
    ----------
    comments_df : pd.DataFrame
        A pandas DataFrame containing the comments to be analysed.

    Returns
    -------
    tuple[float, float]
        A tuple containing the percentage of positive comments and the percentage of negative comments.

    """
    comments_df["score"] = comments_df["text"].apply(vader_sentiment_result)
    
    negative_comments_number = comments_df['score'].value_counts()[0]
    positive_comments_number = comments_df['score'].value_counts()[1]
    total_comments = positive_comments_number + negative_comments_number

    positive_comments_percentage = round((positive_comments_number / total_comments)*100, 2)
    negative_comments_percentage = round((negative_comments_number / total_comments)*100, 2)

    comments_df = comments_df.drop(columns="score")

    return positive_comments_percentage, negative_comments_percentage
