import pickle

TWEET_FILE = "generated_trump_tweets.pickle"

tweets = pickle.load(open(TWEET_FILE, 'rb'))

for tweet in tweets:
    print(tweet)
    print("_"*200)
