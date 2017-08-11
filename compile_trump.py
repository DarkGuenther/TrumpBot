"""Fetches all of Trumps tweets and saves them to a pickle"""
import pickle
import json

RAW_FILE = "trump_tweets_raw.txt"
PICKLE_FILE = "trump_tweets.pickle"

raw = json.load(open(RAW_FILE))

tweets = [e["text"] for e in raw]

clean_tweets = []

for t in tweets:
    parts = []
    for p in t.split(sep=" "):
        if "://" in p:
            p = ""
        parts.append(p)
    t = " ".join(parts)
    clean_tweets.append(t)

tweets = clean_tweets

pickle.dump(tweets, open(PICKLE_FILE, "wb"))
print("Dumped", len(tweets), "tweets to", PICKLE_FILE)