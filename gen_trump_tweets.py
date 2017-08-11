import keras
import pickle
import os.path
import numpy as np

# Hyperparameters
TWEET_LEN = 140
MAX_LEN = 200

# Other Constants
TWEET_FILE = "generated_trump_tweets.pickle"
MODEL_FILE = "Models/goodTrump(3-1024).h5"
BATCH_SIZE = 128

# Construct dictionaries to convert from tokens to strings and back
char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ()#@,.:;-$?!/'\"\n")
num2char = dict(enumerate(char_list, 1))
num2char[0] = "<PAD>"
char2num = dict(zip(num2char.values(), num2char.keys()))
VOCAB_SIZE = len(char_list) + 1

# Load pre-trained model
model = keras.models.load_model(MODEL_FILE)
print("Restored model")

# Load already generated tweets
tweets = []
if os.path.isfile(TWEET_FILE):
    tweets = pickle.load(open(TWEET_FILE, "rb"))

def generate_tweets(num_tweets):
    # Construct one-hot character array and set initial characters
    ix = np.zeros((num_tweets, 1, VOCAB_SIZE))
    for a in ix:
        a[0, np.random.randint(VOCAB_SIZE)] = 1
    while True:
        # Get the character-probabilities from the model
        iy = model.predict(ix)[:, ix.shape[1]-1, :]
        # Select the next characters based on the output of the model
        c = np.array([np.random.choice(np.arange(VOCAB_SIZE), p=ps) for ps in iy])
        #c = np.array([c = np.argmax(ps)) for ps in iy])   
        # Break if all tweets are over (if we encounter the <PAD>-character) or have reached MAX_LEN 
        if np.all(c==0) or ix.shape[1] >= MAX_LEN:
            break
        # Add new characters to the character array
        nx = np.eye(VOCAB_SIZE)[c].reshape(num_tweets, 1, VOCAB_SIZE)
        ix = np.concatenate((ix, nx), axis=1)
    # Convert the one-hot character array to a list of strings
    tweets = ["".join(num2char[n] for n in np.argmax(tweet, axis=1) if n != 0) for tweet in ix]
    return tweets

while True:
    # Append the generated tweets to tweets array
    tweets.extend(generate_tweets(BATCH_SIZE))
    # Save the new array to TWEET_FILE
    pickle.dump(tweets, open(TWEET_FILE, "wb"))
    print("Wrote", len(tweets), "tweets")