import tweepy
import webbrowser
import time
import csv
import argparse
import os


def compose_dict_obj(raw_data, keys, search_words):
    """
    Return a dictionary of selected keys from raw_data
    """
    d = {}
    for key in keys:
        if key == "keyword":
            d[key] = search_words
        else:
            d[key] = raw_data.get(key)
    return d


# the handler is time.sleep(15 * 60) if we reach the rate limit.
def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.TweepError:
            time.sleep(15 * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process each .")
    parser.add_argument("-k", "--keyword", help="The keyword to search", required=True)
    parser.add_argument("--num", help="Max number of data to scrape", default=10000)
    parser.add_argument(
        "-o", "--output", help="The filepath of the output file", default="./raw.csv"
    )

    args = parser.parse_args()

    consumer_key = os.environ["CONSUMER_KEY"]
    consumer_secret = os.environ["CONSUMER_SECRET"]

    # search_words = "Covid19 vaccine -filter:retweets"
    search_words = args.keyword
    max_num = int(args.num)
    csv_file = args.output

    keys = [
        "created_at",
        "id",
        "full_text",
        "entities",
        "source",
        "user",
        "coordinates",
        "place",
        "is_quote_status",
        "retweet_count",
        "favorite_count",
        "possibly_sensitive",
        "keyword",
    ]

    callback_uri = "oob"  # https://cfe.sh/twitter/callback
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret, callback_uri)
    redirect_url = auth.get_authorization_url()
    print(redirect_url)

    webbrowser.open(redirect_url)
    user_pint_input = input("What's the pin?")
    auth.get_access_token(user_pint_input)
    print(auth.access_token, auth.access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    tweetCursor = tweepy.Cursor(
        api.search, q=search_words, lang="en", tweet_mode="extended"
    ).items(max_num)
    try:
        with open(csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for i, tweet in enumerate(tweetCursor):
                if i % 1000 == 0:
                    if i == max_num:
                        break
                    print(i)
                big_json = tweet._json
                if "retweeted_status" in big_json:
                    data = big_json["retweeted_status"]
                else:
                    data = big_json
                struct_data = compose_dict_obj(data, keys, search_words)
                writer.writerow(struct_data)
    except IOError:
        print("I/O error")