import re
import ast
import pandas as pd
import argparse


class JsonParser:
    """Extract hashtags from entities and location from users columns."""

    def __init__(self, json_string):
        self.json_string = json_string

    def convert_to_dict(self):
        """Convert a json string to a Python dictionary object"""
        self.json_dict = ast.literal_eval(self.json_string)
        return self.json_dict

    def extract_hashtags(self):
        """Extract hashtag contents in a json dictionary object, as a list"""
        self.convert_to_dict()
        hashtags_list = []
        for text in self.json_dict["hashtags"]:
            hashtags_list.append(text["text"])

        return hashtags_list

    def extract_location(self):
        """Extract location information in a json dictionary object, as a string"""
        self.convert_to_dict()
        return self.json_dict["location"]


# json_str_entities = "{'hashtags': [{'text': 'COVID19', 'indices': [91, 99]}, {'text': 'EUvaccinationdays', 'indices': [224, 242]}], 'symbols': [], 'user_mentions': [], 'urls': [{'url': 'https://t.co/4Xa17PQkMv', 'expanded_url': 'https://www.pscp.tv/Ursulavonderleyen/1yoJMAOEyXlJQ', 'display_url': 'pscp.tv/Ursulavonderle…', 'indices': [245, 268]}]}"
# json_str_user = "{'id': 124237063, 'id_str': '124237063', 'name': 'Francis S. Collins', 'screen_name': 'NIHDirector', 'location': 'Bethesda, Maryland, USA', 'description': 'Official Twitter account of Francis S. Collins, M.D., Ph.D., NIH Director.  NIH…Turning Discovery Into Health ®. Privacy Policy: https://t.co/QTI46zY00q', 'url': 'http://t.co/chM4AXez5s', 'entities': {'url': {'urls': [{'url': 'http://t.co/chM4AXez5s', 'expanded_url': 'http://www.nih.gov/about/director/', 'display_url': 'nih.gov/about/director/', 'indices': [0, 22]}]}, 'description': {'urls': [{'url': 'https://t.co/QTI46zY00q', 'expanded_url': 'http://bit.ly/2jQa5Rk', 'display_url': 'bit.ly/2jQa5Rk', 'indices': [129, 152]}]}}, 'protected': False, 'followers_count': 158635, 'friends_count': 129, 'listed_count': 2256, 'created_at': 'Thu Mar 18 18:32:15 +0000 2010', 'favourites_count': 114, 'utc_offset': None, 'time_zone': None, 'geo_enabled': True, 'verified': True, 'statuses_count': 3475, 'lang': None, 'contributors_enabled': False, 'is_translator': False, 'is_translation_enabled': False, 'profile_background_color': '022330', 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme15/bg.png', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme15/bg.png', 'profile_background_tile': False, 'profile_image_url': 'http://pbs.twimg.com/profile_images/1282408119889530880/FMvG8yam_normal.jpg', 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/1282408119889530880/FMvG8yam_normal.jpg', 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/124237063/1354810851', 'profile_link_color': '0084B4', 'profile_sidebar_border_color': 'A8C7F7', 'profile_sidebar_fill_color': 'C0DFEC', 'profile_text_color': '333333', 'profile_use_background_image': True, 'has_extended_profile': False, 'default_profile': False, 'default_profile_image': False, 'following': False, 'follow_request_sent': False, 'notifications': False, 'translator_type': 'none'}"

# entity = JsonParser(json_str_entities)
# user = JsonParser(json_str_user)

# assert isinstance(entity.convert_to_dict(), dict)
# assert isinstance(user.convert_to_dict(), dict)
# assert entity.extract_hashtags() == ['COVID19', 'EUvaccinationdays']
# assert user.extract_location() == 'Bethesda, Maryland, USA'
# print("success!")


def remove_duplicates(df, by=["full_text"]):
    """
    Remove duplicates from raw data file by specific columns and save results in file with name given.
    """
    boolean_mask = df.duplicated(subset=by, keep="first")
    df = df[~boolean_mask]
    return df


def split_city_province(text, mode=0):
    """Split city and province in the location column"""
    if ", " in text and len(text.split(", ")) == 2 and mode == 0:
        city, province = text.split(", ")
        return city
    elif ", " in text and len(text.split(", ")) == 2 and mode == 1:
        city, province = text.split(", ")
        return province
    else:
        return text


def process_tweets(text):
    """Exclude mentions, urls, and html reference characters in a string using regular expression"""
    text = re.sub("(\@|https:\/\/)\S+", "", text)  # remove mentions and urls
    text = re.sub(r"&[a-z]+;", "", text)  # exclude html reference characters
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process each .")
    parser.add_argument(
        "-f",
        "--file",
        help="The file to be processed",
        default="./raw.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="The filepath to save after process",
        default="./preprocessed.csv",
    )

    args = parser.parse_args()
    df = pd.read_csv(args.file)
    # Remove duplicates
    df = remove_duplicates(df, by=["full_text"])

    # Extract hashtags from entities column (raw data)
    df["entities"] = df["entities"].apply(lambda x: JsonParser(x).extract_hashtags())

    # Extract location from user column (raw data)
    df["location"] = df["user"].apply(lambda x: JsonParser(x).extract_location())

    # Split the location into city, province and country
    df["city"] = df["location"].apply(split_city_province)
    df["province"] = df["location"].apply(split_city_province, mode=1)
    df["country"] = ""

    # Preprocess the `full_text` column
    df["full_text"] = df["full_text"].apply(process_tweets)

    # create a column called `count_hashtags` that represnets the number of hastags
    df["count_hashtags"] = df["entities"].apply(lambda x: len(x))

    # Delete the untidy columns
    df = df.drop(columns=["entities", "location", "user"])
    df.to_csv(args.output)
