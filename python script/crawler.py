import os
import re
import csv
import json
import tweepy
import logging
import requests
from elastic_utils import index_doc
from elastic_utils import initialize_index

bearer_token="your_bearer_token"

raw_tweets_text_header = ["id", "text"]
all_imgs_header = ["id_tweet", "path"]

BASE_DIR = "./crawling/"
PATH_TO_RAW_TEXT = BASE_DIR + "raw_tweets_text.csv"
PATH_TO_ALL_IMGS = BASE_DIR + "all_imgs.csv"
INDEX_NAME = "t4sa-2.0"


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    l.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(formatter)
    l.addHandler(handler)


def save_images(tweet_id, urls):
    for i, url in enumerate(urls):
        r = requests.get(url)
        if r.status_code == 200:
            img_data = r.content
            path = BASE_DIR + str(tweet_id)[:5] + "/"
            try:
                if not os.path.exists(path):
                    os.makedirs(path)

                final_path = path + str(tweet_id) + "-" + str(i) + ".jpg"
                with open(final_path, 'wb') as f:
                    f.write(img_data)

                with open(PATH_TO_ALL_IMGS, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    partial_path = str(tweet_id)[:5] + "/" + str(tweet_id) + "-" + str(i) + ".jpg"
                    writer.writerow([tweet_id, partial_path])
            except Exception as err:
                logger.error(f'[IMAGE] Error when saving the image: \'{url}\'. Caused by: {str(err)}')


def save_tweet(tweet_id, tweet_text):
    try:
        with open(PATH_TO_RAW_TEXT, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([tweet_id, tweet_text])
    except Exception as err:
        logger.error(f'[TWEET] Error when saving the tweet: \'{tweet_id}\'. Caused by: {str(err)}')


def check_data(data):
    try:
        if data.get("data").get("lang") == "en" and data.get("data").get("id") and data.get("data").get("text") \
                and data.get("includes") and not data.get("data").get("referenced_tweets"):
            if len(re.findall(r'\w+', data.get("data").get("text"))) > 5:
                return True
    except Exception as err:
            logger.error(f'[TWEET] Error when checking tweet caused by: {str(err)}')
            return False
    return False


class T4SACrawler(tweepy.StreamingClient):

    def __init__(self, bearer):
        super().__init__(bearer)
        if not os.path.exists(PATH_TO_RAW_TEXT):
            initialize_index(INDEX_NAME, logger)
            logger.info("Index initialized")
            os.makedirs(BASE_DIR)
            with open(PATH_TO_RAW_TEXT, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(raw_tweets_text_header)

            with open(PATH_TO_ALL_IMGS, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(all_imgs_header )

        print('Crawler initialized!')
        logger.info("Started!")

    def on_data(self, raw_data):
        decoded_data = raw_data.decode("utf-8")
        json_data = json.loads(decoded_data)

        if check_data(json_data):
            medias = json_data.get("includes").get("media")
            urls = []
            if medias:
                for media in medias:
                    if media.get("type") == "photo":
                        urls.append(media.get("url"))

            if urls:
                tweet_id = int(json_data.get("data").get("id"))    #length=19
                tweet_text = json_data.get("data").get("text")

                tweet_to_save = json_data.get("data")               
                tweet_to_save["urls"] = ' '.join([str(url) for url in urls])
                save_images(tweet_id, urls)
                save_tweet(tweet_id, tweet_text)
                index_doc(INDEX_NAME, tweet_to_save, logger)
                

if __name__ == "__main__":
    setup_logger('crawler_log', './crawler.log')
    logger = logging.getLogger('crawler_log')

    crawler = T4SACrawler(bearer_token)
    crawler.sample(expansions=["attachments.media_keys", "referenced_tweets.id"],
                   media_fields=["url", "type", "media_key"],
                   tweet_fields=["lang", "created_at", "id", "text"]
                   )
