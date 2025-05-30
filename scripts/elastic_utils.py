from elasticsearch import Elasticsearch
from elasticsearch import RequestError

# initialize Elasticsearch
es_address = 'your_es_address'
es = Elasticsearch(es_address, verify_certs=False, ssl_show_warn=False)

mapping = {
    "settings": {
        "mapping": {
            "total_fields": {
                "limit": 2000
            }
        }
    },
    "mappings": {
        "properties": {
            "created_at": {
                "type": "date",
                "format": "strict_date_time"
            }
        }
    }
}


# create index
def initialize_index(index_name, logger):
    try:
        logger.info(f'Creating ElasticSearch index \'{index_name}\' if it doesn\'t exist...')
        es.indices.create(
            index=index_name,
            body=mapping
        )
        logger.info(f'Created and mapped the ElasticSearch index \'{index_name}\'')
        return True
    except RequestError as re:
        logger.warning(f'Index \'{index_name}\' already exists: {str(re)}')
        return True
    except Exception as err:
        logger.info(f'initialize_index - initialize_index(): Cannot create ElasticSearch index \'{index_name}\': {str(err)}. Caused by: {str(err)}')
        return False


def index_doc(index_name, tweet, logger):
    try:
        id_doc = tweet["id"]
        response = es.index(index=index_name, id=id_doc, document=tweet, refresh=True)
        if response.get('result') != 'created':
            message = f'ES response for tweet {tweet} is: {response}'
            logger.info(message)
        return response
    except Exception as err:
        logger.error(f'[ELASTIC] Error when saving the tweet: \'{id_doc}\'. Caused by: {str(err)}')

