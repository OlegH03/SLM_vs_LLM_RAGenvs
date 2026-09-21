from src import config
import json
import random
# als import import config.py
# en.json laden, eine funktion zur verfügung stellen, die mir für eine Query
# deterministisch entsprechend der noise ratio id, queries, positives, negatives gibt. Quasi die Ausgabe eine Query Card


def load_data(path=None):
    with open(path or config.DATA_PATH) as f:
        data = [json.loads(line) for line in f]
    return data


def get_query_card(query_id, data, positive_key="positive", only_negative=False):
    """Returns for one Query: id, query, answer, context (mixed from noise ratio)"""
    id = query_id
    query = data[query_id]["query"]
    answer = data[query_id]["answer"]
    random.seed(config.SEED + query_id)
    negatives = data[query_id]["negative"]
    if only_negative:
        k = min(config.TOP_K, len(negatives))
        context = random.sample(negatives, k)
    else:
        data_list_positives = data[query_id]["positive"]
        data_list_negatives = data[query_id]["negative"]
        amount_positives = min(
            round(config.TOP_K * (1 - config.NOISE_RATIO)), len(data_list_positives)
        )
        amount_negatives = min(
            round(config.TOP_K * config.NOISE_RATIO), len(data_list_negatives)
        )
        list_positives = random.sample(data_list_positives, amount_positives)
        list_negatives = random.sample(data_list_negatives, amount_negatives)
        context = []
        for i in range(len(list_positives)):
            context.append(list_positives[i])
        for i in range(len(list_negatives)):
            context.append(list_negatives[i])
    random.shuffle(context)
    card = {"id": id, "query": query, "answer": answer, "context": context}
    return card
