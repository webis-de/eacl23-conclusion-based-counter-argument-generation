from debater_python_api.api.debater_api import DebaterApi
from debater_python_api.api.sentence_level_index.client.sentence_query_base import SimpleQuery
from debater_python_api.api.sentence_level_index.client.sentence_query_request import SentenceQueryRequest
from debater_python_api.api.clients.narrative_generation_client import Polarity

debater_api = DebaterApi('0abeffa5335cc942fc7c43e75d41fe33L05')
pro_con_client = debater_api.get_pro_con_client()
arg_quality_client = debater_api.get_argument_quality_client()

def get_stances(targets, conc):
    conc_to_targets = list(zip(targets, conc))
    sentence_topic_dicts = [{'sentence' : x[0], 'topic' : x[1] if x[1] != None else x[0] } for x in conc_to_targets]
    scores = pro_con_client.run(sentence_topic_dicts)
    return scores

def get_arg_scores(topic, sentences):
    sentence_topic_dicts = [{'sentence' : sentence, 'topic' : topic } for sentence in sentences]
    scores = arg_quality_client.run(sentence_topic_dicts)
    return list(zip(sentences, scores))