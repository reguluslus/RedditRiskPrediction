import utils
from collections import Counter
from googleapiclient.errors import HttpError
from empath import Empath
import retrieve_data_from_db
import time
import json
import numpy as np
import pandas as pd

def get_user_empath_score(reddit_user_list):
    empath_scores = []
    lexicon = Empath()
    for index in range(len(reddit_user_list)):
        user = reddit_user_list.loc[index, "Author"]
        comments = retrieve_data_from_db.select_user_comments(user)
        scores = []
        for comment in comments:
            try:
                result_dict = lexicon.analyze(comment,
                                              categories=["anger", "hate","nervousness","positive_emotion"])
                scores.append(list(result_dict.values()))
            except Exception as e:
                print(e)
                pass
        empath_scores.append(np.nanmean(scores, axis=0))

    empath_scores = np.asarray(empath_scores)
    reddit_user_list['anger'] = empath_scores[:, 0]
    reddit_user_list['hate'] = empath_scores[:, 1]
    reddit_user_list['nervousness'] = empath_scores[:, 2]
    reddit_user_list['positive_emotion'] = empath_scores[:, 3]
    return reddit_user_list
def compute_morality_scores(reddit_user_list):

    DICT_TYPE = 'emfd'
    PROB_MAP = 'all'
    SCORE_METHOD = 'bow'
    OUT_METRICS = 'sentiment'
    OUT_CSV_PATH = 'all-vv.csv'
    template_input = pd.read_csv('template_input.csv', header=None)
    template_input.head()
    num_docs = len(template_input)

    morality_scores = []
    for index in range(len(reddit_user_list)):
        user = reddit_user_list.loc[index, "Author"]
        comments = retrieve_data_from_db.select_user_comments(user)
        comments= pd.DataFrame(data=comments)
        df = score_docs(comments, DICT_TYPE, PROB_MAP, SCORE_METHOD, OUT_METRICS, len(comments))
        morality_scores.append(list(np.mean(df.values, axis=0)))

    empath_scores = np.asarray(morality_scores)
    reddit_user_list['Care / Harm(R)'] = empath_scores[:, 0]
    reddit_user_list['Fairness / Cheating(R)'] = empath_scores[:, 1]
    reddit_user_list['Loyalty / Betrayal(R)'] = empath_scores[:, 2]
    reddit_user_list['Authority / Subversion(R)'] = empath_scores[:, 3]
    reddit_user_list['Sanctity / Degradation(R)'] = empath_scores[:, 4]

    return reddit_user_list


def get_toxicity_score(reddit_user_list):
    toxicity = []
    for index in range(len(reddit_user_list)):
        user = reddit_user_list.loc[index, "Author"]
        comments = retrieve_data_from_db.select_user_comments(user)
        comment_text = ""
        scores_tox = []
        for comment in comments:
            test_text = comment_text + ' ' + comment
            if len(test_text.encode('utf-8')) < 20000:
                comment_text = test_text
            else:
                try:
                    t = utils.toxicity_score(comment_text)
                    scores_tox.append(t)
                    comment_text = ""
                except HttpError as err:
                    print('HTTPError ', err)
                    if err.status_code == 429:
                        time.sleep(10)
                    elif err.status_code == 400:
                        comment_text = ""
                        pass
                    else:
                        comment_text = ""
                        pass
                except Exception as e:
                    print(e)
                    pass
        if comment_text != "":
            t = utils.toxicity_score(comment_text)
            scores_tox.append(t)
        toxicity.append(np.mean(scores_tox))
    reddit_user_list['toxicity'] = toxicity
    return reddit_user_list


def get_user_MBFC_scores(reddit_user_list):
    fact_all = []
    conspiracy_all = []
    bias_all = []

    with open('data/fact_data/MBFC_scores.txt') as f:
        data = f.read()

    my_dictionary = json.loads(data)
    for user_id in reddit_user_list.Author.values:
        avg_fact = []
        avg_cons = []
        avg_bias = []
        user_subs = retrieve_data_from_db.get_user_subreddits(user_id)
        for subreddit in user_subs:
            try:
                if my_dictionary.get(subreddit) is not None:
                    avg_fact.append(my_dictionary[subreddit][0])
                    avg_cons.append(my_dictionary[subreddit][1])
                    avg_bias.append(my_dictionary[subreddit][2])
                else:
                    print(subreddit)
                    a, b, c = utils.compute_subreddit_fact_score(subreddit)
                    my_dictionary[subreddit] = [a, b, c]
                    with open('MBFC_scores.txt', 'w') as file:
                        file.write(json.dumps(my_dictionary))
            except Exception as e:
                print(e)
                pass

        fact_all.append(np.nanmean(avg_fact))
        conspiracy_all.append(np.nanmean(avg_cons))
        bias_all.append(np.nanmean(avg_bias))

    reddit_user_list['avg_fact'] = fact_all
    reddit_user_list['avg_consp'] = conspiracy_all
    reddit_user_list['avg_bias'] = bias_all
    return reddit_user_list




def get_controversy_engagement(reddit_user_list):
    embeddings = pd.read_csv('data/embeddings/embedding-vectors.tsv', sep='\t', index_col=False, header=None)
    embeddings_meta = pd.read_csv('data/embeddings/embedding-metadata.tsv', sep='\t', index_col=False)
    embeddings[len(embeddings.columns)] = embeddings_meta.community.values
    banned_set = pd.read_csv('data/subreddits/subreddit_sorted.csv', index_col=False).Subreddit.values
    embedding_banneds = embeddings[embeddings.iloc[:, -1].isin(banned_set)]
    engagements = []
    for user in reddit_user_list.Author.values:
        sum_similarity = 0
        sum_count = 0
        subs = retrieve_data_from_db.get_user_subreddits(user)
        sub_counts = Counter(subs)
        for sub in sub_counts.keys():
            embedding_test = embeddings[embeddings.iloc[:, -1] == sub]
            if len(embedding_test) != 0:
                sub_similarity = utils.calculate_subreddit_similarity(embedding_test, embedding_banneds)
                sum_similarity = sum_similarity + (sub_similarity * sub_counts[sub])
                sum_count = sum_count + sub_counts[sub]
        print(np.divide(sum_similarity, sum_count))
        engagements.append(np.divide(sum_similarity, sum_count))
    reddit_user_list['engagement'] = engagements
    return reddit_user_list

