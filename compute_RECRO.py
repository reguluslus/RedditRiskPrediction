import utils
from emfdscore.scoring import score_docs
from empath import Empath
import retrieve_data_from_db
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
                                              categories=["anger", "hate", "nervousness", "positive_emotion"])
                scores.append(list(result_dict.values()))
            except Exception as e:
                print(e)
                pass
        empath_scores.append(np.nanmean(scores, axis=0))

    empath_scores = np.asarray(empath_scores)
    reddit_user_list['Anger(R)'] = empath_scores[:, 0]
    reddit_user_list['Hate(R)'] = empath_scores[:, 1]
    reddit_user_list['Nervousness(R)'] = empath_scores[:, 2]
    reddit_user_list['Positive_emotion(R)'] = empath_scores[:, 3]
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
        comments = pd.DataFrame(data=comments)
        df = score_docs(comments, DICT_TYPE, PROB_MAP, SCORE_METHOD, OUT_METRICS, len(comments))
        morality_scores.append(list(np.mean(df.values, axis=0)))

    empath_scores = np.asarray(morality_scores)
    reddit_user_list['Care/Harm(R)'] = empath_scores[:, 0]
    reddit_user_list['Fairness/Cheating(R)'] = empath_scores[:, 1]
    reddit_user_list['Loyalty/Betrayal(R)'] = empath_scores[:, 2]
    reddit_user_list['Authority/Subversion(R)'] = empath_scores[:, 3]
    reddit_user_list['Sanctity/Degradation(R)'] = empath_scores[:, 4]

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

    reddit_user_list['Misinformation(E)'] = [i * (-1) for i in fact_all]
    reddit_user_list['Conspiracy(E)'] = conspiracy_all
    return reddit_user_list


def get_controversy_engagement(reddit_user_list):
    sub_tox_scores = []

    for user in reddit_user_list.Author.values:
        sub_tox_score = get_avg_toxicity_score(user)
        sub_tox_scores.append(sub_tox_score)
        print(sub_tox_score)

    reddit_user_list['Engagement(C)'] = sub_tox_scores
    return reddit_user_list


def get_avg_toxicity_score(user_id):
    sub_tox = []
    with open('../data/subreddits/toxicity_scores.txt') as f:
        data = f.read()

    my_dictionary = json.loads(data)
    user_subs = retrieve_data_from_db.get_user_subreddits(user_id)
    for subreddit in user_subs:
        try:
            if my_dictionary.get(subreddit) is not None:
                sub_tox.append(my_dictionary[subreddit][0])
        except Exception as e:
            print(e)
            pass

    return np.nanmean(sub_tox)

