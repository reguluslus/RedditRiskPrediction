import random
import time
from googleapiclient import discovery
from googleapiclient.errors import HttpError
import json
import retrieve_data_from_db

import numpy as np


def remove_quotes(text):
    """Removes lines that begin with '>', indicating a Reddit quote."""
    lines = text.split('\n')
    nonquote_lines = [l for l in lines if not l.startswith('>')]
    text_with_no_quotes = '\n'.join(nonquote_lines).strip()
    return text_with_no_quotes or text


def toxicity_score_user(text):
    clean_text = remove_quotes(text)
    API_KEY = 'AIzaSyDb_4h37CwVG0JtRddk7peDQ13fHHxmZwo'

    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': clean_text},
        'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}}
    }
    try:
        response = client.comments().analyze(body=analyze_request).execute()
        attr_scores = response['attributeScores']
        toxicity = attr_scores['TOXICITY']['summaryScore']['value']
        sv_toxicity = attr_scores['SEVERE_TOXICITY']['summaryScore']['value']
        return toxicity, sv_toxicity
    except:
        return np.nan, np.nan


def average_toxicity_score_subreddit(subreddit_name):
    comments = retrieve_data_from_db.get_subreddit_comments(subreddit_name)
    if len(comments) > 10000:
        comments = random.sample(comments, 10000)
    scores = []
    comment_text = ""
    for comment in comments:
        if len(comment_text.encode('utf-8')) < 20000:
            comment_text = comment_text + ' ' + comment
        else:
            comment_text = ""
            try:
                a, b = toxicity_score_user(comment)
                scores.append([a, b])
            except HttpError as err:
                print(err)
                if err.status_code == 429:
                    time.sleep(10)
                else:
                    pass
            except Exception as e:
                print(e)
                pass
    return np.nanmean(scores, axis=0)


with open('../data/subreddits/subreddit_list_all.txt') as f:
    user_subs = f.read()

user_subs = user_subs.split(', ')
avg_tox = []

with open('../data/subreddits/toxicity_scores.txt') as f:
    data = f.read()

my_dictionary = json.loads(data)

for subreddit in user_subs:
    try:
        if my_dictionary.get(subreddit) is None:
            print(subreddit)
            score_tox = average_toxicity_score_subreddit(subreddit)
            print(score_tox)
            my_dictionary[subreddit] = score_tox.tolist()
            with open('../data/subreddits/toxicity_scores.txt', 'w') as file:
                file.write(json.dumps(my_dictionary))
        avg_tox.append(my_dictionary[subreddit])
    except Exception as e:
        print(e)
        pass
