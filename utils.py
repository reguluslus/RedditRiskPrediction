from googleapiclient import discovery
import retrieve_data_from_db
import numpy as np
import pandas as pd
from sklearn import metrics
from urllib.parse import urlparse


def remove_quotes_(text):
    """Removes lines that begin with '>', indicating a Reddit quote."""
    lines = text.split('\n')
    nonquote_lines = [l for l in lines if not l.startswith('>')]
    text_with_no_quotes = '\n'.join(nonquote_lines).strip()
    return text_with_no_quotes or text


def toxicity_score(text):
    clean_text = remove_quotes_(text)
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
        return toxicity
    except:
        return np.nan, np.nan


def calculate_subreddit_similarity(test, banneds):
    sim = metrics.pairwise.cosine_similarity(banneds.iloc[:, 0:-1].values, [test.iloc[:, 0:-1].values[0]],
                                             dense_output=True)
    return np.max(sim)


def compute_subreddit_fact_score(subreddit_name):
    df = pd.read_csv('MBFC_data.csv', index_col=False)
    urls = retrieve_data_from_db.get_subreddit_urls(subreddit_name)
    misinformation_scores = []
    conspiracy_scores = []
    bias_scores = []

    for url in urls:
        try:
            domain = urlparse(url).netloc
            result = df.loc[df['Source'].str.contains(domain, na=False)]
            if len(result.Fact) == 1:
                misinformation_scores.append(int(result['Fact'].values[0]))
                conspiracy_scores.append(int(result['Conspiracy'].values[0]))
                if result['Bias'].values[0] in "'Other'":
                    bias_scores.append(np.nan)
                else:
                    bias_scores.append(int(result['Bias'].values[0]))
            elif len(result.Fact) > 1:
                for index, src in enumerate(result.Source):
                    if src in url:
                        misinformation_scores.append(int(result['Fact'].values[index]))
                        conspiracy_scores.append(int(result['Conspiracy'].values[index]))
                        if result['Bias'].values[0] in "'Other'":
                            bias_scores.append(np.nan)
                        else:
                            bias_scores.append(int(result['Bias'].values[index]))
        except Exception as e:
            print(e)
            pass
    if len(misinformation_scores) < 10:
        return np.nan, np.nan, np.nan
    else:
        return np.nanmean(misinformation_scores), np.nanmean(conspiracy_scores), np.nanmean(bias_scores)
