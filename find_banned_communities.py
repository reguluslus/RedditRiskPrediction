import pandas as pd
from bs4 import BeautifulSoup
from urllib.error import URLError, HTTPError
from urllib.request import urlopen


def extract_banned_reasons():
    url = "https://academictorrents.com/details/c398a571976c78d346c325bd75c47b82edf6124e/tech&filelist=1"
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    arr_url = []
    arr_reason = []
    for t in text.split():
        if t.startswith('subreddits/'):
            x = ((t.split('/')[1]).split('.')[0])
            if x.endswith('_comments'):
                subreddit_title = x.split('_comments')[0]
                url = "http://www.reddit.com/r/" + subreddit_title
                try:
                    page = urlopen(url)
                    html_bytes = page.read()
                    html = html_bytes.decode("utf-8")
                except URLError:
                    continue
                except HTTPError:
                    continue
                except TimeoutError:
                    continue
                if "ban-outline" in html:
                    print(url)
                    soup = BeautifulSoup(html, "html.parser")
                    text_sub = soup.get_text()
                    a = text_sub.split('\n')

                    for i in range(len(a)):
                        if ("banned" in a[i]) & ("r/" + subreddit_title not in a[i]):
                            arr_url.append(url)
                            arr_reason.append(a[i])
                            print(a[i])
    d = {'Subreddit': arr_url, 'Reason': arr_reason}
    df = pd.DataFrame(data=d)
    df.to_csv("subreddit_banned.csv", index=False)


def filter_banned_communities():
    with open('ban_reasons_filtered.txt') as f:
        reasons = f.read().splitlines()
    data = pd.read_csv("subreddit_banned.csv")
    data = data[data['Reason'].isin(reasons)]
    print(data.Subreddit.values)
    print(len(data.Subreddit.values))
    data.to_csv("subreddit_banned_filtered.csv", index=False)



