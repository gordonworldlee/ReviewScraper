from bs4 import BeautifulSoup
import requests
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

html_link = 'https://www.rottentomatoes.com/m/fantastic_four_2015/reviews'
html_text = requests.get(html_link).text
soup = BeautifulSoup(html_text, 'lxml')

reviews = soup.find_all('div', class_ = 'review-row')
review_list = []
for review in reviews:
    name = review.find('a', class_ = 'display-name').text.strip()
    text = review.find('p', class_ = 'review-text').text
    tomato = review.find('score-icon-critic-deprecated')
    tomato = tomato['state']
    review_list.append([name, tomato, text])



#df = pd.DataFrame(review_list, columns=['Name', 'Rating', 'Review'])
#df.to_csv('books.csv')
df2 = pd.read_csv('books.csv')



sa = SentimentIntensityAnalyzer()
f = lambda title: sa.polarity_scores(title)['compound']
df2['compound'] = df2['Review'].apply(f)

graph = sns.barplot(data=df2, x ='Rating', y='compound')
graph.set_title('Compound scores of tomato reviews')
plt.show()