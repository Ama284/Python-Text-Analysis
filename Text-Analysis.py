
"""Text-Data Analysis.ipynb



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read dataset
comments=pd.read_csv('E:\Data Analysis Real Project\Text-Data Analysis/GBcomments.csv',error_bad_lines=False)

comments.head()

"""#### sentiment analysis of youtube_comments"""

#!pip install textblob

from textblob import TextBlob

TextBlob('Its more accurate to call it the M+ (1000) be..').sentiment.polarity

comments.isna().sum()

comments.dropna(inplace=True)

polarity=[] # list which will contain the polarity of the comments

for i in comments['comment_text']:
    polarity.append(TextBlob(i).sentiment.polarity)

comments['polarity']=polarity

comments.head(20)



"""#### Lets perform EDA for the Positve sentences"""

comments_positive=comments[comments['polarity']==1]

comments_positive.shape

comments_positive.head()

#!pip install wordcloud

from wordcloud import WordCloud,STOPWORDS

stopwords=set(STOPWORDS)

total_comments=' '.join(comments_positive['comment_text'])

wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments)

plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')



"""#### Its time to go for negative sentences"""

comments_negative=comments[comments['polarity']==-1]

total_comments=' '.join(comments_negative['comment_text'])

wordcloud=WordCloud(width=1000,height=500,stopwords=stopwords).generate(total_comments)

plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')





"""### Analysing Tags column,what are trending tags on youtube"""

videos=pd.read_csv('E:\Data Analysis Real Project\Text-Data Analysis/USvideos.csv',error_bad_lines=False)

videos.head()

tags_complete=' '.join(videos['tags'])

tags_complete

import re

tags=re.sub('[^a-zA-Z]',' ',tags_complete)

tags

tags=re.sub(' +',' ',tags)

wordcloud=WordCloud(width=1000,height=500,stopwords=set(STOPWORDS)).generate(tags)

plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')





"""#### Lets find out the relation among continuous variables
#### As quite obvious the number of likes have very strong relation with views
"""

sns.regplot(data=videos,x='views',y='likes')
plt.title('Regression plot for views & likes')



"""#### dislikes vs views Analysis"""

sns.regplot(data=videos,x='views',y='dislikes')
plt.title('Regression plot for views & dislikes')



"""#### Correlation matrix is the evidence of above analysis!"""

df_corr=videos[['views','likes','dislikes']]

df_corr.corr()

sns.heatmap(df_corr.corr(),annot=True)



"""### Analyse Emojis in comments"""

comments.head()

comments['comment_text'][1]

"""    Every emoji has a Unicode associated with it
     '\U0001F600' is a unicode for 😀
"""

print('\U0001F600')

!pip install emoji

import emoji

len(comments)

comment=comments['comment_text'][1]

[c for c in comment if c in emoji.UNICODE_EMOJI]

str=''
for i in comments['comment_text']:
    list=[c for c in i if c in emoji.UNICODE_EMOJI]
    for ele in list:
        str=str+ele

len(str)

str



"""    lets create a dictionary of having each emoji with its frequency as well"""

result={}
for i in set(str):
    result[i]=str.count(i)

result

"""    sort the emojis according to its count or frequency"""

result.items()

final={}
for key,value in sorted(result.items(),key =lambda item:item[1]):
    final[key]=value

final

## convert dictionary into list for this we have to unzip this dictionary
keys=[*final.keys()]

keys

values=[*final.values()]

values



df=pd.DataFrame({'chars':keys[-20:],'num':values[-20:]})

df

import plotly.graph_objs as go
from plotly.offline import iplot

trace=go.Bar(
x=df['chars'],
y=df['num']
)

iplot([trace])







