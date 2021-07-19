import pandas as pd
import numpy as np
import seaborn as sns
from cleaned import preprocess_text

Contacts=pd.read_excel(r"C:\Users\Admin\BUSOFTTECH\Sample Data\Contacts.xlsx")
Incidents=pd.read_excel(r"C:\Users\Admin\BUSOFTTECH\Sample Data\Incidents.xlsx")
threads=pd.read_excel(r"C:\Users\Admin\BUSOFTTECH\Sample Data\Threads.xlsx")

header_row           =  0
threads.columns      =  threads.iloc[header_row]
Incidents.columns    =  Incidents.iloc[header_row]
Contacts             =  Contacts.iloc[header_row]
threads              =  threads.drop(header_row)
Incidents            =  Incidents.drop(header_row)
Contacts             =  Contacts.drop(header_row)

D1 = pd.merge(Contacts,Incidents, left_on = ['Contact ID'], right_on = ['Contact ID'])
final = pd.merge(D1,threads, left_on = ['Incident ID'], right_on = ['Foreign Key'])

final=final.drop(['Address Line 1', 'Address Line 2',
       'Address Line 3', 'City','Country', 'County','Product ID','Date Of Birth', 'Disabled Flag','Category ID','Contact Type','Date Created','Date Created_y','Email Header','Status','Status Type','Subject','Severity','Incident Thread ID','Account','Attributes'],axis=1)


column_titles = ['Contact ID','Assigned Account','First Name','Last Name','Email Address','Text']
final_df = final.reindex(columns=column_titles)
final_df=final_df.drop(['Contact ID','First Name','Last Name'],axis=1)

final_df['TEXT']=preprocess_text(final_df['TEXT'])

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
sentiment_score_list = []
sentiment_label_list = []

for i in df['TEXT'].values.tolist():
    sentiment_score = analyser.polarity_scores(i)

    if sentiment_score['compound'] >= 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Positive')
    elif sentiment_score['compound'] > -0.05 and sentiment_score['compound'] < 0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Neutral')
    elif sentiment_score['compound'] <= -0.05:
        sentiment_score_list.append(sentiment_score['compound'])
        sentiment_label_list.append('Negative')
final_df['sentiment'] = sentiment_label_list
final_df['sentiment score'] = sentiment_score_list

final_df['sentiment']=final_df['sentiment'].replace(["Negative","Positive","Neutral"],[0,1,2])

processedtext=np.array(df['TEXT'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(processedtext,final_df['sentiment'],test_size=0.30,random_state=2018)
from sklearn.feature_extraction.text import TfidfVectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

from imblearn.under_sampling import NearMiss
nm=NearMiss()
x_nm,y_nm=nm.fit_resample(X_train,y_train)

import xgboost
from xgboost import XGBClassifier
XGB=XGBClassifier()
XGB.fit(x_nm,y_nm)
pred=XGB.predict(X_test)

file = open('vectoriser12','wb')
pickle.dump(vectoriser', file)
file.close()
file = open('Sentiment-LR.pickle','wb')
pickle.dump(XGB, file)
file.close()


    
