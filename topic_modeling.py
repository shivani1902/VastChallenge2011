from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

#Word count
file=open("Enter_CSV_Name","r")
wordcount={}
for word in file.read().split():
    if word not in wordcount:
        wordcount[word] = 1
    else:
        wordcount[word] += 1
print (word,wordcount)
file.close()

#TF-IDF
#documents = Pd.Read_csv
no_features = 1000
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

#Run NMF
no_topics = 20
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

#Display topics
no_top_words = 10
for topic_idx, topic in enumerate(nmf.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([tfidf_feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
