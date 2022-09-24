import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm, naive_bayes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
import time

# 设置Random seed，确保每次产生的随机数一致
np.random.seed(0)

# 读取语料
Corpus = pd.read_csv(r"trains_deal.csv", encoding='latin-1')
Target = pd.read_csv(r"tests_deal.csv", encoding='latin-1')

# 去空格
Corpus['text'].dropna(inplace=True)
Target['text'].dropna(inplace=True)

# 化为小写
Corpus['text'] = [entry.lower() for entry in Corpus['text']]
Target['text'] = [entry.lower() for entry in Target['text']]

# 分词
Corpus['text'] = [word_tokenize(entry) for entry in Corpus['text']]
Target['text'] = [word_tokenize(entry) for entry in Target['text']]

# 词性标注
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

# 词形还原
for index, entry in enumerate(Corpus['text']):
    Final_words1 = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words1.append(word_Final)
    Corpus.loc[index, 'text_final'] = str(Final_words1)

for index, entry in enumerate(Target['text']):
    Final_words2 = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words2.append(word_Final)
    Target.loc[index, 'text_final'] = str(Final_words2)
# print(Corpus['text_final'].head())

# 划分训练集和验证集
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['label'], test_size=0.1)

# 向量化
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# tf-idf权重
Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# print((Train_X_Tfidf,Train_Y))

# 计时
time1 = time.time()

# K近邻
KNN = KNeighborsClassifier()
KNN.fit(Train_X_Tfidf, Train_Y)
predictions_KNN = KNN.predict(Test_X_Tfidf)
print("KNN准确率为：", accuracy_score(predictions_KNN, Test_Y))

time2 = time.time()

# 支持向量机
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM准确率为：", accuracy_score(predictions_SVM, Test_Y))

time3 = time.time()

# 朴素贝叶斯
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes准确率为：", accuracy_score(predictions_NB, Test_Y))

time4 = time.time()

# 逻辑回归
Logistic = LogisticRegression(penalty='l2')
Logistic.fit(Train_X_Tfidf, Train_Y)
predictions_Logistic = Logistic.predict(Test_X_Tfidf)
print("Logistic Regression准确率为：", accuracy_score(predictions_Logistic, Test_Y))

time5 = time.time()

# 随机森林
Random = RandomForestClassifier(n_estimators=500)
Random.fit(Train_X_Tfidf, Train_Y)
predictions_Random = Random.predict(Test_X_Tfidf)
print("Random Forest准确率为：", accuracy_score(predictions_Random, Test_Y))

time6 = time.time()

# 决策树
Tree = tree.DecisionTreeClassifier()
Tree.fit(Train_X_Tfidf, Train_Y)
predictions_Tree = Tree.predict(Test_X_Tfidf)
print("Decision Tree准确率为：", accuracy_score(predictions_Tree, Test_Y))

time7 = time.time()

print('KNN训练和验证共用时(s)：', time2 - time1)
print('SVM训练和验证共用时(s)：', time3 - time2)
print('Naive Bayes训练和验证共用时(s)：', time4 - time3)
print('Logistic Regression训练和验证共用时(s)：', time5 - time4)
print('Random Forest训练和验证共用时(s)：', time6 - time5)
print('Decision Tree训练和验证共用时(s)：', time7 - time6)

# 根据验证集调参后预测测试集
Train_X, Test_X, Train_Y, Test_Y = (Corpus['text_final'], Target['text_final'], Corpus['label'], Target['label'])

# 向量化
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# tf-idf权重
Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
# print((Train_X_Tfidf, Train_Y))

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
# print(predictions_SVM)

with open('texts_prelabel.txt', 'w', encoding='utf-16', newline='') as f:
    for i in predictions_SVM:
        f.write(str(i) + '\n')
