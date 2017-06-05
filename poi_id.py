# coding=utf-8
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict
#print data_dict
# 计算有多少个数据点 employee_num
employee_num = 0
poi_num = 0
no_poi = 0
for item in my_dataset:
    temp = my_dataset[item]
    employee_num += 1
    # 将所有的NaN值都处理成0
    for sub_item in temp:
        if temp[sub_item] == 'NaN':
            temp[sub_item] = 0
    # 计算poi和非poi的个数
    if temp['poi'] != False:
        poi_num += 1
    else:
        no_poi += 1
print 'employee_num',employee_num
print 'poi_num',poi_num
print 'no_poi',no_poi

# 打印出所有的特征
features_sum = []
for item1 in my_dataset:
    temp = my_dataset[item1]
    print temp
    for sub_item in temp:
        features_sum.append(sub_item)
        #print type(temp[sub_item])
    break
#print features_sum
print len(features_sum)

# 移除不需要的特征email_address，由于特征poi必须排在第一位，故调整一下poi的顺序如下：
features_sum.remove('email_address')
features_sum.remove('poi')
features_sum.insert(0, 'poi')
print features_sum
print len(features_sum)

'''
features_sum = ['poi', 'salary', 'to_messages', 'deferral_payments',
                'total_payments', 'exercised_stock_options', 'bonus',
                'restricted_stock', 'shared_receipt_with_poi',
                'restricted_stock_deferred', 'total_stock_value',
                'expenses', 'loan_advances', 'from_messages', 'other',
                'from_this_person_to_poi', 'director_fees', 'deferred_income',
                'long_term_incentive', 'from_poi_to_this_person']
'''

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


# 添加新特性：poi_email_percent
for item in my_dataset:
    temp = my_dataset[item]
    total_email = temp['to_messages'] + temp['from_messages']
    poi_email = temp["from_poi_to_this_person"] + temp["from_this_person_to_poi"]

    if total_email != 0:
        temp["poi_email_percent"] = float(poi_email)/total_email
    else:
        temp["poi_email_percent"] = 0.0
#print my_dataset
features_sum.append("poi_email_percent")
print features_sum
#print my_dataset


# 进行特征缩放
#for item in my_dataset:
for i in range(1, len(features_sum), 1):
    tmp = []
    for item in my_dataset:
        temp = my_dataset[item]
        tmp.append(temp[features_sum[i]])
    tmp_max = max(tmp)
    tmp_min = min(tmp)
    print tmp_max,tmp_min
    for item in my_dataset:
        temp = my_dataset[item][features_sum[i]]
        my_dataset[item][features_sum[i]] = float(temp - tmp_min)/(tmp_max-tmp_min)
#print my_dataset

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_sum, sort_keys = True)
labels, features = targetFeatureSplit(data)
#print labels
#print features

#用SelectKBest方法，选出最好的5个特征
from sklearn.feature_selection import SelectKBest
select = SelectKBest(k= 5)
select.fit_transform(features, labels)
select_score = select.scores_
sort_select = sorted(select_score, reverse=True)[:2]
print select_score
print sort_select

features_list_new = ['poi']
for i in range(len(select_score)):
    if select_score[i] in sort_select:
        features_list_new.append(features_sum[i+1])

print features_list_new

#获取features_list
features_list = features_list_new

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
#clf = MultinomialNB(alpha = 0.5, fit_prior=False)

from sklearn import svm
clf = svm.SVC(C=8, gamma=100)

#from sklearn import tree
#clf =tree.DecisionTreeClassifier(min_samples_split=20)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#print features_train
print labels_train
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)