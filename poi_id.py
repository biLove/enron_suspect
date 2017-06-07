# coding=utf-8
#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict
#print data_dict
# 计算有多少个数据点 employee_num



# 计算poi和非poi的个数
employee_num = len(my_dataset)
poi_num = 0
no_poi = 0
for item in my_dataset:
    temp = my_dataset[item]
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
    #print temp
    for sub_item in temp:
        features_sum.append(sub_item)
        #print type(temp[sub_item])
    break
print features_sum
print len(features_sum)

# 移除不需要的特征 email_address，由于特征poi必须排在第一位，故调整一下poi的顺序如下：
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


# 去掉缺失值超过50%的特征
lenthdata = len(my_dataset)
feature_remove = []
for i in range(len(features_sum)):
    tmp = []
    n = 0
    for item in my_dataset:
        sub = my_dataset[item]
        sub_key = sub[features_sum[i]]
        tmp.append(sub_key)
        if sub_key == 'NaN':
            n += 1
    if n > lenthdata * 0.5:
        #print features_sum[i]
        feature_remove.append(features_sum[i])
        #print tmp

features_new = []
for sub in features_sum:
    if sub not in feature_remove:
        features_new.append(sub)
print features_new
print len(features_new)

features_sum = features_new

# 从my_dataset 里面去掉上述特征
new_data = {}
for item in my_dataset:
    new_data[item] = {}
    temp = my_dataset[item]
    for sub_item in temp:
        if sub_item in features_sum:
            new_data[item][sub_item] = temp[sub_item]

#print my_dataset['METTS MARK']
#print new_data['METTS MARK']

my_dataset = new_data

### Task 2: Remove outliers

# 剔除异常值
individul_list = []
for item in my_dataset:
    individul_list.append(item)
    sub_list = []
    temp = my_dataset[item]
    n = 0
    for sub_item in temp:
        sub_list.append(temp[sub_item])
        if temp[sub_item] == 'NaN':
            n += 1
    if n > len(temp) * 0.8:
        print item
        #print item

#print individul_list
my_dataset.pop("TOTAL", None)
my_dataset.pop("THE TRAVEL AGENCY IN THE PARK", None)
my_dataset.pop("LOCKHART EUGENE E", None)




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


# 添加新特性：poi_email_percent
for item in my_dataset:
    temp = my_dataset[item]
    if temp['to_messages'] == 'NaN':
        temp['to_messages'] = 0
    if temp['from_messages'] == 'NaN':
        temp['from_messages'] = 0
    total_email = temp['to_messages'] + temp['from_messages']
    poi_email = temp["from_poi_to_this_person"] + temp["from_this_person_to_poi"]

    if total_email != 0:
        temp["poi_email_percent"] = float(poi_email)/total_email
    else:
        temp["poi_email_percent"] = 0.0
#print my_dataset
features_sum.append("poi_email_percent")
print features_sum


# 特征选择
# 由于 新特征 poi_email_percent，特征 from_this_person_to_poi 和 from_poi_to_this_person 使用了
#已知的poi信息，故不放在最终的特征集中，可将这3个特征从特征集中去掉。
features_sum.remove('poi_email_percent')
features_sum.remove('from_this_person_to_poi')
features_sum.remove('from_poi_to_this_person')

print features_sum


# 进行特征缩放
#for item in my_dataset:
for i in range(1, len(features_sum), 1):
    tmp = []
    for item in my_dataset:
        temp = my_dataset[item]
        sub_item = temp[features_sum[i]]
        if sub_item != 'NaN':
            tmp.append(sub_item)
    tmp_max = max(tmp)
    tmp_min = min(tmp)
    print tmp_max,tmp_min
    for item in my_dataset:
        temp = my_dataset[item][features_sum[i]]
        if temp != 'NaN':
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
sort_select = sorted(select_score, reverse=True)[:3]
print "select_score："
print select_score
print "sort_select:"
print sort_select

features_list_new = ['poi']
for i in range(len(select_score)):
    if select_score[i] in sort_select:
        features_list_new.append(features_sum[i+1])

print "select_features"
print features_list_new[1:]

#获取features_list
features_list = features_list_new

print features_list

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

'''
# SVMs使用 GradSearchCV 来调参
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import sklearn.metrics as metrics

# clf = SVC(C=1000, gamma=100)

tuned_parameters = [{'kernel': ['rbf'],'gamma': [10,100,1000],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10,100,1000]}]

clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1')
test_classifier(clf, my_dataset, features_list, folds=100)

# def scorer(estimator, X, y):
#     pred = estimator.predict(X)
#     precision = metrics.precision_score(y,pred)
#     recall = metrics.recall_score(y,pred)
#     return min(precision, recall)
# clf = GridSearchCV(SVC(), tuned_parameters, scoring=scorer)
'''

# # Decision Trees使用 GradSearchCV 调参
# from sklearn.model_selection import GridSearchCV
# from sklearn import tree
# # tuned_parameters = {'min_samples_split':[12]}
# # trees = tree.DecisionTreeClassifier()
# # clf = GridSearchCV(trees, tuned_parameters, scoring='f1')
# clf = tree.DecisionTreeClassifier(min_samples_split=12)
# test_classifier(clf, my_dataset, features_list, folds=100)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
