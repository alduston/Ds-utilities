
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numbers
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Perceptron
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import VotingRegressor
from sklearn import neighbors
import scipy.stats as stats
from scipy.stats import boxcox
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import math
import random
import shutil
from sklearn import tree
import warnings
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore", category=RuntimeWarning)




def without(big, little):
    result = []

    # loop through big list, return stuff that isnt in smaller list

    for item in big:
        if item not in little:
            result.append(item)

    # need this for dealing with technichal error with results of len 1

    if len(result) == 1:

        return result[0]
    else:
        return result




def labelled(df, label_name):
    # get number of rows in df

    l = len(df)
    label_col = []

    # loop over rows append label name to list for each row,

    for k in range(l):
        label_col.append(label_name)

    # list needs to become array before we add it as column

    label_col = np.asarray(label_col)

    # Add as column to df with column name 'label'

    df['label'] = label_col

    return df






def preprocess(train, test):
    # getting testing and training column list
    all_columns = train.columns
    predictor_col = test.columns

    # getting the y or ouptut column using "without" function

    ycol = without(all_columns, predictor_col)

    # seperating train into x and y values using .drop() method
    X_train = train.drop([ycol], axis=1)
    y = train[ycol]

    # Adding label column labelling data as test or train

    X_train_labelled = labelled(X_train, "train")
    X_test_labelled = labelled(test, "test")

    # combining all X data into one df

    X = pd.concat([X_train_labelled, X_test_labelled])

    # returning X and y in list, as we want them

    result = [X, y]

    return result





def na_count(col_data):
    # Gets list of all na values in  array

    na_list = pd.isna(col_data)

    # Gets proportion of NA by dividing by total lenght  of array

    result = sum(na_list) / len(col_data)

    # return proportion

    return result







def na_distribution(df, plot=False, cutoff=.25):
    result = []
    nums = []
    columns = df.columns

    # lopping through all columns, recoding columns name and proportion of na values

    for column in columns:
        col_data = df[column]

        num = na_count(col_data)
        nums.append(num)

        info = [column, na_count(col_data)]
        result.append(info)

    # sorting list

    result.sort()
    nums.sort()

    # if given plot argument of true, plots na proportion per column,
    # with line at cutoff value for reference in descending order

    if plot:
        plt.plot(nums)
        plt.axhline(y=cutoff, color='r')
        plt.show()

    # return list of na proportion per column

    return result






def na_remove(df, cutoff=.25):
    # first lets get proportions for each column using na_distribution

    bad_columns = []
    na_dist = na_distribution(df)

    # loops through columns, getting list of "bad columns" whose porportion of na
    # is above acceptable threshold, default is .25

    for data in na_dist:
        if data[1] > cutoff:
            bad_columns.append(data[0])

    # loops through bad column list, dropping them from our inputed df

    for bad_column in bad_columns:
        df = df.drop([bad_column], axis=1)

    # returns this edited dataframe

    return df






def is_num(column, cutoff=7):
    alpha = False
    beta = False

    l = len(column)
    num_num = 0

    # we are using numbers module to check if stuff in column is number

    Number = numbers.Number

    # looping over column keeping count of how many entries are numerical

    for item in column:

        if isinstance(item, Number):
            num_num += 1

    # get proportion of numerical items, if proportion is greater than .85 we call this a numerical column
    # reason we arent insisting on 100% is that some values may have been entered as string on accident like one hundred
    # insted of 100, this gives us flexibility to deal with this.

    num_prop = num_num / l

    if num_prop > .85:
        alpha = True

    levels = set(column)
    num_levels = len(levels)

    if num_levels > cutoff:
        beta = True

    # return true if proportion of numerical items is greater than .85 and more than 7 levels

    if alpha and beta:

        return True

    else:
        return False







def numcat_split(df):
    # setting up empty data frames to be filled and getting column list

    numerical = pd.DataFrame()
    categorical = pd.DataFrame()
    columns = df.columns

    # lopping over column, adding numerical to numerical df, else to categorical

    for column in columns:

        col_data = df[column]

        if is_num(col_data):
            numerical[column] = col_data

        else:

            categorical[column] = col_data

    # return list containing 2 new dataframes

    return [numerical, categorical]






def nan_to_mean(column):
    # get version of column with no na values and get its mean

    nona = column[~np.isnan(column)]
    mean = np.mean(nona)

    # loops through column, replacing nan with mean

    result = []
    for item in column:
        if np.isnan(item):
            result.append(mean)
        else:
            result.append(item)

    # return new column

    result = np.asarray(result)

    return result






def mean_df(df):
    # creating blank df to build up with converted columns, and get column list

    result = pd.DataFrame()
    columns = df.columns

    # looping over columns converting with nan_to_mean and adding to our result df
    for column in columns:
        col_data = df[column]
        mean_data = nan_to_mean(col_data)

        result[column] = mean_data

    # return df of converted columns

    return result





def prop_0(column):

    l = len(column)

    zero_count = 0

    for item in column:
        if item == 0:
            zero_count += 1

    prop = zero_count / l
    return prop







def df_prop_0(df, plot=False):
    columns = df.columns

    result = []
    props = []

    for column in columns:
        col_data = df[column]

        prop = prop_0(col_data)
        report = [column, prop]

        props.append(prop)
        result.append(report)

    if plot:
        props.sort()
        plt.plot(props)
        plt.show()

    return result







def cat_from_zero(column):
    cat_column = []

    for item in column:

        if item != 0:
            cat_column.append(1)

        else:

            cat_column.append(item)

    cat_column = np.asarray(cat_column)
    return cat_column






def zero_to_cat(df, cutoff=.4):
    props = df_prop_0(df)
    add_cat = []

    for item in props:
        if item[1] > cutoff:
            add_cat.append(item[0])

    for column in add_cat:
        col_name = column + "cat"

        col_data = df[column]
        col_data = cat_from_zero(col_data)

        df[col_name] = col_data

    return df





def clean_cat_col(column):
    clean_col = []

    # loop throgh column change "nan" to "none" and leave other values unchanged

    for item in column:

        if str(item) == "nan":
            clean_col.append("none")


        else:
            clean_col.append(item)

    clean_col = np.asarray(clean_col)

    return clean_col







def clean_cat_df(df):
    # set up empty output df and get column list

    clean_df = pd.DataFrame()
    columns = df.columns

    # loop through columns, converting each with clean_cat_col

    for column in columns:
        col_data = df[column]
        clean_df[column] = clean_cat_col(col_data)

    return clean_df






def column_namer(name, len):
    col_names = []

    for i in range(len):
        col_name = name + str(i)
        col_names.append(col_name)

    return col_names






def get_dummydf(df):
    # first make sure our input df is clean of na's with clean_cat_df, and get column list
    clean_df = clean_cat_df(df)
    columns = clean_df.columns

    # create empty ouput df for us to fill with dummy vars

    dummy_df = pd.DataFrame()

    # loop through columns, getting dummy columns with pd.pd.get_dummies()
    for column in columns:

        col_data = clean_df[column]
        dummies = pd.get_dummies(col_data)
        columns = dummies.columns

        l = len(columns)

        # for each column get names for dummie columns with column_namer, set 0 as start index

        names = column_namer(column, l)
        index = 0

        # give dummies correct names and add them to output df
        for column in columns:
            data = dummies[column].values
            name = names[index]

            dummy_df[name] = data
            index = index + 1

    # return output df made up of dummy columns

    return dummy_df






def df_proccesed0(train, test, na_cut=.25, zcat=True):

    X, y = preprocess(train, test)


    X = na_remove(X, na_cut)

    X_num, X_cat = numcat_split(X)


    X_num = mean_df(X_num)

    if zcat:
        X_num = zero_to_cat(X_num)


    X_cat = get_dummydf(X_cat)


    result = pd.concat([X_num, X_cat], axis=1)

    train = sum(result['label1'])

    X_train = result[:train]
    X_test = result[train:]

    result = [X_train, X_test, y]

    return result





def get_predictions0(data, model=LinearRegression(), ytran=True):
    # data is going to be output to our df_proccesed function, this outputs [X_train, X_test, y]

    X_train, X_test, y = data

    if ytran:
        y = np.log(y)

    # fitting model to our training data
    model.fit(X_train, y)

    # getting predictions for our test data
    predictions = model.predict(X_test)

    if ytran:
        predictions = np.exp(predictions)

    # code below just gets predictions into acceptable format for submission
    start_index = len(X_train) + 1
    stop_index = len(X_train) + len(X_test) + 1
    Predictions = pd.DataFrame(predictions, columns=["SalePrice"], index=range(start_index, stop_index))
    Predictions.index.name = "Id"

    # return predictions in acceptable format

    return Predictions



def k_fold_crossval(data, model=LinearRegression(), k=4, printout=True):
    # set X_train as our X data and y as y,
    # X_test not used since cross vallidation requires y values for all entries

    X, y = data[0], data[2]

    # Get array of scores from k fold cross validation, note this has strange error for k>4
    # it may be some kind of overflow error, I recommend just using the k=4 default arg

    scores = cross_val_score(model, X, y, cv=k)

    mean = scores.mean()
    std = scores.std()

    low = mean - (2 * std)
    high = mean + (2 * std)

    result = [low, mean, high]

    if printout:
        print("score is", mean, "+/-", (2 * std), )

        print('range is', [low, high])

    return result




def feature_score(data, feature, model=LinearRegression(), k=4):
    X = (data[0])[feature]
    X = X.values.reshape(-1, 1)

    y = data[2]
    y = y.values.reshape(-1, 1)

    scores = cross_val_score(model, X, y, cv=k)
    mean = scores.mean()

    return mean




def rank_features(data,model, plot=True, plotrange=15):

    data=data_copy(data)
    X, y = data[0], data[2]

    columns = X.columns

    result = []
    scores = []
    for feature in columns:

        score = feature_score(data, feature,model)

        scores.append(score)

        report = [score, feature]
        result.append(report)
        result.sort(reverse=True)

    if plot:
        scores = np.asarray(scores)
        indices = np.argsort(scores)[::-1][0:plotrange]

        plt.figure()
        plt.title("Feature importances")
        plt.barh(range(plotrange), scores[indices][::-1],
                 color="r", align="center")
        plt.yticks(range(plotrange), columns[indices][::-1])

        plt.xlim([0, 1])
        plt.show()

    return result






def drop_neg(data, model=LinearRegression()):

    X_train, X_test, y = data_copy(data)

    ranks = rank_features(data, model, plot=False)

    for item in ranks:
        if item[0] < 0:
            column = item[1]
            X_train = X_train.drop([column], axis=1)
            X_test = X_test.drop([column], axis=1)

    data = [X_train, X_test, y]

    return data


def data_copy(data):
    X_train, X_test, y = data

    X_train_copy = pd.DataFrame.copy(X_train, deep=True)
    X_test_copy = pd.DataFrame.copy(X_test, deep=True)

    return [X_train_copy, X_test_copy, y]



def normal_col(col):
    mean=col.mean()
    std=col.std()

    result=(col-mean)/std

    return result



def makepos(column):


    colmin=min(column)
    if colmin>0:
        factor=0
    else:
        factor=abs(colmin)+1

    result=column+ factor

    return result




def log_feature(data, feature):

    data_temp = data_copy(data)

    X_train, X_test, y = data_temp

    X_train[feature] = makepos(X_train[feature])
    X_test[feature] = makepos(X_test[feature])

    X_train[feature] = np.log(X_train[feature])
    X_test[feature] = np.log(X_test[feature])

    result = [X_train, X_test, y]

    return result




def normalize_feature(data, feature):

    data_temp = data_copy(data)

    X_train, X_test, y = data_temp

    train_copy=pd.DataFrame.copy(X_train, deep=True)
    test_copy=pd.DataFrame.copy(X_test, deep=True)

    X=pd.concat([train_copy,test_copy])


    mean=X.mean()
    std=X.std()

    train_normal=[]
    test_normal=[]

    for item in train_copy[feature]:
        normal_item=(item-mean)/std
        train_normal.append(normal_item)

    for item in test_copy[feature]:
        normal_item=(item-mean)/std
        test_normal.append(normal_item)

    train_normal=np.asarray(train_normal)
    test_normal=np.asarray(test_normal)

    train_copy[feature]=train_normal
    test_copy[feature]=test_normal

    result=[train_copy,test_copy,y]

    return result





def bcox_feature(data,feature):

    data_temp = data_copy(data)

    X_train, X_test, y = data_temp

    train_copy = pd.DataFrame.copy(X_train, deep=True)
    test_copy = pd.DataFrame.copy(X_test, deep=True)

    train_copy[feature]=makepos(train_copy[feature])
    test_copy[feature]=makepos(test_copy[feature])

    train_bcox = boxcox(train_copy[feature])[0]
    test_bcox = boxcox(test_copy[feature])[0]




    train_copy[feature] = train_bcox
    test_copy[feature] = test_bcox

    result = [train_copy, test_copy, y]

    return result







def get_prob_plot(data, feature, transform=False):

    if transform:
        data = log_feature(data, feature)
        X = pd.concat([data[0], data[1]])
        res = stats.probplot(X[feature], plot=plt)

    else:
        X = pd.concat([data[0], data[1]])
        res = stats.probplot(X[feature], plot=plt)








def df_proccesed1(train, test, na_cut=.25, ytran=True, drop=True):
    # getting X and y from train, test using preprocessed

    X, y = preprocess(train, test)

    if ytran:
        y = np.log(y)

    # Removing na columns and splitting X into numerical and categorical

    X = na_remove(X, na_cut)

    X_num, X_cat = numcat_split(X)

    # cleaning numericals with mean_df  getting categorical dumm
    X_num = mean_df(X_num)

    # getting categorical dummiesd with get_dummy_df

    X_cat = get_dummydf(X_cat)

    # combining 2 together into one df with all features

    result = pd.concat([X_num, X_cat], axis=1)

    # finding how many train values there are, using label column we created earlier with labelled function

    train = sum(result['label1'])

    # splitting X back into train and test rows usin
    X_train = result[:train]
    X_test = result[train:]

    # returning all data we need to feed model

    result = [X_train, X_test, y]

    if drop:
        result = drop_neg(result)

    return result







def get_predictions1(data, model=LinearRegression(), ytran=False):
    # data is going to be output to our df_proccesed function, this outputs [X_train, X_test, y]

    X_train, X_test, y = data

    # fitting model to our training data
    model.fit(X_train, y)

    # getting predictions for our test data
    predictions = model.predict(X_test)

    if ytran:
        predictions = np.exp(predictions)

    # code below just gets predictions into acceptable format for submission
    start_index = len(X_train) + 1
    stop_index = len(X_train) + len(X_test) + 1
    Predictions = pd.DataFrame(predictions, columns=["SalePrice"], index=range(start_index, stop_index))
    Predictions.index.name = "Id"

    # return predictions in acceptable format

    return Predictions






def normal_transformer(data, cut_val=0.005, early_cut=40):

    X_train, X_test, y = data

    features = X_train.columns
    to_normalize = []

    for feature in features[0:early_cut]:

        data_new = normalize_feature(data, feature)

        val1 = k_fold_crossval(data, printout=False)

        val2 = k_fold_crossval(data_new, printout=False)

        diff = val2[1] - val1[1]

        if (diff > cut_val):


            to_normalize.append(feature)

    for feature in to_normalize:
        data = normalize_feature(data, feature)


    return data


def log_transformer(data, cut_val=0.005, early_cut=40):
    X_train, X_test, y = data

    features = X_train.columns
    to_transform = []

    for feature in features[0:early_cut]:

        data_new = log_feature(data, feature)

        val1 = k_fold_crossval(data, printout=False)

        val2 = k_fold_crossval(data_new, printout=False)

        diff = val2[1] - val1[1]

        if (diff > cut_val):
            to_transform.append(feature)

    for feature in to_transform:
        data = log_feature(data, feature)

    return data





def bcox_transformer(data, cut_val=0.0001, early_cut=40):
    X_train, X_test, y = data

    features = X_train.columns
    to_transform = []

    for feature in features[0:early_cut]:

        data_new = bcox_feature(data, feature)

        val1 = k_fold_crossval(data, printout=False)

        val2 = k_fold_crossval(data_new, printout=False)

        diff = val2[1] - val1[1]


        if (diff > cut_val):
            to_transform.append(feature)

    for feature in to_transform:
        data = log_feature(data, feature)

    return data






def poly_feature(data, feature1, feature2, centered=False):

    data_temp = data[:]
    X_train, X_test, y = data_temp

    train_col1 = X_train[feature1]
    train_col2 = X_train[feature2]

    test_col1 = X_test[feature1]
    test_col2 = X_test[feature2]

    train_poly_col = np.multiply(train_col1, train_col2)

    test_poly_col = np.multiply(test_col1, test_col2)

    if centered:
        mean_train = train_poly_col.mean()
        train_poly_col = train_poly_col - mean_train

        mean_test = test_poly_col.mean()
        test_poly_col = test_poly_col - mean_test

    poly_name = feature1 + 'x' + feature2

    X_train[poly_name] = train_poly_col

    X_test[poly_name] = test_poly_col

    result = [X_train, X_test, y]

    return result





def best_poly_k(data, k=4, features_considered=8, reresult=False, centered=False, printout=True):
    ranks = rank_features(data, plot=False)[0:features_considered]

    poly_added = []
    result = []

    for i in range(k):

        best_score = 0

        for rank1 in ranks:
            feature1 = rank1[1]

            for rank2 in ranks:

                feature2 = rank2[1]

                features = [feature1, feature2]

                data_temp = data_copy(data)

                data_temp = poly_feature(data_temp, feature1, feature2, centered=centered)

                score = k_fold_crossval(data_temp, printout=False)[1]

                if (score > best_score) and (features not in poly_added):
                    best_score = score
                    bestf_1 = feature1
                    bestf_2 = feature2

                    report = [best_score, [bestf_1, bestf_2]]

        data = poly_feature(data, bestf_1, bestf_2)

        if printout:
            print(f'report:, {report}')

        poly_added.append([bestf_1, bestf_2])
        poly_added.append([bestf_2, bestf_1])
        result.append(report)

    if reresult:

        return [data, result]

    else:

        return data



def add_poly(data,polylist):

    for item in polylist:

        feature1=item[0]
        feature2 =item[1]

        data=poly_feature(data,feature1,feature2)

    return data




def y_to_cat(data,groupnum=50):

    data_temp = data_copy(data)

    X_train, X_test, y = data_temp

    ymin=min(y)
    ymax=max(y)

    spanned=ymax-ymin
    step=spanned/groupnum

    bins=[]
    bincut=ymin

    for k in range(groupnum):

        bins.append(bincut)
        bincut+=step
    bins.append(bincut)

    y_cat=[]
    for val in y:
        for item in bins:

            if val>item:
                catval=bins.index(item)

        y_cat.append(catval)

    y_cat=np.asarray(y_cat)

    result=[X_train, X_test,y_cat]

    return result




def getsplits(l,splits):

    splitsize=int(l/splits)

    splitlist=[]

    high=0

    for k in range(splits):

        low=high
        high=low+splitsize
        splitlist.append([low,high])


    last_split=splitlist[-1]
    splitlist[-1] = [last_split[0], l]

    return splitlist




def meta_tt_split(X,y,split):

    X=pd.DataFrame.copy(X)
    y=y[:]

    l=len(X)

    X_test=X[split[0]:split[1]]
    y_test=y[split[0]:split[1]]

    X_train_l=X[0:split[0]]
    X_train_h= X[split[1]:l]
    X_train=pd.concat([X_train_l,X_train_h])

    y_train_l = list(y[0:split[0]])
    y_train_h = list(y[split[1]:l])
    y_train=y_train_l+y_train_h

    y_train=np.asarray(y_train)
    result=[X_train,X_test,y_train,y_test]

    return result


def diff_eval(diff,X_train,y_train,X_test,groupnum=50):

    data=data_copy([X_train,X_test,y_train])

    if not diff[0]:
        return 0

    else:
        model_info=diff[1]

        model =model_info[0]
        params =model_info[1]
        is_cont=model_info[2]

        model = model(**params)

        if is_cont:
            data=data

        else:
            data=y_to_cat(data,groupnum=groupnum)

        X_train, X_test, y_train=data

        model.fit(X_train,y_train)

        predict=model.predict(X_test)
        predict=normal_col(predict)

        return predict





def get_meta_features(data, base_models, groupnum=50,splits=4,diff=[False,[]],stacked=False,average=True):

    data=data_copy(data)
    l=len(data[0])
    split_list=getsplits(l,splits)

    names=[]
    feature_data=[]
    test_feature_data=[]

    if stacked:
        X,X_test_true,y = data




    for base_model in base_models:

        name=str(base_model)
        names.append(name)

        model = base_model[0]
        params = base_model[1]
        is_cont = base_model[2]

        model = model(**params)

        if is_cont:
            data_temp=data_copy(data)

        else:
            data_temp= y_to_cat(data,groupnum=groupnum)

        if not stacked:
            X_test_true = data_temp[1]

        model.fit(data_temp[0],data_temp[2])


        test_predictions=model.predict(X_test_true)
        test_predictions=normal_col(test_predictions)

        test_predictions=test_predictions-diff_eval(diff,data_temp[0],data_temp[2],X_test_true,groupnum=groupnum)
        test_feature_data.append(test_predictions)

        predictions=[]

        for split in  split_list:

            if not stacked:
                X, y = data_temp[0], data_temp[2]

            X_train,X_test,y_train,y_test=meta_tt_split(X,y,split)

            model.fit( X_train,y_train)

            prediction=model.predict(X_test)
            prediction=normal_col( prediction)

            prediction=prediction-diff_eval(diff,X_train,y_train,X_test,groupnum=groupnum)
            predictions=predictions+list(prediction)

        if stacked:
            X[name]=predictions
            X_test_true[name]=test_predictions




        feature_data.append(predictions)

    if stacked:
        X_train = X
        result = [X_train, X_test_true, y]
        return result

    l_train = len(X)
    l_test = len(X_test_true)

    average_train=np.zeros(l_train)
    average_test=np.zeros(l_test)



    for k in range(len(feature_data)):


        name=names[k]
        train_data=feature_data[k]
        test_data=test_feature_data[k]

        X[name]=train_data
        X_test_true[name]= test_data

        if average:
            average_train=average_train+train_data
            average_test = average_test + test_data


    if average:
        X["average"]=average_train
        X_test_true["average"]=average_test




    result=[X, X_test_true,y]

    return result







def getshape(data,printout=True):

    X_train, X_test, y = data

    train_shape=X_train.shape
    test_shape = X_test.shape
    y_shape=y.shape

    if printout:
        print(' ')
        print('X_train shape:',train_shape)
        print('X_test shape:', test_shape)
        print('y shape:', y_shape)
        print(' ')

    result=[ train_shape, test_shape, y_shape]

    return result




def meta_validate(meta_data,main_model,splits=4,printout=True):

    model = main_model[0]
    params = main_model[1]

    data=data_copy(meta_data)

    X, y = data[0],data[2]

    l=len(X)

    split_list = getsplits(l, splits)
    scores=[]

    model = model(**params)
    for split in split_list:


        X_train, X_test, y_train, y_test = meta_tt_split(X, y, split)

        model.fit(X_train,y_train)
        score=model.score(X_test,y_test)
        scores.append(score)

    scores=np.asarray(scores)

    mean = scores.mean()
    std = scores.std()

    low = mean - (2 * std)
    high = mean + (2 * std)

    result = [low, mean, high]

    if printout:
        print("score is", mean, "+/-", (2 * std), )

        print('range is', [low, high])

    return result





def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name





def save_df(df):
    name=str(get_df_name(df))+'.csv'
    df.to_csv(name)




def voting_predictions(data,base_models,val=True):

    data=data_copy(data)
    Xtrain,Xtest,y=data

    index=0
    vote_params=[]
    for base_model in base_models:
        name='model'+str(index)
        index+=1
        model=base_model[0]
        params=base_model[1]
        model=model(**params)
        result=(name,model)
        vote_params.append(result)

    votemodel=VotingRegressor(vote_params)
    votemodel.fit( Xtrain,y)
    y_pred=votemodel.predict(Xtest)
    y_pred=np.exp(y_pred)

    if val:
        k_fold_crossval(data, model=votemodel)

    y_pred = np.exp(y_pred)

    return y_pred



def true_score(data,y_test,model=LinearRegression(),y_tran=True,printout=True,scorer=r2_score):

    data=data_copy(data)
    y_pred=get_predictions1(data, model=model, ytran=y_tran)["SalePrice"]

    score=scorer(y_test,y_pred)

    if printout:
        print("score on testing is ", score)

    return score






########################################################################################################################
#Testing:



ridge_info=[Ridge,{},True]

regtree_info=[DecisionTreeRegressor,{},True]

gamma_info=[GammaRegressor,{},True]

hubber_info=[HuberRegressor,{},True]

tree_info=[DecisionTreeClassifier,{},False]

lin_regression_info=[LinearRegression,{},True]

log_regression_info=[LogisticRegression,{'max_iter':2000},False]

lasso_info=[Lasso,{},True]

perceptron_info=[MLPRegressor,{'max_iter':5000},True]

bay_ridge_info=[BayesianRidge,{},True]

elastic_info=[ElasticNet,{},True]

omp_info=[OrthogonalMatchingPursuit,{},True]

ada_info=[AdaBoostRegressor,{'base_estimator':LinearRegression()},True]

svr_rbf_info = [SVR,{'kernel':'rbf', 'C':100, 'gamma':0.1, 'epsilon':.1},True]
svr_lin_info = [SVR,{'kernel':'linear','C':100, 'gamma':'auto'},True]
svr_poly_info =[SVR,{'kernel':'poly', 'C':100, 'gamma':'auto', 'degree':3, 'epsilon':.1,'coef0':1},True]

gradient_boost_info=[GradientBoostingRegressor,{'n_estimators': 750,'max_depth': 3,
'min_samples_split': 5, 'learning_rate': 0.1,'loss': 'ls'},True]


sgd_params={'alpha':0.0001, 'epsilon':0.1, 'eta0':0.01,
            'fit_intercept':True,'l1_ratio':0.15, 'learning_rate':'invscaling',
            'loss':'squared_loss', 'penalty':'l2', 'power_t':0.25,
            'shuffle':False, 'verbose':0, 'warm_start':False}

sgd_info=[SGDRegressor,sgd_params,True]




data=df_proccesed1()





















