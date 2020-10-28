import csv

import pandas
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, GenericUnivariateSelect
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from learning import dataHelper
from learning.ExpectedValLearner import ExpectedValLearner
from learning.FieldClassifier import FieldsClassifier
#from learning.NNlearn import NNLearner, get_parms
from learning.dataHelper import Stats, RANK_METHODS, get_queries_from_df, create_report_file, MajorityClassifier, \
    ValToClassMode
from learning.kmeansClassifier import KMeansClassifier
from learning.multipleBinaries import MultipleBinaryCls
#from learning.networks import Layer, TwoLayersNet
from learning.resultsAnalyzer import create_query_report_file, gen_metrics_comparison, gen_all_metrics_comparison
from learning.sklearner import SKLearner


def test_models(learners, queries, split, method, class2val):
    train_acc = {}
    test_acc = {}
    train_mae = {}
    test_mae = {}
    queries_stats = {}
    for test_query in queries:
        queries_stats[test_query] = {}
        for learner in learners:
            model_name = learner.model_name()
            queries_stats[test_query][model_name] = {}
            for rm in RANK_METHODS[method]:
                queries_stats[test_query][model_name][rm] = 0

    predictions = {}
    for learner in learners:
        model_name = learner.model_name()
        predictions[model_name] = {}
        print('\n \n' + model_name)
        train_acc[model_name] = {x: 0 for x in RANK_METHODS[method]}
        test_acc[model_name] = {x: 0 for x in RANK_METHODS[method]}
        train_mae[model_name] = {x: 0 for x in RANK_METHODS[method]}
        test_mae[model_name] = {x: 0 for x in RANK_METHODS[method]}
        num_queries = 0
        for test_query in queries:
            if 'dummy' in test_query:
                continue
            num_queries +=1
            if test_query == 'all.csv':
                continue
            data = dataHelper.get_data(queries, test_query, method)
            learner.learn(data)
            if split == dataHelper.Split.BY_QUERY:
                print(test_query)
                stats = dataHelper.test_query_set(method, data.train_queries, learner, class2val)
                for rm, stats in stats.items():
                    train_acc[model_name][rm] += stats.acc
                    train_mae[model_name][rm] += stats.mae
                stats = dataHelper.test_query_set(method, data.test_queries, learner, class2val)
                predictions[model_name][test_query] = stats['group'].predictions[test_query]
                for rm, stats in stats.items():
                    test_acc[model_name][rm] += stats.acc
                    test_mae[model_name][rm] += stats.mae
                    queries_stats[test_query][model_name][rm] += stats.mae
        for rm in RANK_METHODS[method]:
            train_acc[model_name][rm] /= num_queries#len(queries)
            test_acc[model_name][rm] /= num_queries#len(queries)
            train_mae[model_name][rm] /= num_queries#len(queries)
            test_mae[model_name][rm] /= num_queries#len(queries)

    for model in test_acc.keys():
        print ('model: '  + model + '\n')
        for rm in RANK_METHODS[method]:
            print (rm + '\n')
            print(' train acc: ' + str(train_acc[model][rm]) + ' test mae: ' + str(train_mae[model][rm]))
            print(' test acc: ' + str(test_acc[model][rm]) + ' test mae: ' + str(test_mae[model][rm]))
        print ('query stats:')
    for q in queries:
        q_stat = {}
        q_stat['query_name'] = q

        stats = ' '
        for model in test_acc.keys():
            q_stat[model] = queries_stats[q][model]['group']
            stats += model + ': '
            for rm in RANK_METHODS[method]:
                stats += str(queries_stats[q][model][rm]) + ', '
        return predictions




def group_all():

    #input_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael_sigal_Irit\\by_group'
    #feature_file = "rel_only_group_features_by_stance_citation_range_1"
    #feature_file = "group_features_by_stance_citation_range_1"
    #df = pd.read_csv(input_dir + '\\group_features_by_stance.csv')
    #df = pd.read_csv(input_dir + '\\group_features_by_stance_no_enum.csv')
    #feature_file = "group_features_by_stance_citation_range_only_clinical1"
    #feature_file = "group_features_by_stance_citation_range_only_rev1"

    #feature_file = "group_features_by_stance_citation_range_1_no_stance"
    #feature_file = "group_features_by_stance_citation_range_1_no_stance_no_rel"


    input_folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\'
    cls = 'all_equal_weights'
    input_dir = input_folder + cls +'\\by_group'
    feature_file = "group_features_by_stance"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    queries = get_queries_from_df(df)
    labels = {q:int(queries[q].label) for q in queries}
    #mc = MajorityClassifier(input_dir + '\\majority.csv')
    decisionTreeLearner1 = SKLearner(DecisionTreeClassifier(random_state=0))
    svcLearner = SKLearner(svm.SVC(gamma='scale'))
    #layers = [Layer(input=5, output=10), Layer(input=10, output=10)]
    #layers = [Layer(input=31, output=20), Layer(input=20, output=10)]
    layers = [Layer(input=39, output=20), Layer(input=20, output=10)]
    net = TwoLayersNet(layers)
    #params = get_parms(net)
    #nnlearner = NNLearner(dataHelper.Method.GROUP_ALL, net=net, params=params)
    learners = [ decisionTreeLearner1]
    #learners = [nnlearner]
    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL)
    reports_dir =input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    query_report_full_file_name =reports_dir+query_report_file_name
    #metrics_report_file_name =input_dir + '\\reports\\'+feature_file+'_metrics_report.csv'
    create_query_report_file(query_report_full_file_name, queries, learners, predictions, labels)
    files = ['google labels', 'majority', query_report_file_name]
    label_file =  input_folder + cls +'\\labels.csv'
    gen_all_metrics_comparison(folder=reports_dir,files= files,label_file=label_file)
    #gen_metrics_comparison(folder=reports_dir, query_filenames=files, label_file=label_file,
    #                       cmp_filename=feature_file+'google_maj_ML.csv')
#    create_report_files(query_report_file_name, metrics_report_file_name, queries=queries,
#                       learners = learners,predictions=predictions,labels=labels)



def get_md_labels(file):
    labels = {}
    with open(file, 'r', newline='') as queries_csv:
        reader = csv.DictReader(queries_csv)
        for row in reader:
            query = row['query']
            value = row['value_label']
            if value.isdigit():
                labels[query] = int(value)
    return labels



def ijcai():
    #input_folder = 'C:\\research\\falseMedicalClaims\\IJCAI\\model input\\ecai_new'
    input_folder = 'C:\\research\\falseMedicalClaims\\IJCAI\\model input\\non'
    #input_folder = 'C:\\research\\falseMedicalClaims\\IJCAI\\model input\\GTIC'
    input_dir = input_folder + '\\by_group'
    feature_file = "dummy_added_group_features_by_stance3"
    #feature_file = "dummy_added_group_features_by_stance_exp"
    #feature_file = "dist_exp"
    #feature_file = "dist_group_features_by_stance_paste_ecai2"
    feature_file = "dist"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    features = list(df.head())[2:]
    queries = get_queries_from_df(df)
    labels = {q:int(queries[q].label) for q in queries}
    decisionTreeLearner = SKLearner(DecisionTreeClassifier(random_state=0),features)
    gnb = SKLearner(GaussianNB())
    ecl = ExpectedValLearner()
    mult = MultipleBinaryCls(ValToClassMode.W_H)
    neigh = SKLearner(KNeighborsClassifier(n_neighbors=5))
    lr = SKLearner(LogisticRegression(C=1e5))
    kmeans = KMeansClassifier(input_dir+'\\dist.csv')

    #decisionforestLearner = SKLearner(RandomForestClassifier(random_state=0))
    learners = [mult]
    #learners = [decisionforestLearner]
    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL)
    reports_dir =input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    query_report_full_file_name =reports_dir+query_report_file_name
    create_query_report_file(query_report_full_file_name, queries, learners, predictions, labels)
    files = [ 'majority', query_report_file_name]
    gen_all_metrics_comparison(folder=reports_dir,files= files,actual_values=labels, cmp_filename=feature_file+'stats_report')

#'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\'
def w_h():
 #   input_folder = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\'
#    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\no outlayers\\'
#    feature_file = "group_features_by_stance_shrink_nol"
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features_all\\'
    feature_file = "group_features_by_stance"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    features = list(df.head())[2:]
    queries = get_queries_from_df(df)
    labels = {q:int(queries[q].label) for q in queries}
    decisionTreeLearner = SKLearner(DecisionTreeClassifier(random_state=0),features)
    gnb = SKLearner(GaussianNB())
    ecl = ExpectedValLearner()
    mult = MultipleBinaryCls(ValToClassMode.W_H)
    neigh = SKLearner(KNeighborsClassifier(n_neighbors=5))
    lr = SKLearner(LogisticRegression(C=1e5))
    kmeans = KMeansClassifier(input_dir+'\\dist.csv')

    decisionforestLearner = SKLearner(RandomForestClassifier(random_state=0))
    #learners = [kmeans, neigh]
    learners = [decisionforestLearner]

    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL, ValToClassMode.W_H)
    reports_dir =input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    query_report_full_file_name =reports_dir+query_report_file_name
    create_query_report_file(query_report_full_file_name, queries, learners, predictions, labels, ValToClassMode.W_H)
    files = [ 'majority', query_report_file_name]
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels, cmp_filename=feature_file+'stats_report', mode=ValToClassMode.W_H)

def w_h_report():
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\no outlayers\\'
    feature_file = "group_features_by_stance_shrink_nol"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
 #   feature_file = "group_features_by_stance_shrink"
    reports_dir = input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    queries = get_queries_from_df(df)
    labels = {q: int(queries[q].label) for q in queries}
    files = ['majority_nol', query_report_file_name]
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename=feature_file + 'stats_report', mode=ValToClassMode.W_H)


def main():
    pandas.set_option('display.max_rows', 50)
    pandas.set_option('display.max_columns', 50)
    pandas.set_option('display.width', 1000)  # Clerical work:
    #group_all()
    w_h()


if __name__ == '__main__':
    main()
