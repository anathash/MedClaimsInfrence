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
from learning.MultipleCls import MultipleCls
from learning.dataHelper import Stats, RANK_METHODS, get_queries_from_df, create_report_file, MajorityClassifier, \
    ValToClassMode, get_queries_from_pairs_df, Method, VAL_TO_CLASS_DICT
from learning.diffLearner import DiffLearnerCls
from learning.kmeansClassifier import KMeansClassifier
from learning.multipleBinaries import MultipleBinaryCls
#from learning.networks import Layer, TwoLayersNet
#from learning.networks import TwoLayersNet, Layer
from learning.resultsAnalyzer import create_query_report_file, gen_metrics_comparison, gen_all_metrics_comparison
from learning.sklearner import SKLearner


def test_models(learners, queries, split, method, val2class, labels=None):
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
            if method == Method.GROUP_ALL:
                data = dataHelper.get_data(queries, test_query, method)
                train_queries = data.train_queries
                test_queries = data.test_queries
                learner.learn(data)
            else:
                learner.learn(queries, test_query)
                train_queries = {x: y for x, y in queries.items() if x != test_query and test_query not in x}
                test_queries = {test_query: queries[test_query]}
            if split == dataHelper.Split.BY_QUERY:
                print(test_query)
                stats = dataHelper.test_query_set(method, train_queries, learner, val2class, labels)
                for rm, stats in stats.items():
                    train_acc[model_name][rm] += stats.acc
                    train_mae[model_name][rm] += stats.mae
                stats = dataHelper.test_query_set(method, test_queries, learner, val2class, labels)
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
            query = row['query'].strip()
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
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\'
    feature_file = "group_features_by_stance_nol"
#    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\posterior\\'
#    feature_file = "weighted_posterior_normed_ratio"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    features = list(df.head())[2:]
    queries = get_queries_from_df(df)
    labels = {q:int(queries[q].label) for q in queries}
    decisionTreeLearner = SKLearner(DecisionTreeClassifier(random_state=0),features)
    gnb = SKLearner(GaussianNB())
    ecl = ExpectedValLearner()
    mult = MultipleBinaryCls(ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    neigh = SKLearner(KNeighborsClassifier(n_neighbors=5))
    lr = SKLearner(LogisticRegression(C=1e5))
    kmeans = KMeansClassifier(input_dir+'\\dist.csv')

    decisionforestLearner = SKLearner(RandomForestClassifier(random_state=0))
    #learners = [kmeans, neigh]
    learners = [mult]

    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL, ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    reports_dir =input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    query_report_full_file_name =reports_dir+query_report_file_name
    create_query_report_file(query_report_full_file_name, queries, learners, predictions, labels, ValToClassMode.W_H)
    files = [ 'majority', query_report_file_name]
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels, cmp_filename=feature_file+'stats_report', mode=ValToClassMode.W_H)


def learn_by_pairs():
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\'
    #feature_file = "group_features_by_stance_nol_pessim"
    feature_file = "pairs_features_by_stance_stand"
    labels_file =  'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\labels.csv'
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    queries = get_queries_from_pairs_df(df)
    labels = get_md_labels(labels_file)

    cls = DiffLearnerCls(LinearRegression())
    learners = [cls]
    #learners = [decisionforestLearner, neigh]

    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.PAIRS_QUERY,  ValToClassMode.THREE_CLASSES_PESSIMISTIC, labels)
    reports_dir = input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    query_report_full_file_name = reports_dir + query_report_file_name
    create_query_report_file(query_report_full_file_name, input_dir, feature_file, queries, learners, predictions, labels, ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    files = ['majority', query_report_file_name]
    cmp_filename = feature_file + '_stats_report'
    gen_all_metrics_comparison(folder=reports_dir, files=files, actual_values=labels,
                               cmp_filename=cmp_filename)


def learn_by_doctors_annotations(val2class, feature_file, directory , resample, majority_filename, quick = False, filter_queries = None):
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\'
    output_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\'+directory+'\\'
   # feature_file = "group_features_by_stance_nol"
    majority_file = input_dir+ majority_filename
    reports_dir = output_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    query_report_full_file_name = reports_dir + query_report_file_name
    majority_report_file_name =  'majority_' + val2class.name + '.csv'
    majority_report_full_file_name = reports_dir + majority_report_file_name
    gen_majority_report(majority_file, majority_report_full_file_name, val2class)

    #feature_file = "group_features_by_label_shrink_nol"
    #feature_file = "group_features_by_stance_label_shrink_nol"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    queries = get_queries_from_df(df)
    labels = {q: int(queries[q].label) for q in queries}
    knn = KNeighborsClassifier(n_neighbors=5)
    neigh = SKLearner(knn, features = None,  resample = resample)
    #weights = {1: 4.0, 3: 1.5 , 5:1.0}
    weights= {1: 100, 3:25 , 5: 1 }
    svc = SKLearner(svm.SVC(class_weight = weights))
    rfc = RandomForestClassifier(random_state=0)
    decisionforestLearner = SKLearner(rfc, features = None,  resample = resample)
#    lr = SKLearner(LinearRegression(C=1e5))
    mult = MultipleBinaryCls(ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    learners = [MultipleCls(rfc=rfc,knn =knn, resample = resample), decisionforestLearner, neigh]
    if quick:
        for learner in learners:
            learner.quick_learn(queries)

    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL, val2class)
    create_query_report_file(query_report_full_file_name, input_dir, feature_file, queries, learners, predictions, labels, val2class)
    files = [majority_report_file_name, query_report_file_name]
    if resample:
        cmp_filename = feature_file + '_stats_report_resample'
    else:
        cmp_filename = feature_file + '_stats_report'
    gen_all_metrics_comparison(folder=reports_dir, files=files, actual_values=labels,
                               cmp_filename=cmp_filename, val2class=val2class)
    if filter_queries:
        gen_all_metrics_comparison(folder=reports_dir, files=files, actual_values=labels,
                                   cmp_filename=cmp_filename, val2class=val2class, filter_queries=filter_queries)


def w_h_report():
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\mult_features\\'
    feature_file = "h_index_stance_label_shrink_neg"
    feature_file = "h_index_stance_label_shrink_neg"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
 #   feature_file = "group_features_by_stance_shrink"
    reports_dir = input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    queries = get_queries_from_df(df)
    labels = {q: int(queries[q].label) for q in queries}
    files = ['majority_nol', query_report_file_name]
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename=feature_file + 'stats_report', mode=ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename=feature_file + 'stats_report', mode=ValToClassMode.THREE_CLASSES_OPTIMISTIC)


def gen_majority_report(maj_file, output_filename, mode):
    out_rows = []
    class_to_val = VAL_TO_CLASS_DICT[mode]
    with open(maj_file, encoding='utf-8', newline='') as input_csv:
        reader = csv.DictReader(input_csv)
        for row in reader:
            shrinked_votes = {x: 0 for x in set(class_to_val.values())}
            for stance in range(1, 6):
                votes = row[str(stance)]
                shrinked_votes[class_to_val[stance]] += int(votes)
                sorted_votes = sorted(shrinked_votes.items(), key=lambda x: x[1], reverse=True)
                maj = sorted_votes[0][0]
                out_row = {'query':row['query'],
                           'label':class_to_val[int(row['label'])],
                           'majority_value':maj,
                           'majority_class':maj}
                for stance, num in shrinked_votes.items():
                    out_row[str(stance)] = num
            out_rows.append(out_row)
    fieldnames = ['query', 'label', 'majority_value', 'majority_class','1','3','5']
    with open(output_filename, 'w', encoding='utf-8', newline='') as outcsv:
        wr = csv.DictWriter(outcsv, fieldnames=fieldnames)
        wr.writeheader()
        for row in out_rows:
            wr.writerow(row)


def majority_reports():
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\majority\\'
    feature_file = "majority_nol_no_medical"
    feature_file = "h_index_stance_label_shrink_neg"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    #   feature_file = "group_features_by_stance_shrink"
    reports_dir = input_dir + '\\reports\\'
    query_report_file_name = feature_file + '_query_report.csv'
    queries = get_queries_from_df(df)
    labels = {q: int(queries[q].label) for q in queries}
    files = ['majority_nol', query_report_file_name]
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename=feature_file + 'stats_report', mode=ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename=feature_file + 'stats_report', mode=ValToClassMode.THREE_CLASSES_OPTIMISTIC)

    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\majority\\'
    query_report_file_name ='majority_query_report.csv'
    reports_dir = input_dir + '\\reports\\'
    files = ['majority_nol', query_report_file_name]
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename='maj_stats_report', mode=ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    gen_metrics_comparison(folder=reports_dir, query_filenames=files, actual_values=labels,
                           cmp_filename='maj_stats_report', mode=ValToClassMode.THREE_CLASSES_OPTIMISTIC)

def analyze_maj(fname, mode):
    input_dir = 'C:\\research\\falseMedicalClaims\\White and Hassan\\model input\\majority\\'
    reports_dir = input_dir + 'reports\\'
    reports_filename = fname + '_' + mode.name + '_report'
    gen_majority_report(input_dir +fname + '.csv', reports_dir+reports_filename+ '.csv',
                        mode)

    df = pd.read_csv(reports_dir + '\\' + reports_filename + '.csv')
    queries = get_queries_from_df(df)
    labels = {q: int(queries[q].label) for q in queries}

    gen_metrics_comparison(folder=reports_dir, query_filenames=[reports_filename], actual_values=labels,
                           cmp_filename=reports_filename + '_stats_report', mode=mode)
def gen_reports_files():
    pandas.set_option('display.max_rows', 50)
    pandas.set_option('display.max_columns', 50)
    pandas.set_option('display.width', 1000)  # Clerical work:
    # analyze_maj('majority_nol', ValToClassMode.THREE_CLASSES_OPTIMISTIC)
    binary_files = ['group_features_by_label_shrink_nol',
                    'group_features_by_stance_label_shrink_nol',
                    'group_features_by_stance_nol_pos_neg']
    opt_files = ['group_features_by_label_shrink_nol',
                 'group_features_by_stance_label_shrink_nol',
                 'group_features_by_stance_nol']
    for f in binary_files:
        learn_by_doctors_annotations(feature_file=f, val2class=ValToClassMode.BINARY, resample=False,
                                     directory="binary")
    for f in opt_files:
        for rs in [False, True]:
            learn_by_doctors_annotations(feature_file=f, val2class=ValToClassMode.THREE_CLASSES_OPTIMISTIC, resample=rs,
                                         directory="optimistic")

def gen_reports_files_dict(bin = False, opt = False, filter_queries = None):
    pandas.set_option('display.max_rows', 50)
    pandas.set_option('display.max_columns', 50)
    pandas.set_option('display.width', 1000)  # Clerical work:
    # analyze_maj('majority_nol', ValToClassMode.THREE_CLASSES_OPTIMISTIC)
    binary_files = ['group_features_by_label_shrink_dict_nol',
                    'group_features_by_stance_label_shrink_dict_nol',
                    'group_features_by_stance_pos_neg_dict_nol']
    opt_files = ['group_features_by_stance_nol', 'group_features_by_label_shrink_dict_nol',
                 'group_features_by_stance_label_shrink_dict_nol',
                 ]
    if bin:
        for f in binary_files:
            learn_by_doctors_annotations(feature_file=f, val2class=ValToClassMode.BINARY, resample=False,
                                         directory="binary", majority_filename='majority_binary_dict_nol.csv', filter_queries=filter_queries)
    if opt:
        for f in opt_files:
            for rs in [False, True]:
                learn_by_doctors_annotations(feature_file=f, val2class=ValToClassMode.THREE_CLASSES_OPTIMISTIC, resample=rs,
                                             directory="optimistic", majority_filename='majority_opt_dict_nol.csv', filter_queries=filter_queries)

def get_test_queries(fname):
    queries = []
    with open(fname,'r',newline='') as input_csv:
        reader = csv.DictReader(input_csv)
        for row in reader:
            if row['used']:
                queries.append(row['query'])
    return queries

def main():
    #learn_by_doctors_annotations(feature_file='group_features_by_stance_nol', val2class=ValToClassMode.THREE_CLASSES_PESSIMISTIC, resample=True,
    #                             directory="pessimistic", majority_filename='majority_nol.csv')
    filter_queries = get_test_queries('C:\\research\\falseMedicalClaims\\White and Hassan\\labeles_non_medical_amazon.csv')
    gen_reports_files_dict(opt= True, bin = False , filter_queries=filter_queries)
    #w_h_report()


if __name__ == '__main__':
    main()
