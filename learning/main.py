import pandas
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from learning import dataHelper
from learning.NNlearn import NNLearner, get_parms
from learning.dataHelper import Stats, RANK_METHODS, get_queries_from_df, create_report_file, MajorityClassifier
from learning.networks import Layer, TwoLayersNet
from learning.sklearner import SKLearner


def test_models(learners, queries, split, method):
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
        for test_query in queries:
            if test_query == 'all.csv':
                continue
            data = dataHelper.get_data(queries, test_query, method)
            learner.learn(data)
            if split == dataHelper.Split.BY_QUERY:
                print(test_query)
                stats = dataHelper.test_query_set(method, data.train_queries, learner)
                for rm, stats in stats.items():
                    train_acc[model_name][rm] += stats.acc
                    train_mae[model_name][rm] += stats.mae
                stats = dataHelper.test_query_set(method, data.test_queries, learner)
                predictions[model_name][test_query] = stats['group'].predictions[test_query]
                for rm, stats in stats.items():
                    test_acc[model_name][rm] += stats.acc
                    test_mae[model_name][rm] += stats.mae
                    queries_stats[test_query][model_name][rm] += stats.mae
        for rm in RANK_METHODS[method]:
            train_acc[model_name][rm] /= len(queries)
            test_acc[model_name][rm] /= len(queries)
            train_mae[model_name][rm] /= len(queries)
            test_mae[model_name][rm] /= len(queries)

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
    input_dir = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael\\by_group'
    #df = pd.read_csv(input_dir + '\\group_features_by_stance.csv')
    #df = pd.read_csv(input_dir + '\\group_features_by_stance_no_enum.csv')
    feature_file = "group_features_by_stance_citation_range_10"
    df = pd.read_csv(input_dir + '\\' + feature_file + '.csv')
    queries = get_queries_from_df(df)
    labels = {q:int(queries[q].label) for q in queries}
    mc = MajorityClassifier(input_dir + '\\majority.csv')
    decisionTreeLearner = SKLearner(DecisionTreeClassifier(random_state=0))
    layers = [Layer(input=5, output=10), Layer(input=10, output=10)]
    #layers = [Layer(input=31, output=20), Layer(input=20, output=10)]
    #layers = [Layer(input=26, output=20), Layer(input=20, output=10)]
    net = TwoLayersNet(layers)
    params = get_parms(net)
    nnlearner = NNLearner(dataHelper.Method.GROUP_ALL, net=net, params=params)
    learners = [decisionTreeLearner]
    #learners = [nnlearner]
    predictions = test_models(learners, queries, dataHelper.Split.BY_QUERY, dataHelper.Method.GROUP_ALL)
    create_report_file(input_dir + '\\'+feature_file+'_report.csv',queries=queries,
                       learners = learners,predictions=predictions,majority_classifier=mc,labels=labels
                       )

def main():
    pandas.set_option('display.max_rows', 50)
    pandas.set_option('display.max_columns', 50)
    pandas.set_option('display.width', 1000)  # Clerical work:
    group_all()


if __name__ == '__main__':
    main()
