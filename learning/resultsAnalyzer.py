import csv
import math

from numpy import mean

from learning.dataHelper import get_class


class Metrics:
    def __init__(self):
        self.val_false_optimism = []
        self.val_false_pessimism = []
        self.class_false_optimism = []
        self.class_false_pessimism = []
        self.conf = {'reject_acc': 0,
                     'reject_as_support': 0,
                     'reject_as_neutral': 0,
                     'neutral_as_reject': 0,
                     'neutral_acc': 0,
                     'neutral_as_support': 0,
                     'support_as_reject': 0,
                     'support_as_neutral': 0,
                     'support_acc': 0,
                     'actual_rejects': 0,
                     'actual_neutral': 0,
                     'actual_support': 0,
                     'val_false_optimism_mae': 0,
                     'val_false_pessimism_mae': 0,
                     'class_false_optimism_mae': 0,
                     'class_false_pessimism_mae': 0,
                     'val_false_optimism_num': 0,
                     'val_false_pessimism_num': 0,
                     'class_false_optimism_num': 0,
                     'class_false_pessimism_num': 0,
                     'val_false_optimism_rate': 0,
                     'val_false_pessimism_rate': 0,
                     'class_false_optimism_rate': 0,
                     'class_false_pessimism_rate': 0,
                     }

    @staticmethod
    def update_err(actual, prediction, false_pes, false_opt):
        err = actual - prediction
        if err > 0:
            false_pes.append(err)
        elif err < 0:
            false_opt.append(math.fabs(err))

    def update_prediction_err(self, actual_val, prediction_val, actual_class, prediction_class):
        self.update_err(actual=actual_val, prediction=prediction_val,
                        false_pes=self.val_false_pessimism, false_opt=self.val_false_optimism)
        self.update_err(actual=actual_class, prediction=prediction_class,
                        false_pes=self.class_false_pessimism, false_opt=self.class_false_optimism)

    def process_results(self):
        self.conf['val_false_optimism_mae'] = mean(self.val_false_optimism)
        self.conf['val_false_pessimism_mae'] = mean(self.val_false_pessimism)
        self.conf['class_false_optimism_mae'] = mean(self.class_false_optimism)
        self.conf['class_false_pessimism_mae'] = mean(self.class_false_pessimism)

        self.conf['val_false_optimism_num'] = len(self.val_false_optimism)
        self.conf['val_false_pessimism_num'] = len(self.val_false_pessimism)
        self.conf['class_false_optimism_num'] = len(self.class_false_optimism)
        self.conf['class_false_pessimism_num'] = len(self.class_false_pessimism)

        all_size = self.conf['actual_rejects'] + self.conf['actual_neutral'] + self.conf['actual_support']
        self.conf['val_false_optimism_rate'] = len(self.val_false_optimism) /all_size
        self.conf['val_false_pessimism_rate'] = len(self.val_false_pessimism)/all_size
        self.conf['class_false_optimism_rate'] = len(self.class_false_optimism)/all_size
        self.conf['class_false_pessimism_rate'] = len(self.class_false_pessimism)/all_size

    def update_confusion_counters(self, actual_counter, as_reject_counter, as_neutral_counter, as_support_counter,
                                  prediction_class):
        self.conf[actual_counter] += 1
        if prediction_class == 1:
            self.conf[as_reject_counter] += 1
            return
        if prediction_class == 3:
            self.conf[as_neutral_counter] += 1
            return
        elif prediction_class == 5:
            self.conf[as_support_counter] += 1
            return

    def update_confusion(self, actual_class, prediction_class):
        if actual_class == 1:
            self.update_confusion_counters(actual_counter='actual_rejects', as_reject_counter='reject_acc',
                                                as_neutral_counter='reject_as_neutral',
                                                as_support_counter='reject_as_support',
                                                prediction_class=prediction_class)
        if actual_class == 3:
            self.update_confusion_counters(actual_counter='actual_neutral', as_reject_counter='neutral_as_reject',
                                                as_neutral_counter='neutral_acc',
                                                as_support_counter='neutral_as_support',
                                                prediction_class=prediction_class)

        if actual_class == 5:
            self.update_confusion_counters(actual_counter='actual_support', as_reject_counter='support_as_reject',
                                                as_neutral_counter='support_as_neutral',
                                                as_support_counter='support_acc',
                                                prediction_class=prediction_class)


def create_report_files(report_fname, confusion_fname, queries, learners, predictions, majority_classifier, labels):
    with open(report_fname, 'w', encoding='utf-8', newline='') as out:
        model_names = [learner.model_name() for learner in learners]
        model_names.append('majority')
        predictions['majority'] = majority_classifier.get_predictions()
        fieldnames = ['query', 'value_label', 'class_label']
        fieldnames.extend([model_name + '_class' for model_name in model_names])
        fieldnames.extend([model_name + '_value' for model_name in model_names])
        fieldnames.extend([model_name + '_acc' for model_name in model_names])
        fieldnames.extend([model_name + '_error' for model_name in model_names])
        fieldnames.extend([model_name + '_val_pessim' for model_name in model_names])
        fieldnames.extend([model_name + '_val_optim' for model_name in model_names])
        fieldnames.extend([model_name + '_class_pessim' for model_name in model_names])
        fieldnames.extend([model_name + '_calss_optim' for model_name in model_names])
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        metrics = {model_name:Metrics() for model_name in model_names}
        for q in queries:
            value_label = labels[q]
            class_label = get_class(value_label)
            row = {'query': q, 'value_label': value_label, 'class_label': class_label}
            for model_name in model_names:
                predicted_class = predictions[model_name][q].class_prediction
                predicted_val = predictions[model_name][q].mean_prediction
                metrics[model_name].update_confusion(actual_class=class_label,
                                 prediction_class=predicted_class)
                metrics[model_name].update_prediction_err(actual_val=value_label,
                                                          prediction_val=predicted_val,
                                                          actual_class=class_label,
                                                          prediction_class=predicted_class)
                row[model_name + "_class"] = predicted_class
                row[model_name + "_value"] = predicted_val
                row[model_name + "_acc"] = int(class_label == predicted_class)
                row[model_name + "_error"] = math.fabs(value_label - predicted_val)
                row[model_name + "_val_pessim"] = 1 if predicted_val < value_label else 0
                row[model_name + "_val_optim"] = 1 if predicted_val > value_label else 0
                row[model_name + "_class_pessim"] = 1 if predicted_class < class_label else 0
                row[model_name + "_calss_optim"] = 1 if predicted_class > class_label else 0
            writer.writerow(row)

        for model_name in model_names:
            metrics[model_name].process_results()

        with open(confusion_fname, 'w', encoding='utf-8', newline='') as conf_out:
            fieldnames = ['metric_name']
            fieldnames.extend([model_name for model_name in model_names])
            writer = csv.DictWriter(conf_out, fieldnames=fieldnames)
            writer.writeheader()
            for k in metrics['majority'].conf.keys():
                row = {'metric_name':k}
                for model_name in model_names:
                    row[model_name] = metrics[model_name].conf[k]
                writer.writerow(row)



def get_year_dict(query_file):
    years = {}
    with open(query_file, 'r', encoding='utf-8', newline='') as query_csv:
        query_reader = csv.DictReader(query_csv)
        for row in query_reader:
            date = row['date']
            year = date.split('.')[2].strip()
            years[row['long query']] = year
    return years


def get_key_from_range(from_year, to_year):
    return str(from_year)+'_to_' + str(to_year)


def gen_query_lists(ranges):
    queries = {}
    for (from_year,to_year) in ranges:
        k = get_key_from_range(from_year,to_year)
        queries[k] = []
    return queries


def get_range_key_by_year(ranges, year):
    for (from_year,to_year) in ranges:
        if year >= from_year and year <=to_year:
            return get_key_from_range(from_year,to_year)


def group_by_year(ranges, result_file, query_file, file_prefix):
    years = get_year_dict(query_file)
    queries = gen_query_lists(ranges)
    avgs = {get_key_from_range(range[0],range[1]):{} for range in ranges}
    with open(result_file, 'r', encoding='utf-8', newline='') as res_csv:
        res_reader = csv.DictReader(res_csv)
        for row in res_reader:
            year = int(years[row['query']])
            range_key = get_range_key_by_year(ranges, year)
            queries[range_key].append(row)

        fieldnames = res_reader.fieldnames
        metrics = [x for x in fieldnames if x!='query']
        for k, q_list in queries.items():
            avgs[k] = {x:[] for x in metrics}
            with open(file_prefix + k +'.csv', 'w', encoding='utf-8', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=fieldnames)
                writer.writeheader()
                for row in q_list:
                    writer.writerow(row)
                    for metric in metrics:
                        avgs[k][metric].append(float(row[metric]))

        with open(file_prefix + 'ranges_metrics.csv', 'w', encoding='utf-8', newline='') as ranges_out:
            fieldnames = ['metrics_name']
            fieldnames.extend(avgs.keys())

            writer = csv.DictWriter(ranges_out, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                row = {'metrics_name':metric}
                for k in avgs.keys():
                    row[k] = mean(avgs[k][metric])
                writer.writerow(row)


def main():
    folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael_sigal_Irit\\by_group\\reports\\'
    #ranges = [(1998, 2002), (2003, 2007), (2007, 2011), (2011, 2015), (2016, 2021)]
    #ranges = [(1998, 2010), (2010, 2014), (2014, 2020)]
    ranges = [(1998, 2014), (2015, 2020)]
    #ranges = [(1998, 2000), (2001, 2003), (2012, 2019)]
    group_by_year(ranges, folder + 'group_features_by_stance_citation_range_1query_report.csv',
                           'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv',
                  folder + 'citation_range_1_by_years_')
if __name__ == '__main__':
    main()




