import csv
import math

from numpy import mean

from learning.dataHelper import get_class, Prediction, ValToClassMode, REJECT, NEUTRAL, SUPPORT
from learning.metrics import Metrics, METRICS_NAMES


def create_report_files_old(report_fname, confusion_fname, queries, learners, predictions, labels):
    with open(report_fname, 'w', encoding='utf-8', newline='') as out:
        model_names = [learner.model_name() for learner in learners]
        model_names.append('majority')
        #predictions['majority'] = majority_classifier.get_predictions()
        fieldnames = ['query', 'value_label', 'class_label']
        fieldnames.extend([model_name + '_class' for model_name in model_names])
        fieldnames.extend([model_name + '_value' for model_name in model_names])
        fieldnames.extend([model_name + '_acc' for model_name in model_names])
        fieldnames.extend([model_name + '_error' for model_name in model_names])
        fieldnames.extend([model_name + '_val_pessim' for model_name in model_names])
        fieldnames.extend([model_name + '_val_optim' for model_name in model_names])
        fieldnames.extend([model_name + '_class_pessim' for model_name in model_names])
        fieldnames.extend([model_name + '_class_optim' for model_name in model_names])
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
                row[model_name + "_class_optim"] = 1 if predicted_class > class_label else 0
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


def create_query_report_file(report_fname, input_dir, feature_file,  queries, learners, predictions, labels, val2class, md = False):
    with open(report_fname, 'w', encoding='utf-8', newline='') as out:
        model_names = [learner.model_name() for learner in learners]
        error_queries = {x: {} for x in model_names}
        fieldnames = ['query', 'value_label', 'class_label']
        fieldnames.extend([model_name + '_class' for model_name in model_names])
        fieldnames.extend([model_name + '_value' for model_name in model_names])
        fieldnames.extend([model_name + '_acc' for model_name in model_names])
        fieldnames.extend([model_name + '_error' for model_name in model_names])
        fieldnames.extend([model_name + '_val_pessim' for model_name in model_names])
        fieldnames.extend([model_name + '_val_optim' for model_name in model_names])
        fieldnames.extend([model_name + '_class_pessim' for model_name in model_names])
        fieldnames.extend([model_name + '_class_optim' for model_name in model_names])
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for q in queries:
            if 'dummy' in q:
                continue
            if md and not q in labels:
                continue
            value_label = labels[q]
            class_label = get_class(value_label, val2class)
            row = {'query': q, 'value_label': value_label, 'class_label': class_label}
            for model_name in model_names:
                predicted_class = predictions[model_name][q].class_prediction
                predicted_val = predictions[model_name][q].mean_prediction
                if predicted_class !=  class_label:
                    error_queries[model_name][q]= predicted_class
                row[model_name + "_class"] = predicted_class
                row[model_name + "_value"] = predicted_val
                row[model_name + "_acc"] = int(class_label == predicted_class)
                row[model_name + "_error"] = math.fabs(value_label - predicted_val)
                row[model_name + "_val_pessim"] = 1 if predicted_val < value_label else 0
                row[model_name + "_val_optim"] = 1 if predicted_val > value_label else 0
                row[model_name + "_class_pessim"] = 1 if predicted_class < class_label else 0
                row[model_name + "_class_optim"] = 1 if predicted_class > class_label else 0
            writer.writerow(row)
    create_false_predictions_feature_file(input_dir, feature_file, error_queries)


def create_false_predictions_feature_file(input_dir, feature_file, error_queries_dict):
    for model, error_queries in error_queries_dict.items():
        error_query_names = error_queries.keys()
        rows = []
        featuer_file_path = input_dir  + feature_file  +'.csv'
        with open(featuer_file_path, encoding='utf-8', newline='') as queries_csv:
            reader = csv.DictReader(queries_csv)
            for row in reader:
                query = row ['query']
                if query in error_query_names:
                    row['prediction'] = error_queries[query]
                    rows.append(row)
        output_filename = input_dir + '/reports/' + feature_file +'_' + str(model) + '_false_predictions.csv'
        with open(output_filename, 'w', encoding='utf-8', newline='') as output_csv:
            fn = list(reader.fieldnames)
            fn.insert(2, 'prediction')
            print(fn)
            writer = csv.DictWriter(output_csv, fn)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def get_results_from_file(filename):
    with open(filename, encoding='utf-8', newline='') as queries_csv:
        reader = csv.DictReader(queries_csv)
        fieldnames = reader.fieldnames
        model_names = []
        for field in fieldnames:
            split_field = field.split('_')
            if len(split_field) > 1 and split_field[1]=='value':
                model_names.append(split_field[0])
        results = []
        for row in reader:
            query_name = row['query']
            value_label = row['value_label']
            class_label = get_class(value_label)
            result_entry = {'query': query_name, 'value_label': value_label, 'class_label': class_label}
            for model in model_names:
                model_value_entry = model + '_value'
                model_value_prediction = int(row[model_value_entry])
                model_class_prediction = get_class(model_value_prediction)
                result_entry[model] = Prediction(model_value_prediction, model_class_prediction)
                results.append(result_entry)
        return results, model_names


def get_model_names_from_fieldnames(fieldnames):
    model_names = []
    for field in fieldnames:
        split_field = field.split('_')
        if len(split_field) > 1 and split_field[1] == 'value':
            model_names.append(split_field[0])
    return model_names


def gen_actual_labels_dict(labels_filename):
    labels = {}
    with open(labels_filename,'r', newline='') as labels_csv:
        reader = csv.DictReader(labels_csv)
        for row in reader:
            query = row['query']
            value_label = row['value_label']
            labels[query] = value_label
    return labels


def compute_class_label_distribution_old(labels, mode):
    dist = {'actual_rejects':0,'actual_neutral':0,'actual_support':0, 'actual_initial':0}
    for q, value_label in labels.items():
        if 'dummy' in q or not value_label:
            continue
        class_label = get_class(int(value_label), mode)
        if class_label == 1:
            dist['actual_rejects'] += 1
        elif class_label == 3:
            dist['actual_neutral'] += 1
        elif class_label == 4:
            dist['actual_initial'] += 1
        elif class_label == 5:
            dist['actual_support'] += 1
    return dist


def compute_class_label_distribution(labels, mode):
    dist = {'actual_rejects':0,'actual_neutral':0,'actual_support':0, 'actual_initial':0}
    for q, value_label in labels.items():
        if 'dummy' in q or not value_label:
            continue
        class_label = get_class(int(value_label), mode)
        if class_label == REJECT:
            dist['actual_rejects'] += 1
        elif class_label == NEUTRAL:
            dist['actual_neutral'] += 1
        elif class_label == SUPPORT:
            dist['actual_support'] += 1
    return dist


def gen_metrics(query_filename, actual_values, mode, md = False, filter_queries= None):
    with open(query_filename,'r',  newline='') as queries_csv:
        reader = csv.DictReader(queries_csv)
        fieldnames = reader.fieldnames
        model_names = get_model_names_from_fieldnames(fieldnames)
        label_dist = compute_class_label_distribution(actual_values, mode)
        metrics = {model_name: Metrics(label_dist, mode) for model_name in model_names}
        for row in reader:
            query = row['query'].strip()
            if  query not in actual_values or not actual_values[query]:
                #print(query + ' not in actual values')
                #assert(False)
                continue
            if filter_queries and query not in filter_queries:
                continue
            value_label = int(actual_values[query])
            if not md and 'value_label' in row:
                assert(value_label == int(row['value_label']))
                #move  class infrence to metrics. In metrics - gen 2 forms of classes, 2 metrics, one for each model
            for model in model_names:
                model_value_entry = model + '_value'
                model_value_prediction = float(row[model_value_entry])
                metrics[model].update_metrics(value_label=value_label,model_value_prediction= model_value_prediction)
        for model_name in model_names:
            metrics[model_name].process_results()
    return metrics


def gen_all_metrics_comparison(folder, files, actual_values, cmp_filename='models_cmp', md = False, val2class = None, filter_queries = None):
    if not val2class:
        gen_metrics_comparison(folder=folder, query_filenames=files, actual_values=actual_values, cmp_filename=cmp_filename, mode=ValToClassMode.THREE_CLASSES_OPTIMISTIC, md=md ,filter_queries=filter_queries)
        gen_metrics_comparison(folder=folder, query_filenames=files, actual_values=actual_values, cmp_filename=cmp_filename, mode=ValToClassMode.THREE_CLASSES_PESSIMISTIC, md=md,filter_queries=filter_queries)
        gen_metrics_comparison(folder=folder, query_filenames=files, actual_values=actual_values, cmp_filename=cmp_filename, mode=ValToClassMode.FOUR_CLASSES, md=md,filter_queries=filter_queries)

    else:
        gen_metrics_comparison(folder=folder, query_filenames=files, actual_values=actual_values,
                               cmp_filename=cmp_filename, mode=val2class, md=md, filter_queries=filter_queries)
#    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename=cmp_filename, mode=ValToClassMode.THREE_CLASSES_OPTIMISTIC, md=md)
#    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename=cmp_filename, mode=ValToClassMode.THREE_CLASSES_PESSIMISTIC, md=md)
#    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename=cmp_filename, mode=ValToClassMode.FOUR_CLASSES, md=md)



def gen_metrics_comparison(folder, query_filenames, actual_values, cmp_filename, mode, md = False, filter_queries=None):
    metrics = {}
    fieldnames = ['metric_name']

    if mode == ValToClassMode.FOUR_CLASSES: #4 classes
        metric_names = METRICS_NAMES
    else:
        metric_names = [x for x in METRICS_NAMES if 'initial' not in x]

    #actual_values = gen_actual_labels_dict(label_file)
    for f in query_filenames:
        query_filename = folder+f
        if not query_filename.endswith('.csv'):
            query_filename +='.csv'
        metrics_entry = gen_metrics(query_filename=query_filename, actual_values=actual_values, mode=mode, md = md, filter_queries=filter_queries)
        fieldnames.extend([x for x in metrics_entry.keys()])
        metrics[f] = metrics_entry
    if filter_queries:
        filename = folder+cmp_filename+'_'+str(mode.name)+'_filtered.csv'
    else:
        filename = folder + cmp_filename + '_' + str(mode.name) + '.csv'
    with open(filename, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metric_names:
            row = {'metric_name': metric}
            for f in query_filenames:
                for model in metrics[f]:
                    row[model] = metrics[f][model].conf[metric]
            writer.writerow(row)


def merge_metrices(folder, files, merge_file_name):
    metrics = {}
    metric_names = set()
    fieldnames = {}
    for f in files:
        metrics[f] = {}
        with open(folder + f, 'r', encoding='utf-8', newline='') as f_csv:
            reader = csv.DictReader(f_csv)
            fieldnames[f]= [x for x in reader.fieldnames if x != 'metric_name']
            for row in reader:
                metric_name = row['metric_name']
                metric_names.update(metric_name)
                metrics[f][metric_name] = row

    with open(merge_file_name, 'w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metric_names:
            row = {'metric_name':metric_name}
            for f in files:
                for field in fieldnames[f]:
                    row[f + '_' + field] = metrics[f][metric][field]
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


def group_by_year_exp():
    folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael_sigal_Irit\\by_group\\reports\\'
    # ranges = [(1998, 2002), (2003, 2007), (2007, 2011), (2011, 2015), (2016, 2021)]
    # ranges = [(1998, 2010), (2010, 2014), (2014, 2020)]
    ranges = [(1998, 2014), (2015, 2020)]
    # ranges = [(1998, 2000), (2001, 2003), (2012, 2019)]
    group_by_year(ranges, folder + 'group_features_by_stance_citation_range_1query_report.csv',
                  'C:\\research\\falseMedicalClaims\\ECAI\\examples\\classified\\queries1_2.csv',
                  folder + 'citation_range_1_by_years_')


def cmp_Yael_sigal_Irit():
    folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael_sigal_Irit\\by_group\\reports\\'
    files = ['google labels', 'majority','group_features_by_stance_shrinkquery_report']
    label_file = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael_sigal_Irit\\labels.csv'
    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename='google_maj_ML.csv')

def cmp_Yael():
    folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael\\by_group\\reports\\'
    files = ['google labels', 'majority','group_features_by_stance_query_report']
    label_file = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael\\labels_all.csv'
    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename='google_maj_ML_Yael.csv')


def cmp_Sigal():
    folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Sigal\\by_group\\reports\\'
    files = ['google labels', 'majority','group_features_by_stancequery_report']
    label_file = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\Yael\\labels.csv'
    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename='google_maj_ML_Yael.csv')


def cmp_sample_1_2_all():
    folder = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\all_equal_weights\\by_group\\reports\\'
    files = ['google labels', 'majority','group_features_by_stance_query_report']
    label_file = 'C:\\research\\falseMedicalClaims\\ECAI\\model input\\all_equal_weights\\labels.csv'
    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename='google_maj_ML', mode=ValToClassMode.THREE_CLASSES_OPTIMISTIC)
    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename='google_maj_ML', mode=ValToClassMode.THREE_CLASSES_PESSIMISTIC)
    gen_metrics_comparison(folder=folder, query_filenames=files, label_file=label_file, cmp_filename='google_maj_ML', mode=ValToClassMode.FOUR_CLASSES)


def gen_google_labels_error_report(label_file, google_file, output_file):
    actual_values = gen_actual_labels_dict(label_file)
    google_stats = []
    with open(google_file, 'r', newline='') as res_csv:
        res_reader = csv.DictReader(res_csv)
        for row in res_reader:
            query = row['query']
            value_label =int(actual_values[query])
            actual_class = get_class(value_label, ValToClassMode.THREE_CLASSES_PESSIMISTIC)
            predicted_value = int(row['Google_value'])
            if predicted_value < 0:
                continue
            predicted_class = get_class(predicted_value,ValToClassMode.THREE_CLASSES_PESSIMISTIC)
            mae = math.fabs(value_label-predicted_value)
            acc = int(actual_class == predicted_class)
            google_stats.append({'query':query,'value_label':value_label,'class_label':actual_class,
                        'predicted_value':predicted_value,'predicted_class':predicted_class,
                        'google_mae':mae,'google_acc': acc})

    with open(output_file , 'w', encoding='utf-8', newline='') as out:
        fieldnames = ['query','value_label','class_label','predicted_value','predicted_class','google_mae','google_acc']
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for stat_entry in google_stats:
            writer.writerow(stat_entry)


def get_md_labels(file):
    labels = {}
    with open(file, 'r', encoding='utf-8', newline='') as queries_csv:
        reader = csv.DictReader(queries_csv)
        for row in reader:
            query = row['query']
            value = row['value_label']
            if value.isdigit():
                labels[query] = int(value)
    return labels



def cmp_md(md_file, query_report_full_file_name, output_file, reports_dir, feature_file):
    md_lables = get_md_labels(md_file)
    md_rows = []
    with open(query_report_full_file_name, 'r', encoding='utf-8', newline='') as report_csv:
        reader = csv.DictReader(report_csv)
        fieldnames = reader.fieldnames

        for row in reader:
            query = row['query']
            if query in md_lables:
                new_row = {'query': row['query']}
                for f in fieldnames:
                    if f == 'query':
                        continue
                    new_row['annotators_' + f] = row[f]
                new_row['value_label'] = md_lables[query]
                new_row['class_label'] = get_class(md_lables[query], ValToClassMode.THREE_CLASSES_PESSIMISTIC)
                md_rows.append(new_row)

        with open(reports_dir+ output_file+'.csv', 'w', encoding='utf-8', newline='') as out:
            fieldnames = reader.fieldnames
            fieldnames.extend(['annotators_value_label', 'annotators_class_label'])
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in md_rows:
                writer.writerow(row)
#    files = ['google labels', 'majority', output_file]
#    label_file = 'C:\\research\\falseMedicalClaims\\IJCAI\\annotators\\Ruthi\\Cochrane_Ruthy.csv'
#    gen_all_metrics_comparison(folder=reports_dir,files= files,label_file=label_file, cmp_filename=feature_file+'md_cmp')


def main():
    cmp_md(md_file='C:\\research\\falseMedicalClaims\\IJCAI\\annotators\\Ruthi\\Cochrane_Ruthy.csv',
           query_report_full_file_name ='C:\\research\\falseMedicalClaims\\IJCAI\model input\\ecai_paste\\by_group\\reports\\group_features_by_stance_query_report.csv',
           output_file ='md_group_features_by_stance_query_report',
           reports_dir='C:\\research\\falseMedicalClaims\\IJCAI\model input\\ecai_paste\\by_group\\reports\\',
           feature_file ='group_features_by_stance')
    return
    #cmp_sample_1_2_all()
   # gen_all_metrics_comparison(folder=reports_dir,files= files,label_file='C:\\research\\falseMedicalClaims\\IJCAI\\annotators\Ruthi\\')

    gen_google_labels_error_report(label_file='C:\\research\\falseMedicalClaims\\ECAI\\model input\\all_equal_weights\\labels.csv',
                                   google_file='C:\\research\\falseMedicalClaims\\ECAI\\model input\\all_equal_weights\\by_group\\reports\\google labels.csv',
                                   output_file='C:\\research\\falseMedicalClaims\\ECAI\\model input\\all_equal_weights\\by_group\\reports\\google labels_stats.csv')

if __name__ == '__main__':
    main()




