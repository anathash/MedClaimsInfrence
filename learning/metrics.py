from enum import Enum

from numpy import mean, math

from learning.dataHelper import get_class


METRICS_NAMES = ['reject_acc',
                 'reject_as_support',
                 'reject_as_neutral',
                 'reject_as_initial',
                 'neutral_acc',
                 'neutral_as_reject',
                 'neutral_as_support',
                 'neutral_as_initial',
                 'initial_acc',
                 'initial_as_support',
                 'initial_as_reject',
                 'initial_as_neutral',
                 'support_acc',
                 'support_as_initial',
                 'support_as_reject',
                 'support_as_neutral',
                 'actual_rejects',
                 'actual_neutral',
                 'actual_initial',
                 'actual_support',
                 'val_false_optimism_mae',
                 'val_false_pessimism_mae',
                 'class_false_optimism_mae',
                 'class_false_pessimism_mae',
                 'val_false_optimism_num',
                 'val_false_pessimism_num',
                 'class_false_optimism_num',
                 'class_false_pessimism_num',
                 'val_false_optimism_rate',
                 'val_false_pessimism_rate',
                 'class_false_optimism_rate',
                 'class_false_pessimism_rate',
                 'num_queries',
                 'num_rel',
                 'acc',
                 'mae',
                 'support_acc_percent',
                 'initial_acc_percent',
                 'neutral_acc_percent',
                 'reject_acc_percent',
                 'initial_acc_percent']


class Metrics:
    def __init__(self, initial_metrics, mode):
        self.val_false_optimism = []
        self.val_false_pessimism = []
        self.class_false_optimism = []
        self.class_false_pessimism = []
        self.mode = mode
        self.mae = []
        self.conf ={x:0 for x in METRICS_NAMES}
        for k,v in initial_metrics.items():
            assert (k in self.conf)
            self.conf[k] = v


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

        all_size = self.conf['actual_rejects'] + self.conf['actual_neutral'] + self.conf['actual_support'] + self.conf['actual_initial']
        self.conf['num_queries'] = all_size
        self.conf['val_false_optimism_rate'] = len(self.val_false_optimism)/all_size
        self.conf['val_false_pessimism_rate'] = len(self.val_false_pessimism)/all_size
        self.conf['class_false_optimism_rate'] = len(self.class_false_optimism)/all_size
        self.conf['class_false_pessimism_rate'] = len(self.class_false_pessimism)/all_size
        self.conf['acc'] = (self.conf['support_acc'] + self.conf['neutral_acc'] +self.conf['reject_acc']) /len((self.mae))
        self.conf['mae'] = mean(self.mae)

        self.conf['support_acc_percent'] = 0 if self.conf['actual_support'] == 0 else self.conf['support_acc'] / self.conf['actual_support']
        self.conf['neutral_acc_percent'] = 0 if self.conf['actual_neutral'] == 0 else self.conf['neutral_acc'] / self.conf['actual_neutral']
        self.conf['reject_acc_percent'] = 0 if self.conf['actual_rejects'] == 0 else self.conf['reject_acc'] / self.conf['actual_rejects']
        self.conf['initial_acc_percent'] = 0 if self.conf['actual_initial'] == 0 else self.conf['initial_acc'] / self.conf['actual_initial']

    def update_confusion_counters(self, as_reject_counter, as_neutral_counter, as_support_counter,as_initial_counter,
                                  prediction_class):
        if prediction_class == 1:
            self.conf[as_reject_counter] += 1
            return
        if prediction_class == 3:
            self.conf[as_neutral_counter] += 1
            return
        if prediction_class == 4:
            self.conf[as_initial_counter] += 1
            return
        elif prediction_class == 5:
            self.conf[as_support_counter] += 1
            return

    def update_confusion(self, actual_class, prediction_class):
        if actual_class == 1:
            self.update_confusion_counters(as_reject_counter='reject_acc',
                                                as_neutral_counter='reject_as_neutral',
                                                as_support_counter='reject_as_support',
                                                as_initial_counter='reject_as_initial',
                                                prediction_class=prediction_class)
        if actual_class == 3:
            self.update_confusion_counters( as_reject_counter='neutral_as_reject',
                                                as_neutral_counter='neutral_acc',
                                                as_support_counter='neutral_as_support',
                                                as_initial_counter='neutral_as_initial',
                                                prediction_class=prediction_class)
        if actual_class == 4:
            self.update_confusion_counters( as_reject_counter='initial_as_reject',
                                                as_neutral_counter='initial_as_neutral',
                                                as_support_counter='initial_as_support',
                                                as_initial_counter='initial_acc',
                                                prediction_class=prediction_class)


        if actual_class == 5:
            self.update_confusion_counters(as_reject_counter='support_as_reject',
                                                as_neutral_counter='support_as_neutral',
                                                as_support_counter='support_acc',
                                                as_initial_counter='support_as_initial',
                                                prediction_class=prediction_class)

    def update_metrics(self,value_label, model_value_prediction):
        if model_value_prediction > 0:
            self.mae.append(math.fabs(model_value_prediction-value_label))
        class_label = get_class(value_label, self.mode)
        model_class_prediction = get_class(model_value_prediction, self.mode)
        if model_value_prediction < 0:
            return
        self.conf['num_rel'] += 1
        self.update_confusion(actual_class=class_label,
                                        prediction_class=model_class_prediction)
        self.update_prediction_err(actual_val=value_label,
                                             prediction_val=model_value_prediction,
                                             actual_class=class_label,
                                             prediction_class=model_class_prediction)


