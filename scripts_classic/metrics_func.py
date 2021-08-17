import os
from sklearn import metrics

def save_model_func(model, epoch, path='outputs', **kwargs):
    """
    parameters:
    model: trained model
    path: the file path to save model
    loss: loss
    last_loss: the loss of beset epoch
    kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        name = cur_time + '_epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')
            
def performance_evaluation_func(y_predicted, y_label,epoch='epoch',flag='train',loss=None):
    """
    parameters:
    y_predicted:
    y_label:
    """

    # define a dict to save parameters.
    parameters_dict = {'epoch':epoch,'flag':flag}
    
    if flag=='train':
        parameters_dict.update({'loss':loss})

    # calculate accuracy
    accuracy = metrics.accuracy_score(y_label, y_predicted)
    # calculate precision
    precision_macro = metrics.precision_score(y_label, y_predicted, average="macro")
    precision_micro = metrics.precision_score(y_label, y_predicted, average="micro")
    precision_weighted = metrics.precision_score(y_label, y_predicted, average="weighted")
    # Recall
    recall_macro = metrics.recall_score(y_label, y_predicted, average='macro')
    recall_micro = metrics.recall_score(y_label, y_predicted, average='micro')
    recall_weighted = metrics.recall_score(y_label, y_predicted, average='weighted')
    # F1 score
    f1_score_macro = metrics.f1_score(y_label, y_predicted, average='macro')
    f1_score_micro = metrics.f1_score(y_label, y_predicted, average='micro')
    f1_score_weighted = metrics.f1_score(y_label, y_predicted, average='weighted')

    # confusion matrix, return tn, fp, fn, tp.
    # classifier_labels = ['treatment','symptomatic relief','contradiction','effect']
    #classifier_labels = [0,1,2,3]
    #confusion_matrix = metrics.confusion_matrix(y_label, y_predicted, labels=classifier_labels)
    confusion_matrix = metrics.confusion_matrix(y_label, y_predicted)
    print('flag: \n',flag)
    print(confusion_matrix)

    # add parameters to dict
    parameters_dict.update({'accuracy':format(accuracy,'4f')})
    parameters_dict.update({'precision_macro':format(precision_macro, '.4f')})  
    parameters_dict.update({'precision_micro':format(precision_micro, '.4f')})
    parameters_dict.update({'precision_weighted':format(precision_weighted, '.4f')})
    parameters_dict.update({'recall_macro':format(recall_macro, '.4f')})
    parameters_dict.update({'recall_micro':format(recall_micro, '.4f')})
    parameters_dict.update({'recall_weighted':format(recall_weighted, '.4f')})
    parameters_dict.update({'f1_score_macro':format(f1_score_macro, '.4f')})
    parameters_dict.update({'f1_score_micro':format(f1_score_micro, '.4f')})
    parameters_dict.update({'f1_score_weighted':format(f1_score_weighted, '.4f')})
    parameters_dict.update({'confusion_matrix':confusion_matrix})

    return parameters_dict
    
    
def save_parameters_txt(params_list, path='outputs',**kwargs):
    """
    parameters:
    kwargs: every_epoch or best_epoch
    return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    save_fname = os.path.join(path, 'parameters_info.txt')
    with open(save_fname,"w") as f:
        for line in params_list:
            f.write(str(line).replace(os.linesep,'') + '\n')
            
