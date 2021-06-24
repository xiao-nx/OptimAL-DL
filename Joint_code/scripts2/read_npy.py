import os 
import numpy as np
import pandas as pd

fpath_cnn = './outputs_lstm/' # outputs_cnn outputs_lstm  outputs_bilstm



files = os.listdir(fpath_cnn)
f1_score_weighted_list = []
for file in files:   
    if file.endswith(".npy"):     
        fname = os.path.join(fpath_cnn, file)
        parameters_dic = np.load(fname,allow_pickle=True)
        parameters_dic = parameters_dic.item()
        f1_score_weighted_list.append(parameters_dic['f1_score_macro'])

print(len(f1_score_weighted_list))
idx = np.argmax(f1_score_weighted_list, axis=None)
print(idx)

best_fname = fpath_cnn + 'parameters_info_' + str(idx) + '.npy'
best_parameters = np.load(best_fname,allow_pickle=True)
print(fpath_cnn)
print(best_parameters)