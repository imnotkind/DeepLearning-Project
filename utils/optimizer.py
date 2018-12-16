import subprocess
from sklearn.model_selection import ParameterGrid
import copy

FILE_PATH = "result.csv"

if __name__ == "__main__":
    param_grid = {'epoches' : ['500'],
                  'batch_size' : ['500'],
                  'patience' : ['50'],
                  'learning_rate' : ['0.001']}

    param_grid_fcn = copy.deepcopy(param_grid)
    param_grid_fcn['num_layers'] = ['(512,256,100)', '(256,128,50)', '(128,64,25)']
    param_gird_fcn['model'] = ['fcn']

    for params in ParameterGrid(param_grid_fcn):
        print(params)
        proc = subprocess.Popen(['python', 'learner.py',
                                 params['epoches'],
                                 params['batch_size'],
                                 params['patience'],
                                 params['learning_rate'],
                                 params['num_layers'],
                                 params['model'],
                                 FILE_PATH])
        proc.wait()

    param_grid_cnn = copy.deepcopy(param_grid)
    param_grid_cnn['num_layers'] = ['(64,256,100)', '(32,128,50)', '(16,64,25)']
    param_gird_cnn['model'] = ['cnn']
    for params in ParameterGrid(param_grid_cnn):
        print(params)
        proc = subprocess.Popen(['python', 'learner.py',
                                 params['epoches'],
                                 params['batch_size'],
                                 params['patience'],
                                 params['learning_rate'],
                                 params['num_layers'],
                                 params['model'],
                                 FILE_PATH])
        proc.wait()
