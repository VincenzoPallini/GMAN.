import argparse
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.utils_ import log_string, metric
from utils.utils_ import count_parameters, load_data

from model.model_ import GMAN
from model.train import train
from model.test import test

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type=int, default=5,
                    help='a time step is 5 mins')
parser.add_argument('--num_his', type=int, default=12,
                    help='history steps')
parser.add_argument('--num_pred', type=int, default=12,
                    help='prediction steps')
parser.add_argument('--L', type=int, default=1,
                    help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=8,
                    help='number of attention heads')
parser.add_argument('--d', type=int, default=8,
                    help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.7,
                    help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2,
                    help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--max_epoch', type=int, default=1,
                    help='epoch to run')
parser.add_argument('--patience', type=int, default=10,
                    help='patience for early stop')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='decay epoch')
parser.add_argument('--traffic_file', default='/kaggle/working/GMAN./data/pems-bay.h5',
                    help='traffic file')
parser.add_argument('--SE_file', default='/kaggle/working/GMAN./data/SE(PeMS).txt',
                    help='spatial embedding file')
parser.add_argument('--model_file', default='/kaggle/working/GMAN./data/GMAN.pkl',
                    help='save the model to disk')
parser.add_argument('--log_file', default='/kaggle/working/GMAN./data/log',
                    help='log file')
args = parser.parse_args()

def main():
    log = open(args.log_file, 'w')
    log_string(log, str(args)[10: -1])
    T = 24 * 60 // args.time_slot  # Number of time steps in one day

    # load data
    log_string(log, 'loading data...')
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)
    log_string(log, f'trainX: {trainX.shape}\t\t trainY: {trainY.shape}')
    log_string(log, f'valX:   {valX.shape}\t\tvalY:   {valY.shape}')
    log_string(log, f'testX:   {testX.shape}\t\ttestY:   {testY.shape}')
    log_string(log, f'mean:   {mean:.4f}\t\tstd:   {std:.4f}')
    log_string(log, 'data loaded!')

    # build model
    log_string(log, 'compiling model...')
    model = GMAN(SE, args, bn_decay=0.1)
    loss_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.decay_epoch,
                                          gamma=0.9)
    parameters = count_parameters(model)
    log_string(log, 'trainable parameters: {:,}'.format(parameters))

    # train model
    start = time.time()
    loss_train, loss_val = train(model, args, log, loss_criterion, optimizer, scheduler)
    log_string(log, 'Training completed')

    # test model
    log_string(log, 'Testing model...')
    trainPred, valPred, testPred = test(args, log)
    
    # Print final results
    train_mae, train_rmse, train_mape = metric(trainPred, trainY)
    val_mae, val_rmse, val_mape = metric(valPred, valY)
    test_mae, test_rmse, test_mape = metric(testPred, testY)
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))

    # Print performance for each prediction step
    log_string(log, 'Performance in each prediction step')
    for step in range(args.num_pred):
        mae, rmse, mape = metric(testPred[:, step], testY[:, step])
        log_string(log, f'Step {step + 1:02d}: MAE {mae:.2f}, RMSE {rmse:.2f}, MAPE {mape * 100:.2f}%')

    end = time.time()
    log_string(log, f'Total time: {(end - start) / 60:.1f} minutes')
    log.close()

if __name__ == '__main__':
    main()
