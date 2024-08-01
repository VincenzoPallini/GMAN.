import torch
import time
import math
import numpy as np
from utils.utils_ import log_string, metric
from utils.utils_ import load_data

def test(args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)
    
    # Load the model and move it to the appropriate device
    model = torch.load(args.model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Move data to the same device as the model
    trainX, trainTE = trainX.to(device), trainTE.to(device)
    valX, valTE = valX.to(device), valTE.to(device)
    testX, testTE = testX.to(device), testTE.to(device)
    
    num_train, _, num_vertex = trainX.shape
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    # Test model
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():
        model.eval()

        # Train
        trainPred = []
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            trainPred.append(pred_batch.cpu().numpy())
        trainPred = np.concatenate(trainPred, axis=0)
        trainPred = trainPred * std + mean

        # Val
        valPred = []
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            X = valX[start_idx: end_idx]
            TE = valTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            valPred.append(pred_batch.cpu().numpy())
        valPred = np.concatenate(valPred, axis=0)
        valPred = valPred * std + mean

        # Test
        testPred = []
        start_test = time.time()
        for batch_idx in range(test_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            X = testX[start_idx: end_idx]
            TE = testTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            testPred.append(pred_batch.cpu().numpy())
        testPred = np.concatenate(testPred, axis=0)
        testPred = testPred * std + mean
    end_test = time.time()

    train_mae, train_rmse, train_mape = metric(trainPred, trainY.cpu().numpy())
    val_mae, val_rmse, val_mape = metric(valPred, valY.cpu().numpy())
    test_mae, test_rmse, test_mape = metric(testPred, testY.cpu().numpy())
    
    log_string(log, 'testing time: %.1fs' % (end_test - start_test))
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae, train_rmse, train_mape * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae, val_rmse, val_mape * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))
    log_string(log, 'performance in each prediction step')
    
    for step in range(args.num_pred):
        test_mae, test_rmse, test_mape = metric(testPred[:, step], testY[:, step].cpu().numpy())
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                   (step + 1, test_mae, test_rmse, test_mape * 100))
    
    return trainPred, valPred, testPred
