import torch
import time
import math
import numpy as np
from utils.utils_ import log_string

def train(model, args, log, loss_criterion, optimizer, scheduler):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)
    
    # Move data to the same device as the model
    device = next(model.parameters()).device
    trainX, trainTE, trainY = trainX.to(device), trainTE.to(device), trainY.to(device)
    valX, valTE, valY = valX.to(device), valTE.to(device), valY.to(device)
    
    num_train, _, num_vertex = trainX.shape
    num_val = valX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)

    wait = 0
    val_loss_min = float('inf')
    best_model_wts = None
    train_total_loss = []
    val_total_loss = []

    # Train model
    log_string(log, '**** training model ****')
    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'early stop at epoch: {epoch:04d}')
            break
        
        # Train
        model.train()
        train_loss = 0
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            
            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            label = trainY[start_idx: end_idx]
            
            optimizer.zero_grad()
            pred = model(X, TE)
            pred = pred * std + mean
            loss_batch = loss_criterion(pred, label)
            train_loss += float(loss_batch) * (end_idx - start_idx)
            loss_batch.backward()
            optimizer.step()
            
            if (batch_idx+1) % 5 == 0:
                print(f'Training batch: {batch_idx+1} in epoch:{epoch}, training batch loss:{loss_batch:.4f}')
        
        train_loss /= num_train
        train_total_loss.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx in range(val_num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
                X = valX[start_idx: end_idx]
                TE = valTE[start_idx: end_idx]
                label = valY[start_idx: end_idx]
                pred = model(X, TE)
                pred = pred * std + mean
                loss_batch = loss_criterion(pred, label)
                val_loss += loss_batch * (end_idx - start_idx)
        val_loss /= num_val
        val_total_loss.append(val_loss)
        
        log_string(
            log,
            '%s | epoch: %04d/%d, training loss: %.4f, validation loss: %.4f' %
            (time.ctime(), epoch + 1, args.max_epoch, train_loss, val_loss))
        
        if val_loss <= val_loss_min:
            log_string(
                log,
                f'val loss decrease from {val_loss_min:.4f} to {val_loss:.4f}, saving model to {args.model_file}')
            wait = 0
            val_loss_min = val_loss
            best_model_wts = model.state_dict()
        else:
            wait += 1
            
        scheduler.step()

    model.load_state_dict(best_model_wts)
    torch.save(model, args.model_file)
    log_string(log, f'Training and validation are completed, and model has been stored as {args.model_file}')
    return train_total_loss, val_total_loss
