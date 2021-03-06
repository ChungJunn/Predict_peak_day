import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import FSIterator

import numpy as np
import pandas as pd

from sklearn.metrics import recall_score, f1_score, precision_score


def train_main(args, model, train_path, criterion, optimizer):
    iloop=0
    current_loss =0
    all_losses = [] 
    batch_size = args.batch_size
    train_iter = FSIterator(train_path, batch_size) 
    for input,target, mask in train_iter: #TODO for debugging
        loss = train(args, model, input, mask, target, optimizer, criterion)
        current_loss += loss
        
        if (iloop+1) % args.logInterval == 0:
            print("%d %.4f" % (iloop+1, current_loss/args.logInterval))
            all_losses.append(current_loss /args.logInterval)
            current_loss=0
            

        iloop+=1


def train(args, model, input, mask, target, optimizer, criterion):
    model = model.train()
    loss_matrix = []
    optimizer.zero_grad()

    output, hidden = model(input)
    
    for t in range(input.size(0)):
        #import pdb; pdb.set_trace()
        loss = criterion(output[t].view(args.batch_size, -1),
                        target[0][t])
        loss_matrix.append(loss.view(1,-1))
    #import pdb; pdb.set_trace()
    loss_matrix = torch.cat(loss_matrix, dim=0)

    masked = loss_matrix * mask

    loss = torch.sum(masked) / torch.sum(mask)

    loss.backward()

    optimizer.step()

    return loss.item()



def evaluate(args, model, input, mask, target, criterion):
    loss_matrix = []
    acc_matrix = []
    recall_matrix = []
    f1_matrix = []
    precision_matrix = []

    daylen = args.daytolook
    output, hidden = model(input)
    
    #input : daylen * batchsize

    '''Part of loss'''
    for t in range(input.size(0)):
        loss = criterion(output[t].view(args.batch_size,-1),
                    target[0][t])

        loss_matrix.append(loss.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)

    masked = loss_matrix * mask
    lossPerDay = torch.sum(masked, dim = 1)/torch.sum(mask, dim=1 ) #1*daylen
    loss = torch.sum(masked[:daylen]) / torch.sum(mask[:daylen])

    '''Part of accuracy'''
    #import pdb; pdb.set_trace()
    for t in range(input.size(0)):
        #import pdb; pdb.set_trace()
        result = torch.max(output[t].data, 1)[1]
        accuracy = (target.squeeze()[t] == result)
        acc_matrix.append((accuracy).view(1,-1))
        # if(t==0):
        #     print(torch.max(output[t].data, 1)[1])
        #     print((target.squeeze()[t]))

    

    acc_matrix = torch.cat(acc_matrix, dim=0)
    #import pdb; pdb.set_trace()
    
    masked_acc = acc_matrix * mask
    accPerDay = torch.sum(masked_acc, dim =1)/torch.sum(mask, dim=1) #TODO actually, don't need mask
    accuracy = torch.sum(masked_acc[:daylen])/torch.sum(mask[:daylen])
    
    '''Part of recall'''
    for t in range(daylen):
        try:
            #import pdb; pdb.set_trace()
            result = torch.max(output[t].data,1)[1]
            recall = recall_score(target.squeeze()[t].cpu(), result.cpu(), average='macro')    
            precision = precision_score(target.squeeze()[t].cpu(), result.cpu(), average='macro')
            f1 = f1_score(target.squeeze()[t].cpu(), result.cpu(), average='macro')       

            recall_matrix.append(recall)    
            precision_matrix.append(precision)    
            f1_matrix.append(f1)         
        except:
            import pdb; pdb.set_trace()
    #recall_matirx = torch.cat(recall_matrix, dim=0)

        
        
    

    return  f1_matrix, precision_matrix, recall_matrix, accPerDay, accuracy.item(), lossPerDay, loss.item()






def test(args, model, test_path, criterion):
    current_loss = 0
    current_acc =0
    current_recall =0 
    
    lossPerDays = []
    accPerDays = []
    recallPerDays = []
    precisionPerDays = []
    f1PerDays = []

    lossPerDays_avg = []
    accPerDays_avg = []
    recallPerDays_avg = []
    precisionPerDays_avg = []
    f1PerDays_avg = []

    model = model.eval()

    daylen = args.daytolook
    with torch.no_grad():
        iloop =0
        test_iter = FSIterator(test_path, args.batch_size, 1)
        for input, target, mask in test_iter:

            f1PerDay, precisionPerDay, recallPerDay, accPerDay, acc, lossPerDay, loss = evaluate(args, model, input, mask, target, criterion)
            lossPerDays.append(lossPerDay[:daylen]) #n_batches * 10
            accPerDays.append(accPerDay[:daylen])
            recallPerDays.append(recallPerDay)
            precisionPerDays.append(precisionPerDay)
            f1PerDays.append(f1PerDay)

            current_acc += acc
            current_loss += loss
            iloop+=1
        
        
        lossPerDays = torch.stack(lossPerDays)
        lossPerDays_avg = (lossPerDays.sum(dim =0))/iloop

        accPerDays = torch.stack(accPerDays)
        accPerDays_avg = (accPerDays.sum(dim = 0))/iloop

        #recallPerDays = torch.stack(recallPerDays)
        recallPerDays_avg = torch.FloatTensor(recallPerDays).sum(dim=0)/iloop
        precisionPerDays_avg = torch.FloatTensor(precisionPerDays).sum(dim=0)/iloop
        f1PerDays_avg = torch.FloatTensor(f1PerDays).sum(dim=0)/iloop
        
        
        #lossPerDays_avg = lossPerDays_avg/iloop
        #accPerDays_avg = accPerDays_avg/iloop

        current_acc = current_acc/iloop
        current_loss = current_loss/iloop

    return  precisionPerDays_avg, f1PerDays_avg, recallPerDays_avg, accPerDays_avg, current_acc, lossPerDays_avg, current_loss
