'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

import argparse

from data import FSIterator
from model import RNN
from train import train_main, test
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument('--logInterval', type=int, default=100, help='')
parser.add_argument('--saveModel', type=str, default="bestmodel", help='')
parser.add_argument('--savePath', type=str, default="png", help='')
parser.add_argument('--fileName', type=str, default="short", help='')
parser.add_argument('--max_epochs', type=int, default=8, help='')
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--hidden_size', type=int, default=8, help='')
parser.add_argument('--input_size', type=int, default=22, help='')

parser.add_argument('--output_size', type=int, default=20, help='') # if use 0412 data it can be 20
parser.add_argument('--saveDir', type=str, default="png", help='')
parser.add_argument('--patience', type=int, default=5, help='')
parser.add_argument('--daytolook', type=int, default=5, help='')
parser.add_argument('--optim', type=str, default="Adam")  # Adam, SGD, RMSprop
parser.add_argument('--lr', type=float, metavar='LR', default=0.01,
                    help='learning rate (no default)')

args = parser.parse_args()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    # prepare data
    batch_size = args.batch_size
    n_epoches = args.max_epochs

    device = torch.device("cuda")

    # setup model

    input_size = args.input_size
    hidden_size = args.hidden_size
    output_size = args.output_size

    model = RNN(input_size, hidden_size, output_size, batch_size).to(device)

    # define loss
    
    criterion =  nn.CrossEntropyLoss()

    # define optimizer
    optimizer = "optim." + args.optim
    optimizer = eval(optimizer)(model.parameters(), lr=args.lr)

    logInterval = args.logInterval
    current_loss = 0
    all_losses = []

    start = time.time()

    patience = args.patience
    savePath = args.savePath

    #train_path = "../data/dummy/classification_test.csv"
    train_path = "../data/0412/regression/train"
    test_path = "../data/0412/regression/test"
    valid_path = "../data/0412/regression/valid"

    for ei in range(args.max_epochs):
        bad_counter = 0
        best_loss = -1.0

        #train_main(args, model, train_path, criterion, optimizer)

        #dayloss, valid_loss, dayerr, valid_err = test(args, model, valid_path, criterion)
        train_main(args, model, train_path, criterion, optimizer)
        precision, f1, recallPerDays, accPerDays, valid_acc, lossPerDays, valid_loss = test(args, model, valid_path, criterion)
        
        print("valid loss : {}".format(valid_loss))
        print("accuracy")
        print(accPerDays.tolist())
        print("recall")
        print(recallPerDays)
        print("f1")
        print(f1)
        print("precision")
        print(precision)

        #print("valid loss : {}".format(valid_loss))
        #print(dayloss)
        if valid_loss < best_loss or best_loss < 0:
            print("find best")
            best_loss = valid_loss
            bad_counter = 0
            torch.save(model, args.saveModel)
        else:
            bad_counter += 1

        if bad_counter > patience:
            print('Early Stopping')
            break

    print("------------test-----------------")
    #dayloss, valid_loss, dayerr, valid_err  = test(args, model, test_path, criterion)
    precision, f1, recallPerDays, accPerDays, valid_acc, lossPerDays, valid_loss = test(args, model, test_path, criterion)
    '''
    print("valid loss : {}".format(valid_loss))
    print(dayloss)
    
    print("valid error: {}".format(valid_err))
    print(dayerr)
    '''
    print("accuracy ")
    print(accPerDays.tolist())
    print("Recall")
    print(recallPerDays)
    print("F1")
    print(f1)
    print("Precision")
    print(precision)


    #write to csv
    
    import csv
    f = open('peak_result_macro.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(["batch_size: " + str(args.batch_size), " max_epoches: " + str(args.max_epochs), " hidden_size: " + str(args.hidden_size),
                 " patience: " + str(args.patience), " daytolook: " + str(args.daytolook), " optimizer: "+args.optim, " learning_rate: " + str(args.lr)])
    wr.writerow(["accuracy", accPerDays.tolist()])
    wr.writerow(["recall per days", recallPerDays])
    wr.writerow(["f1", f1])
    wr.writerow(["precision", precision])
    f.close()
        
    '''
    result_matrix = pd.DataFrame(
        data={"dayloss": list(dayloss.data), "valid_loss": valid_loss})
    result_matrix.to_csv("test.csv", index=False, header=True)
    '''
    '''
    #draw a plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dataset = [dayloss.tolist(),dayerr.tolist()]
    category = list()
    daylength = []
    for i in range(args.daytolook):
        daylength.append(' ')
    data = {'day': daylength,  args.optim+str(args.lr) : dataset[0],'dayerr': dataset[1]}

    df1 = pd.DataFrame(data=data)
    df2 = pd.DataFrame(data=data)

    ax = plt.gca()

    #draw and save file
    df1.plot(kind='line', x='day', y = args.optim+str(args.lr), ax=ax, color='blue')
    save_path = os.path.join("./", args.saveDir)
    ax.figure.savefig(save_path + "/test_loss.png")

    df2.plot(kind='line', x='day', y = 'dayerr', ax=ax, color='red')
    save_path = os.path.join("./", args.saveDir)
    ax.figure.savefig(save_path + "/test_err.png")
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_path = os.path.join("./", args.saveDir) 
    plt.plot(accPerDays.tolist())
    plt.savefig(savePath + "/" + args.fileName + "acc_peak_macro.png")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_path = os.path.join("./", args.saveDir) 
    plt.plot(f1)
    plt.savefig(savePath + "/" + args.fileName + "f1_peak_macro.png")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_path = os.path.join("./", args.saveDir) 
    plt.plot(recallPerDays)
    plt.savefig(savePath + "/" + args.fileName + "recall_peak_macro.png")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    save_path = os.path.join("./", args.saveDir) 
    plt.plot(precision)
    plt.savefig(savePath + "/" + args.fileName + "precision_peak_macro.png")

    '''
    dataset = dayloss.tolist()

    category = list()
    daylength = []
    for i in range(args.daytolook):
        daylength.append(' ')
    data = {'day': daylength,  args.optim+str(args.lr) : dataset}

    df = pd.DataFrame(data=data)

    ax = plt.gca()
    df.plot(kind='line', x='day', y = args.optim+str(args.lr), ax=ax, color='blue')

    plt.show()

    # save file
    save_path = os.path.join("./", args.saveDir)
    ax.figure.savefig(save_path + "/test.jpg")

    
    train_end_iter = FSIterator(train_path + "/end.csv", batch_size)
    train_start_iter = FSIterator(train_path + "/start.csv", batch_size)
    train_low_iter = FSIterator(train_path + "/low.csv", batch_size)
    train_high_iter = FSIterator(train_path + "/high.csv", batch_size)
    '''

    #train_iter = FSIterator(train_path, batch_size)

    '''
    for input_end, target, mask in train_iter:

        
        input_start, __, __ = train_start_iter.next()
        input_low, __, __ = train_low_iter.next()
        input_high, __, __ = train_high_iter.next()
        output = model(input_end, input_start, input_low, input_high)
        
        output = model(input)

        # draw graph
        
        input = input_end[:, :, 0].transpose(1, 0)
        output = output.squeeze().transpose(1, 0)
        mask = mask.transpose(1, 0)

        for i in range(batch_size):
            daylen = np.count_nonzero(mask[i].cpu())-1
            plt.plot(input[i, 1:daylen+1].cpu())
            plt.plot(output[i, :daylen].detach().cpu())
            plt.savefig(savePath + "/" + args.fileName + str(i) + ".png")
            plt.clf()
        break
        
    '''
