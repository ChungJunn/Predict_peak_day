import numpy as np
import torch

class FSIterator:
    def __init__(self, filename, batch_size=32, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch        
        self.fp_end = open(filename + "/END.csv", 'r')
        self.fp_he = open(filename + "/H_E.csv", 'r')
        self.fp_high = open(filename + "/HIGH.csv", 'r')
        self.fp_le = open(filename + "/L_E.csv", 'r')
        self.fp_low = open(filename + "/LOW.csv", 'r')
        self.fp_ma5 = open(filename + "/MA5.csv", 'r')
        self.fp_ma10 = open(filename + "/MA10.csv", 'r')
        self.fp_se = open(filename + "/S_E.csv", 'r')
        self.fp_start = open(filename + "/START.csv", 'r')
        self.fp_trade = open(filename + "/TRADE.csv", 'r')
        self.fp_turnover = open(filename + "/TURNOVER.csv", 'r')


        self.fps = [self.fp_end, self.fp_he,self.fp_high, self.fp_le,
        self.fp_low, self.fp_ma5, self.fp_ma10, self.fp_se, self.fp_start,
        self.fp_trade, self.fp_turnover ]

    def __iter__(self):
        return self

    def reset(self):
        for fp in self.fps:
            fp.seek(0)

    def __next__(self):
 
        bat_seq = []
        touch_end = 0

        while(len(bat_seq)< self.batch_size):
            seq_end = self.fps[0].readline()
            seq_he = self.fps[1].readline()
            seq_high = self.fps[2].readline()
            seq_le = self.fps[3].readline()
            seq_low =self.fps[4].readline()
            seq_ma5 = self.fps[5].readline()
            seq_ma10 = self.fps[6].readline()
            seq_se = self.fps[7].readline()
            seq_start = self.fps[8].readline()
            seq_trade = self.fps[9].readline()
            seq_turnover = self.fps[10].readline()
                
            if touch_end:
                raise StopIteration

            if seq_end == "":
                print("touch end")
                touch_end = 1

                '''
                if self.just_epoch:
                    end_of_data = 1
                    if self.batch_size==1:
                        raise StopIteration
                    else:
                        break
                '''                                                                                                                                                                                                                                                                                                                                              
                self.reset()
                # read the first line

                seq_end = self.fps[0].readline()
                seq_he = self.fps[1].readline()
                seq_high = self.fps[2].readline()
                seq_le = self.fps[3].readline()
                seq_low =self.fps[4].readline()
                seq_ma5 = self.fps[5].readline()
                seq_ma10 = self.fps[6].readline()
                seq_se = self.fps[7].readline()
                seq_start = self.fps[8].readline()
                seq_trade = self.fps[9].readline()
                seq_turnover = self.fps[10].readline()



            seq_end =  [float(s) for s in seq_end.split(',')]
            seq_he =  [float(s) for s in seq_he.split(',')]
            seq_high =  [float(s) for s in seq_high.split(',')]
            seq_le =  [float(s) for s in seq_le.split(',')]
            seq_low = [float(s) for s in seq_low.split(',')]
            seq_ma5 =  [float(s) for s in seq_ma5.split(',')]
            seq_ma10 =  [float(s) for s in seq_ma10.split(',')]
            seq_se =  [float(s) for s in seq_se.split(',')]
            seq_start =  [float(s) for s in seq_start.split(',')]
            seq_trade =  [float(s) for s in seq_trade.split(',')]
            seq_turnover =  [float(s) for s in seq_turnover.split(',')]                

            #if(np.count_nonzero(~np.isnan(seq_end))>7 and seq_end[-1] == 1):
            #if(np.count_nonzero(~np.isnan(seq_end)) >= 10 and np.count_nonzero(~np.isnan(seq_end)) < 21): # short data
            #if(np.count_nonzero(~np.isnan(seq_end)) >= 21): # long data
                #if(np.count_nonzero(~np.isnan(seq_f))>4):

            if(sum(~np.isnan(seq_end)) == sum(~np.isnan(seq_start)) == sum(~np.isnan(seq_low))== sum(~np.isnan(seq_high))):
                if(max(seq_end)<=2):
                    seqs = [seq_end, seq_he, seq_high, seq_high, seq_le, seq_low, seq_ma5,
                    seq_ma10, seq_se, seq_start, seq_start, seq_trade, seq_turnover]
                    bat_seq.append(seqs)
                
        x_data, y_data, mask_data = self.prepare_data(np.array(bat_seq))#B x [[E*daylen],[S*daylen],[L*daylen],[H*daylen]]
        
        device = torch.device("cuda")
        x_data = torch.tensor(x_data).type(torch.float32).to(device)
        y_data = torch.tensor(y_data).type(torch.LongTensor).to(device)
        mask_data = torch.tensor(mask_data).type(torch.float32).to(device)

        return x_data, y_data, mask_data

    def getSeq_len(self,row):
        '''                                                                                                                                 
        returns: count of non-nans (integer)
        adopted from: M4rtni's answer in stackexchange
        '''
        return np.count_nonzero(~np.isnan(row))


    def getMask(self,batch):
        '''
        returns: boolean array indicating whether nans
        '''
        return (~np.isnan(batch)).astype(np.int32)

    def trimBatch(self, batch):
        '''
        args: npndarray of a batch (bsz, n_features)
        returns: trimmed npndarray of a batch.
        '''
        max_seq_len = 0
        for n in range(batch.shape[0]):
            max_seq_len = max(max_seq_len, self.getSeq_len(batch[n]))

        if max_seq_len == 0:
            print("error in trimBatch()")
            sys.exit(-1)

        batch = batch[:,:max_seq_len]
        return batch

    def make_target(self,seq):
        target = []
        temp = []
        seq_end_x = seq[:,0,:-1]
        #import pdb; pdb.set_trace()
        for l in seq_end_x:
            temp = [(np.argmax(l) - i) for i in range(0, len(l))]
            temp = [i if i >= 0 else 0 for i in temp]
            temp = [i if i < 30 else 29 for i in temp]

            target.append(temp)
            
        return np.array(target)
    
    def prepare_data(self, seq):
        PRE_STEP = 1 # this is for delta
        #import pdb; pdb.set_trace()
        seq_end_x = self.trimBatch(seq[:,0,:-1])
        seq_he_x = self.trimBatch(seq[:,1,:-1])
        seq_high_x = self.trimBatch(seq[:,2,:-1])
        seq_le_x = self.trimBatch(seq[:,3,:-1])
        seq_low_x = self.trimBatch(seq[:,4,:-1])
        seq_ma5_x = self.trimBatch(seq[:,5,:-1])
        seq_ma10_x = self.trimBatch(seq[:,6,:-1])
        seq_se_x = self.trimBatch(seq[:,7,:-1])
        seq_start_x = self.trimBatch(seq[:,8,:-1])
        seq_trade_x = self.trimBatch(seq[:,9,:-1])
        seq_turnover_x = self.trimBatch(seq[:,10,:-1])

        
       
        seq_y =  self.make_target(seq)
        

        # resize into the longest day length

        '''
        seq_end_x = self.trimBatch(seq_end_x)
        seq_start_x = self.trimBatch(seq_start_x)
        seq_low_x = self.trimBatch(seq_low_x)
        seq_high_x = self.trimBatch(seq_high_x)
        '''
        
        seq_mask = self.getMask(seq_end_x[:,1:-PRE_STEP])
        
        seq_end_x = np.nan_to_num(seq_end_x)
        seq_start_x = np.nan_to_num(seq_start_x)
        seq_low_x = np.nan_to_num(seq_low_x)
        seq_high_x = np.nan_to_num(seq_high_x)

        seq_end_x = np.nan_to_num(seq_end_x)
        seq_he_x = np.nan_to_num(seq_he_x)
        seq_high_x = np.nan_to_num(seq_high_x)
        seq_le_x = np.nan_to_num(seq_le_x)
        seq_low_x = np.nan_to_num(seq_low_x)
        seq_ma5_x = np.nan_to_num(seq_ma5_x)
        seq_ma10_x = np.nan_to_num(seq_ma10_x)
        seq_se_x = np.nan_to_num(seq_se_x)
        seq_start_x = np.nan_to_num(seq_start_x)
        seq_trade_x = np.nan_to_num(seq_trade_x)
        seq_turnover_x = np.nan_to_num(seq_turnover_x)



        seq_end_x_delta = seq_end_x[:,1:] - seq_end_x[:,:-1]
        seq_start_x_delta = seq_start_x[:,1:] - seq_start_x[:,:-1]
        seq_low_x_delta = seq_low_x[:,1:] - seq_low_x[:,:-1]
        seq_high_x_delta = seq_high_x[:,1:] - seq_high_x[:,:-1]


        seq_end_xd =  seq_end_x[:,1:] - seq_end_x[:,:-1]
        seq_he_xd =  seq_he_x[:,1:] - seq_he_x[:,:-1]
        seq_high_xd =  seq_high_x[:,1:] - seq_high_x[:,:-1]
        seq_le_xd =  seq_le_x[:,1:] - seq_le_x[:,:-1]
        seq_low_xd =  seq_low_x[:,1:] - seq_low_x[:,:-1]
        seq_ma5_xd =  seq_ma5_x[:,1:] - seq_ma5_x[:,:-1]
        seq_ma10_xd =  seq_ma10_x[:,1:] - seq_ma10_x[:,:-1]
        seq_se_xd =  seq_se_x[:,1:] - seq_se_x[:,:-1]
        seq_start_xd =  seq_start_x[:,1:] - seq_start_x[:,:-1]
        seq_trade_xd =  seq_trade_x[:,1:] - seq_trade_x[:,:-1]
        seq_turnover_xd = seq_turnover_x[:,1:] - seq_turnover_x[:,:-1]








        try : 
            x_data = np.stack([seq_end_x[:,1:-PRE_STEP], seq_end_xd[:,:-PRE_STEP],
                           seq_he_x[:,1:-PRE_STEP], seq_he_xd[:,:-PRE_STEP],
                           seq_high_x[:,1:-PRE_STEP], seq_high_xd[:,:-PRE_STEP],
                           seq_le_x[:,1:-PRE_STEP], seq_le_xd[:,:-PRE_STEP],
                           seq_low_x[:,1:-PRE_STEP], seq_low_xd[:,:-PRE_STEP],
                           seq_ma5_x[:,1:-PRE_STEP], seq_ma5_xd[:,:-PRE_STEP],
                           seq_ma10_x[:,1:-PRE_STEP], seq_ma10_xd[:,:-PRE_STEP],
                           seq_se_x[:,1:-PRE_STEP], seq_se_xd[:,:-PRE_STEP],
                           seq_start_x[:,1:-PRE_STEP], seq_start_xd[:,:-PRE_STEP],
                           seq_trade_x[:,1:-PRE_STEP], seq_trade_xd[:,:-PRE_STEP],
                           seq_turnover_x[:,1:-PRE_STEP], seq_turnover_xd[:,:-PRE_STEP]], axis=2) #batch * daylen * inputdim(2)

        except:
            import pdb; pdb.set_trace()
        x_data = x_data.transpose(1,0,2) # daylen * batch * inputdim
        
        #y_data = seq_y.reshape(1,-1) # batch * 1
        y_data = np.stack([seq_y.transpose(1,0)])# 1*batch*1

        #y_data = (seq_delta[:,1:] > 0)*1.0 # the diff
        
        mask_data = np.stack(seq_mask.transpose(1,0))
        '''
        x_data : daymaxlen-2, batch, inputdim(=2)
        y_data : 1 * batch * 1
        mask_data : 1*daymaxlen-2, batch
        '''
        return x_data, y_data, mask_data

if __name__ == "__main__":
    import os
    import numpy as np

    #filename = os.environ['HOME']+'/FinSet/data/GM.csv.seq.shuf'
    filename = "../data/dummy/classification_train.csv"
    #df_train = pd.read_csv("../data/dummy/classification_train.csv")
    bs = 4
    train_iter = FSIterator(filename, batch_size=bs, just_epoch=True)

    i = 0
    for tr_x, tr_y, tr_m, end_of_data in train_iter:
        print(i, tr_x, tr_y, tr_m)
        i = i + 1
        if i > 2:
            break