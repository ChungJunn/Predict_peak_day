import numpy as np
import torch

class FSIterator:
    def __init__(self, filename, batch_size=32, just_epoch=False):
        self.batch_size = batch_size
        self.just_epoch = just_epoch       
 
        self.end = open(filename + "/END.csv", 'r')
        self.h_e = open(filename + "/H_E.csv", 'r')
        self.high = open(filename + "/HIGH.csv", 'r')
        self.l_e = open(filename + "/L_E.csv", 'r')
        self.low = open(filename + "/LOW.csv", 'r')
        self.ma5 = open(filename + "/MA5.csv", 'r')
        self.ma10 = open(filename + "/MA10.csv", 'r')
        self.s_e = open(filename + "/S_E.csv", 'r')
        self.start = open(filename + "/START.csv", 'r')
        self.trade = open(filename + "/TRADE.csv", 'r')
        self.turnover = open(filename + "/TURNOVER.csv", 'r')


        self.fps = [self.end, self.h_e, self.high, self.l_e, self.low, self.ma5, self.ma10, self.s_e, self.start, self.trade, self.turnover]


    def __iter__(self):
        return self

    def reset(self):
        for fp in self.fps:
            fp.seek(0)

    def __next__(self):
        import pdb; pdb.set_trace()
        bat_seq = []

        touch_end = 0

        while(len(bat_seq)< self.batch_size):
            split_seq = []
            pre_seq = []
        
        
            for i in range (0,len(self.fps)):
                pre_seq.append([self.fps[i].readline()])

            if touch_end:
                raise StopIteration

            if len(pre_seq[0]) == 0:
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
                for i in range (0,len(self.fps)):
                    pre_seq.append([self.fps[i].readline()])
            

            for i in range (0,len(self.fps)):
                split_seq.append([float(s) for s in pre_seq[i][0].split(',')])

            #if(np.count_nonzero(~np.isnan(seq_end))>7 and seq_end[-1] == 1):
            #if(np.count_nonzero(~np.isnan(seq_end)) >= 10 and np.count_nonzero(~np.isnan(seq_end)) < 21): # short data
            #if(np.count_nonzero(~np.isnan(seq_end)) >= 21): # long data
                #if(np.count_nonzero(~np.isnan(seq_f))>4):
            if(~(np.isnan(split_seq).any())):
                import pdb; pdb.set_trace()

            seqs = split_seq
            bat_seq.append(seqs)
        
        
        x_data, y_data, mask_data = self.prepare_data(np.array(bat_seq))#B x [[E*daylen],[S*daylen],[L*daylen],[H*daylen]]
        import pdb; pdb.set_trace()
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
        import pdb; pdb.set_trace()
        for l in seq_end_x:
            temp = [(np.argmax(l) - i) for i in range(0, len(l))]
            temp = [i if i >= 0 else 0 for i in temp]
            temp = [i if i < 30 else 29 for i in temp]

            target.append(temp)
            
        return np.array(target)
    
    def prepare_data(self, seq):
        PRE_STEP = 1 # this is for delta
        import pdb; pdb.set_trace()
        seq_x = []
        seq_delta = []
        x_data = []

        for i in range(0,len(self.fps)):
            seq_x.append(seq[:,i,:-1]) #append each factors of all batches # B, Factors, -1


        seq_y =  self.make_target(seq)
        

        # resize into the longest day length
        for i in range(0,len(self.fps)):
            seq_x[i] = self.trimBatch(seq_x[i])

        seq_mask = self.getMask(seq_x[0][:,1:-PRE_STEP])


        for i in range(0,len(self.fps)):
            seq_x[i] = np.nan_to_num(seq_x[i])


        for i in range(0,len(self.fps)):

            seq_delta.append(seq_x[i][:,1:] - seq_x[i][:,:-1])

        
        try : 
            for i in range(0,len(self.fps)):

                x_data.append(seq_x[i][:,1:-PRE_STEP])
                x_data.append(seq_delta[i][:,:-PRE_STEP]) # factors, batch_Size, daylen
 
            '''
            x_data = np.stack([seq_end_x[:,1:-PRE_STEP], seq_end_x_delta[:,:-PRE_STEP],
                           seq_start_x[:,1:-PRE_STEP], seq_start_x_delta[:,:-PRE_STEP],
                           seq_low_x[:,1:-PRE_STEP], seq_low_x_delta[:,:-PRE_STEP],
                           seq_high_x[:,1:-PRE_STEP], seq_high_x_delta[:,:-PRE_STEP]], axis=2) #batch * daylen * inputdim(2)
            '''
        except:
            print("error")
            import pdb; pdb.set_trace()

    # factors, batch, daylen
        #x_data = np.array(x_data).transpose(1,0,2) # daylen * batch * inputdim
        x_data = np.array(x_data).transpose(2,1,0)
        #y_data = seq_y.reshape(1,-1) # batch * 1
        y_data = np.stack([seq_y.transpose(1,0)])# 1*batch*datlen...?????!

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
    filename = "../data/0412/regression/train"
    #df_train = pd.read_csv("../data/dummy/classification_train.csv")
    bs = 4
    train_iter = FSIterator(filename, batch_size=bs, just_epoch=True)

    for input, target, mask in train_iter: 
        print(input)
        print(input.shape)
        print(target.shape)
        print(mask.shape)



        break

                    
