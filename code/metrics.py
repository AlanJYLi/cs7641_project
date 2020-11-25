import os
import matplotlib.pyplot as plt
import torch.utils
from torch.autograd import Variable
from image_loader import ImageLoader
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import numpy as np


class Metrics():
    def __init__(self, model, model_dir, model_name, data_dir, batch_size=64, shuffle=True):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = model
        self.test_dataset = ImageLoader(data_dir, split='test')
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        
    def predict_results(self, output):
        '''
        Predict on the test set
        '''
        return torch.argmax(output, dim=1)
        
    def metrics(self):
        check_point=torch.load(os.path.join(self.model_dir,self.model_name), map_location=torch.device('cuda'))
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()
        
        num_examples = 0
        num_correct = 0
        loss = 0
        top5 = 0
        yarr=np.array([])
        truelabel=np.array([])
        
        for _, batch in enumerate(self.test_loader):
            input_data, target_data = Variable(batch[0]), Variable(batch[1])
            
            output_data = self.model(input_data)
            num_examples += input_data.shape[0]
            
            truelabel = np.append(truelabel, target_data.cpu().numpy())
            
            predicted_labels = self.predict_results(output_data)
            yarr = np.append(yarr, predicted_labels.cpu().numpy())
            
            num_correct += torch.sum(predicted_labels == target_data).item()
            top5opt = Variable(output_data)
            _, maxk = torch.topk(top5opt, 5, -1)
            test_labels = target_data.view(-1,1)
            top5 += (test_labels == maxk).sum().item()
            
        Acc = float(num_correct/num_examples)
        top5acc= float(top5/num_examples)
        
        print('Accuracy:{:.4f}, Top5Accuracy:{:.4f}'.format(Acc,top5acc))
        
        precision, recall, fscore, _= score(truelabel, yarr, average='macro')

        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
       
        df = pd.DataFrame({'prediction': yarr, 'Truelabel': truelabel})
        df.to_csv(self.model_name+'.csv')
       
        #return Acc,top5acc, precision, recall, fscore 