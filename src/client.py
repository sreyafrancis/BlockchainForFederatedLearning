"""
  - Blockchain for Federated Learning - 
    Client script 
"""
import tensorflow as tf
import pickle
from federatedlearner import *
from blockchain import *
from uuid import uuid4
import requests
import data.federated_data_extractor as dataext
import time

class Client:
    def __init__(self,miner,dataset):
        self.id = str(uuid4()).replace('-','')
        self.miner = miner
        self.dataset = self.load_dataset(dataset)

    def get_last_block(self):
            return self.get_chain()[-1]

    def get_chain(self):
        response = requests.get('http://{node}/chain'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()['chain']

    def get_full_block(self,hblock):
        response = requests.post('http://{node}/block'.format(node=self.miner),
            json={'hblock': hblock})
        if response.json()['valid']:
            return Block.from_string(response.json()['block'])
        print("Invalid block!")
        return None

    def get_model(self,hblock):
        response = requests.post('http://{node}/model'.format(node=self.miner),
            json={'hblock': hblock})
        if response.json()['valid']:
            return dict(pickle.loads(codecs.decode(response.json()['model'].encode(), "base64")))
        print("Invalid model!")
        return None

    def get_miner_status(self):
        response = requests.get('http://{node}/status'.format(node=self.miner))
        if response.status_code == 200:
            return response.json()

    def load_dataset(self,name):

        ''' 
        Function to load federated data for client side training
        '''
        if name==None:
            return None
        return dataext.load_data(name)

    def update_model(self,model,steps):

        ''' 
        Function to compute the client model update based on 
        client side training
        '''
        reset()
        t = time.time()
        worker = NNWorker(self.dataset['train_images'],
            self.dataset['train_labels'],
            self.dataset['test_images'],
            self.dataset['test_labels'],
            len(self.dataset['train_images']),
            self.id,
            steps)
        worker.build(model)
        worker.train()
        update = worker.get_model()
        accuracy = worker.evaluate()
        worker.close()
        return update,accuracy,time.time()-t

    def send_update(self,update,cmp_time,baseindex):

        ''' 
        Function to post client update details to blockchain
        '''
	
        requests.post('http://{node}/transactions/new'.format(node=self.miner),
            json = {
                'client': self.id,
                'baseindex': baseindex,
                'update': codecs.encode(pickle.dumps(sorted(update.items())), "base64").decode(),
                'datasize': len(self.dataset['train_images']),
                'computing_time': cmp_time
            })
       
    def work(self,device_id,elimit):
        ''' 
        Function to check the status of mining and wait accordingly to output 
        the final accuracy values 
        '''
        last_model = -1
        
        for i in range(elimit):
            wait = True
            while wait:
                status = client.get_miner_status()
                if status['status']!="receiving" or last_model==status['last_model_index']:
                    time.sleep(10)
                    print("waiting")
                else:
                    wait = False
            hblock = client.get_last_block()
            baseindex = hblock['index']
            print("Accuracy global model",hblock['accuracy'])
            last_model = baseindex
            model = client.get_model(hblock)
            update,accuracy,cmp_time = client.update_model(model,10)
            with open("clients/device"+str(device_id)+"_model_v"+str(i)+".block","wb") as f:
                pickle.dump(update,f)
		#j = j+1
            print("Accuracy local update---------"+str(device_id)+"--------------:",accuracy)
            client.send_update(update,cmp_time,baseindex)
            

if __name__ == '__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-m', '--miner', default='127.0.0.1:5000', help='Address of miner')
    parser.add_argument('-d', '--dataset', default='data/mnist.d', help='Path to dataset')
    parser.add_argument('-e', '--epochs', default=10,type=int, help='Number of epochs')
    args = parser.parse_args()
    client = Client(args.miner,args.dataset)
    print("--------------")
    print(client.id," Dataset info:")
    Data_size, Number_of_classes = dataext.get_dataset_details(client.dataset)
    print("--------------")
    device_id = client.id[:2]
    print(device_id,"device_id")
    print("--------------")
    client.work(device_id, args.epochs)
