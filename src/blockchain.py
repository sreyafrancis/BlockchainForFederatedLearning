"""
 - Blockchain for Federated Learning -
   Blockchain script 
"""

import hashlib
import json
import time
from flask import Flask,jsonify,request
from uuid import uuid4
from urllib.parse import urlparse
import requests
import random
from threading import Thread, Event
import pickle
import codecs
import data.federated_data_extractor as dataext
import numpy as np
from federatedlearner import *


def compute_global_model(base,updates,lrate):

    '''
    Function to compute the global model based on the client 
    updates received per round
    '''

    upd = dict()
    for x in ['w1','w2','wo','b1','b2','bo']:
        upd[x] = np.array(base[x], copy=True)
    number_of_clients = len(updates)
    for client in updates.keys():
        for x in ['w1','w2','wo','b1','b2','bo']:
            model = updates[client].update
            upd[x] += (lrate/number_of_clients)*(model[x]+base[x])
    upd["size"] = 0
    reset()
    dataset = dataext.load_data("data/mnist.d")
    worker = NNWorker(None,
        None,
        dataset['test_images'],
        dataset['test_labels'],
        0,
        "validation")
    worker.build(upd)
    accuracy = worker.evaluate()
    worker.close()
    return accuracy,upd

def find_len(text,strk):

    ''' 
    Function to find the specified string in the text and return its starting position 
    as well as length/last_index
    '''
    return text.find(strk),len(strk)

class Update:
    def __init__(self,client,baseindex,update,datasize,computing_time,timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters
        '''
        self.timestamp = timestamp
        self.baseindex = baseindex
        self.update = update
        self.client = client
        self.datasize = datasize
        self.computing_time = computing_time

    @staticmethod
    def from_string(metadata):

        ''' 
        Function to get the update string values
        '''
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'baseindex':")
        i3,l3 = find_len(metadata,"'update': ")
        i4,l4 = find_len(metadata,"'client':")
        i5,l5 = find_len(metadata,"'datasize':")
        i6,l6 = find_len(metadata,"'computing_time':")
        baseindex = int(metadata[i2+l2:i3].replace(",",'').replace(" ",""))
        update = dict(pickle.loads(codecs.decode(metadata[i3+l3:i4-1].encode(), "base64")))
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        client = metadata[i4+l4:i5].replace(",",'').replace(" ","")
        datasize = int(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        computing_time = float(metadata[i6+l6:].replace(",",'').replace(" ",""))
        return Update(client,baseindex,update,datasize,computing_time,timestamp)


    def __str__(self):

        ''' 
        Function to return the update string values in the required format
        '''
        return "'timestamp': {timestamp},\
            'baseindex': {baseindex},\
            'update': {update},\
            'client': {client},\
            'datasize': {datasize},\
            'computing_time': {computing_time}".format(
                timestamp = self.timestamp,
                baseindex = self.baseindex,
                update = codecs.encode(pickle.dumps(sorted(self.update.items())), "base64").decode(),
                client = self.client,
                datasize = self.datasize,
                computing_time = self.computing_time
            )


class Block:
    def __init__(self,miner,index,basemodel,accuracy,updates,timestamp=time.time()):

        ''' 
        Function to initialize the update string parameters per created block
        '''
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(metadata):

        ''' 
        Function to get the update string values per block
        '''
        i,l = find_len(metadata,"'timestamp':")
        i2,l2 = find_len(metadata,"'basemodel': ")
        i3,l3 = find_len(metadata,"'index':")
        i4,l4 = find_len(metadata,"'miner':")
        i5,l5 = find_len(metadata,"'accuracy':")
        i6,l6 = find_len(metadata,"'updates':")
        i9,l9 = find_len(metadata,"'updates_size':")
        index = int(metadata[i3+l3:i4].replace(",",'').replace(" ",""))
        miner = metadata[i4+l4:i].replace(",",'').replace(" ","")
        timestamp = float(metadata[i+l:i2].replace(",",'').replace(" ",""))
        basemodel = dict(pickle.loads(codecs.decode(metadata[i2+l2:i5-1].encode(), "base64")))
        accuracy = float(metadata[i5+l5:i6].replace(",",'').replace(" ",""))
        su = metadata[i6+l6:i9]
        su = su[:su.rfind("]")+1]
        updates = dict()
        for x in json.loads(su):
            isep,lsep = find_len(x,"@|!|@")
            updates[x[:isep]] = Update.from_string(x[isep+lsep:])
        updates_size = int(metadata[i9+l9:].replace(",",'').replace(" ",""))
        return Block(miner,index,basemodel,accuracy,updates,timestamp)

    def __str__(self):

        ''' 
        Function to return the update string values in the required format per block
        '''
        return "'index': {index},\
            'miner': {miner},\
            'timestamp': {timestamp},\
            'basemodel': {basemodel},\
            'accuracy': {accuracy},\
            'updates': {updates},\
            'updates_size': {updates_size}".format(
                index = self.index,
                miner = self.miner,
                basemodel = codecs.encode(pickle.dumps(sorted(self.basemodel.items())), "base64").decode(),
                accuracy = self.accuracy,
                timestamp = self.timestamp,
                updates = str([str(x[0])+"@|!|@"+str(x[1]) for x in sorted(self.updates.items())]),
                updates_size = str(len(self.updates))
            )



class Blockchain(object):
    def __init__(self,miner_id,base_model=None,gen=False,update_limit=10,time_limit=1800):
        super(Blockchain,self).__init__()
        self.miner_id = miner_id
        self.curblock = None
        self.hashchain = []
        self.current_updates = dict()
        self.update_limit = update_limit
        self.time_limit = time_limit
        
        if gen:
            genesis,hgenesis = self.make_block(base_model=base_model,previous_hash=1)
            self.store_block(genesis,hgenesis)
        self.nodes = set()

    def register_node(self,address):
        if address[:4] != "http":
            address = "http://"+address
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)
        print("Registered node",address)

    def make_block(self,previous_hash=None,base_model=None):
        accuracy = 0
        basemodel = None
        time_limit = self.time_limit
        update_limit = self.update_limit
        if len(self.hashchain)>0:
            update_limit = self.last_block['update_limit']
            time_limit = self.last_block['time_limit']
        if previous_hash==None:
            previous_hash = self.hash(str(sorted(self.last_block.items())))
        if base_model!=None:
            accuracy = base_model['accuracy']
            basemodel = base_model['model']
        elif len(self.current_updates)>0:
            base = self.curblock.basemodel
            accuracy,basemodel = compute_global_model(base,self.current_updates,1)
        index = len(self.hashchain)+1
        block = Block(
            miner = self.miner_id,
            index = index,
            basemodel = basemodel,
            accuracy = accuracy,
            updates = self.current_updates
            )
        hashblock = {
            'index':index,
            'hash': self.hash(str(block)),
            'proof': random.randint(0,100000000),
            'previous_hash': previous_hash,
            'miner': self.miner_id,
            'accuracy': str(accuracy),
            'timestamp': time.time(),
            'time_limit': time_limit,
            'update_limit': update_limit,
            'model_hash': self.hash(codecs.encode(pickle.dumps(sorted(block.basemodel.items())), "base64").decode())
            }
        return block,hashblock

    def store_block(self,block,hashblock):
        if self.curblock:
            with open("blocks/federated_model"+str(self.curblock.index)+".block","wb") as f:
                pickle.dump(self.curblock,f)
        self.curblock = block
        self.hashchain.append(hashblock)
        self.current_updates = dict()
        return hashblock

    def new_update(self,client,baseindex,update,datasize,computing_time):
        self.current_updates[client] = Update(
            client = client,
            baseindex = baseindex,
            update = update,
            datasize = datasize,
            computing_time = computing_time
            )
        return self.last_block['index']+1

    @staticmethod
    def hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @property
    def last_block(self):
        return self.hashchain[-1]


    def proof_of_work(self,stop_event):
        block,hblock = self.make_block()
        stopped = False
        while self.valid_proof(str(sorted(hblock.items()))) is False:
            if stop_event.is_set():
                stopped = True
                break
            hblock['proof'] += 1
            if hblock['proof']%1000==0:
                print("mining",hblock['proof'])
        if stopped==False:
            self.store_block(block,hblock)
        if stopped:
            print("Stopped")
        else:
            print("Done")
        return hblock,stopped

    @staticmethod
    def valid_proof(block_data):
        guess_hash = hashlib.sha256(block_data.encode()).hexdigest()
        k = "00000"
        return guess_hash[:len(k)] == k


    def valid_chain(self,hchain):
        last_block = hchain[0]
        curren_index = 1
        while curren_index<len(hchain):
            hblock = hchain[curren_index]
            if hblock['previous_hash'] != self.hash(str(sorted(last_block.items()))):
                print("prev_hash diverso",curren_index)
                return False
            if not self.valid_proof(str(sorted(hblock.items()))):
                print("invalid proof",curren_index)
                return False
            last_block = hblock
            curren_index += 1
        return True

    def resolve_conflicts(self,stop_event):
        neighbours = self.nodes
        new_chain = None
        bnode = None
        max_length = len(self.hashchain)
        for node in neighbours:
            response = requests.get('http://{node}/chain'.format(node=node))
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length>max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
                    bnode = node
        if new_chain:
            stop_event.set()
            self.hashchain = new_chain
            hblock = self.hashchain[-1]
            resp = requests.post('http://{node}/block'.format(node=bnode),
                json={'hblock': hblock})
            self.current_updates = dict()
            if resp.status_code == 200:
                if resp.json()['valid']:
                    self.curblock = Block.from_string(resp.json()['block'])
            return True
        return False
