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
import data.extractor as dataext
import numpy as np
from nn import *

class Block:
    def __init__(self,miner,index,basemodel,accuracy,updates,timestamp=time.time()):
        self.index = index
        self.miner = miner
        self.timestamp = timestamp
        self.basemodel = basemodel
        self.accuracy = accuracy
        self.updates = updates

    @staticmethod
    def from_string(updstr):
        i,l = find_len(updstr,"'timestamp':")
        i2,l2 = find_len(updstr,"'basemodel': ")
        i3,l3 = find_len(updstr,"'index':")
        i4,l4 = find_len(updstr,"'miner':")
        i5,l5 = find_len(updstr,"'accuracy':")
        i6,l6 = find_len(updstr,"'updates':")
        i9,l9 = find_len(updstr,"'updates_size':")
        index = int(updstr[i3+l3:i4].replace(",",'').replace(" ",""))
        miner = updstr[i4+l4:i].replace(",",'').replace(" ","")
        timestamp = float(updstr[i+l:i2].replace(",",'').replace(" ",""))
        basemodel = dict(pickle.loads(codecs.decode(updstr[i2+l2:i5-1].encode(), "base64")))
        accuracy = float(updstr[i5+l5:i6].replace(",",'').replace(" ",""))
        su = updstr[i6+l6:i9]
        su = su[:su.rfind("]")+1]
        updates = dict()
        for x in json.loads(su):
            isep,lsep = find_len(x,"@|!|@")
            # print(x[:isep])
            updates[x[:isep]] = Update.from_string(x[isep+lsep:])
        updates_size = int(updstr[i9+l9:].replace(",",'').replace(" ",""))
        return Block(miner,index,basemodel,accuracy,updates,timestamp)

    def __str__(self):
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
        # Create the genesis block
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
        # print(self.nodes)

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
        # print(accuracy)
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
            with open("blocks/b"+str(self.curblock.index)+".block","wb") as f:
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
                # print(node,length,max_length,self.valid_chain(chain))
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
