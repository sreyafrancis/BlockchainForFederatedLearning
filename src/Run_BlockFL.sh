#!/usr/bin/env bash

clear

echo "Start miner with genesis block"
gnome-terminal -e "python3 miner.py -g 1 -l 2"

sleep 3

for i in `seq 0 9`;
        do
                echo "Start client $i"
                gnome-terminal -e "python3 client.py -d \"data/federated_data_$i.d\" -e 2"
        done



