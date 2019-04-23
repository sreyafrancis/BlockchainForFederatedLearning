#!/usr/bin/env bash

clear

echo "Creating datasets for n clients:"

gnome-terminal -e "python3 data/federated_data_extractor.py"

sleep 3

echo "Start federated learning on n clients:"
gnome-terminal -e "python3 miner.py -g 1 -l 2"

sleep 3

for i in `seq 0 1`;
        do
                echo "Start client $i"
                gnome-terminal -e "python3 client.py -d \"data/federated_data_$i.d\" -e 1"
        done

sleep 3

gnome-terminal -e "python3 create_csv.py"
