import glob
import csv

path = '/home/user/Documents/src/clients/'
n_devices = 2
Model_version = 0
l = len(path)
i = 1

Device_gradient_path = glob.glob(path+"*.block")
print (Device_gradient_path)

client_details = dict()

for gradient in Device_gradient_path:
    client_details[gradient] = gradient[ l+5 : l+7]

csvData = [['Device_id', 'Device_delta_path', 'Model_version']]
csvData = [[] for _ in range(n_devices+1)]
csvData[0] = ['Device_id', 'Device_delta_path', 'Model_version']

for k,v in client_details.items():
    csvData[i].append(v)
    csvData[i].append(k)
    csvData[i].append(Model_version)
    i += 1

#print (csvData)

with open('DeltaOffChainDatabase.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)

csvFile.close()
    
