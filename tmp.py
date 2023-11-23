import torch
from torch.nn import functional as F

# for a in range(10):
#     print(a)
#     z=torch.zeros(1,1000)
#     z=z-a
#     # print(z)
#     # print(z.shape)

#     z[0,0]=a

#     # print(z)

#     z=F.softmax(z)

#     print(z[0,:2])
#     print(torch.sum(z))


import datasets


data = datasets.load_dataset("wmt14", "de-en", cache_dir="../../dataset/WMT")

datalist = []
for i in data["test"]:
    datalist.append(i["translation"]["en"]+"\n")
    
with open("tmp.txt","w") as fp:
    for i in datalist:
        fp.write(i)