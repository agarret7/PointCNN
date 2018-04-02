import h5py

f = h5py.File("./mnist/zips/train_0.h5", 'r')

data = f["data"]
label = f["label"]

for r in data[0]:
    print(r)
