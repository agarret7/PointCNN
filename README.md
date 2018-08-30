# PointCNN
PyTorch implementation of PointCNN model specified in the white paper located here: https://arxiv.org/pdf/1801.07791.pdf

Current MNIST accuracy: ~98%

```
python ./download_datasets.py -d mnist -f ./
python ./prepare_mnist_data.py -f ./mnist/zips
python ./pointcnn_cls.py
```

Type annotations are liberally used in this project, including annotations
indicating input/outputs shapes. (x,y,z) just indicate that any value is
accepted at runtime.
