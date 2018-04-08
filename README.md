# PointCNN
PyTorch implementation of PointCNN model specified in the white paper located here: https://arxiv.org/pdf/1801.07791.pdf

Current MNIST accuracy: ~96%

My coding style is somewhat unique, but ultimately geared towards maximal
readability. Along with extensive documentation in the code, I use type 
annotations and code comments indicating input/outputs shapes.
(x,y,z) just indicate that any value is accepted at runtime.

WARNING: Code is almost correct, but I still have to throw in a few details
to make it easy to access as an external library, as well as test. You can feel
free to use whatever is available right now, but I'll likely have the kinks ironed
out by Monday evening.
