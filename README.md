# vigilant-carnival
# two convolutional layers of kernel width 16: 8 filters and 16 filters
# two maxPool layers in between of stride and poolsize 2
# conv output flattened, and subtracted and then fed to an LSTM
# LSTM: 48 hidden units, output is weighted, added with bias and tanh-ed
# This value is sign-compared with class value, and used to train by reducing entropy
