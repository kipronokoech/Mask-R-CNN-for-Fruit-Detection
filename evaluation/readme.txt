fruits220210917T0825 - Pretrained weights: Imagenet, Epoch(s): 275(Best), 380. Learning rate: decaying as follows -> IR=0.001, IR(epochs 1-40), IR/5(epochs 41-80), IR/10 (epochs 81-120), IR/20 (epochs 121-200), IR/40 (epochs 201-280), IR/50 (epochs 281-380). Note: Epoch can be identified from the filename fruits220210917T0825-####ext


fruits220210918T0926 - Pretrained weights: COCO, Epoch(s): 273 (Best, obviously). Learning rate: decaying as follows -> IR=0.001, IR(epochs 1-40), IR/5(epochs 41-80), IR/10 (epochs 81-120), IR/20 (epochs 121-200), IR/40 (epochs 201-280), IR/50 (epochs 281-380). Note: Epoch can be identified from the filename fruits220210917T0825-####ext

fruits220210919T1311 - Pretrained weights: Imagenet, LR: Not decaying. Trained for 300 epochs. Best epoch (based on the val_loss) is 156. The training losses were extracted from the TF event files and analysed to make that determination.

fruits220210919T1323 - Pretrained weights: COCO, LR: Not decaying. Trained for 300 epochs. Best epoch is 103 based on the analysis of training losses (val_loss) across all the epochs.

fruits220210921T1549 - Pretrained weights: ImageNet. New way of implementing step-based decay of LR. LR decay is now defined inside model.py file using Keras LR Scheduler. The LR is set to decay as follows: ir = 0.001 (epoch: 0-80),
ir/5 (epoch: 81-120), ir/10 (epoch: 121-180), ir/20 (epoch 181-220), ir/40 (epoch: above 220). Best epoch: 131 - based on the analysis of the losses saved on the TF event files. LR is part of the data saved on the event files and 
the decay process happened as it was required (Good!!!!!!!). 

fruits220210922T0742: Pretrained weights: COCO. The LR is set to decay as follows: ir = 0.001 (epoch: 0-80),
ir/5 (epoch: 81-120), ir/10 (epoch: 121-180), ir/20 (epoch 181-220), ir/40 (epoch: above 220). Best epoch: 71 - based on teh analysis of the lossed saved on the TensorFlow event files. The loss for the best epoch: 0.4365.
