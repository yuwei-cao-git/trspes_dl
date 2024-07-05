# trspes_dl
Tree Species Composition Estimation with Deep Learning and Cloud Computing

# record
[x] pn2_noPA_7168_Noaug_DP
run 1:
32 batch size: 40%-50% GPU power, gpu memory allocated: 30%; 
lr: 1e-4
run 2:
96 batch size

[x] pn2_noPA_7168_Aug2_DP
run 1:
96 bs, 1e-4
run 2:
96 bs, 5e-4
run 3:
96 bs, 5e-4, loss function with torch.sum()

[ ] pn2_PA_7168_NOAUG_DDP

[x] dgcnn_noPA_7168_Aug2_DDP
20 bs:

out: cuda:1
bn1: 2, bn2: 2, iden: 1, fc2: 1, fc3: 1
bn1: 2, bn2: 2, bn3: 2, conv1: 1, conv2: 1, conv3: 1
bn1: 2, bn2: 2, iden: 1, fc2: 1, fc3: 1, bn4: 2, bn4: 2, conv4: 1, rot: 1
dgcnn bn1: 2, bn2: 2, bn3: 2, bn4: 2, bn4: 2, bn4: 2, conv4: 1, conv4: 1, conv4: 1, conv4: 1,conv4: 1, bn7: 2, slef.linear2: 1, slef.linear2: 1, slef.linear2: 1, x1: 0, x4: 0, x: 0
dgcnn bn1: 2, bn2: 2, bn3: 2, bn4: 2, bn4: 2, bn4: 2, conv4: 1, conv4: 1, conv4: 1, conv4: 1,conv4: 1, bn7: 2, slef.linear2: 1, slef.linear2: 1, slef.linear2: 1, x1: 0, x4: 0, x: 0
y_loss: 0
aug_y_loss: 0
loss: 0
Epoch: 1, weights_version: 0, label_device: 0,  loss: 0, augmentor: 1, classifier: 1, out_aug[rank1]: Traceback (most recent call last):         
[rank1]:   File "/home/ycao68/code/trspes_dl/Pytorch/models/PointAugment/main_ddp.py", line 108, in <module>
[rank1]:     main(params)
[rank1]:   File "/home/ycao68/code/trspes_dl/Pytorch/models/PointAugment/main_ddp.py", line 62, in main
[rank1]:     train(params, io, trainset, testset)
[rank1]:   File "/home/ycao68/code/trspes_dl/Pytorch/models/PointAugment/utils/train_ddp.py", line 208, in train
[rank1]:     aug_loss.backward()
[rank1]:   File "/home/ycao68/venv/lib/python3.10/site-packages/torch/_tensor.py", line 525, in backward
[rank1]:     torch.autograd.backward(
[rank1]:   File "/home/ycao68/venv/lib/python3.10/site-packages/torch/autograd/__init__.py", line 267, in backward
[rank1]:     _engine_run_backward(
[rank1]:   File "/home/ycao68/venv/lib/python3.10/site-packages/torch/autograd/graph.py", line 744, in _engine_run_backward
[rank1]:     return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
[rank1]: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [256]] is at version 3; expected version 2 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!

focus on the order of optimizer, classifier, augmentor...