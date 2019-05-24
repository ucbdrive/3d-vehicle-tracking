#!/usr/bin/env bash

GPU_ID=0,1,2,3
DATASET="gta_det"
NET="res101"
BATCH_SIZE=8
WORKER_NUMBER=4
LEARNING_RATE=1e-3
DECAY_STEP=10
SESSION=201
EPOCH=20

LOADSESSION=200
LOADEPOCH=14
LOADCHECKPOINT=18895

# If you are training with pretrain model 'faster_rcnn_200_14_18895.pth', uncomment the last three line

CUDA_VISIBLE_DEVICES=${GPU_ID} python trainval_net.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --s ${SESSION} \
    --bs ${BATCH_SIZE} \
    --nw ${WORKER_NUMBER} \
    --lr ${LEARNING_RATE} \
    --lr_decay_step ${DECAY_STEP} \
    --cuda \
    --mGPUs \
    --epochs ${EPOCH} \
    --o adam \
    #--checksession ${LOADSESSION} \
    #--checkepoch ${LOADEPOCH} \
    #--checkpoint ${LOADCHECKPOINT} 

echo $(whoami)" : "${SESSION}_${LOADEPOCH}_${LOADCHECKPOINT}" Finish!"
