#!/usr/bin/env bash

GPU_ID=1
PHASE="val" # "test", "train"
DATASET="gta_det"
NET="res101"
BATCH_SIZE=16
WORKER_NUMBER=4
SESSION=200
TEST_EPOCH=14
TEST_CHECKPOINT=18895

CUDA_VISIBLE_DEVICES=${GPU_ID} python test_net.py \
    --dataset ${DATASET} \
    --net ${NET} \
    --checksession ${SESSION} \
    --checkepoch ${TEST_EPOCH} \
    --checkpoint ${TEST_CHECKPOINT} \
    --anno ${PHASE} \
    --cuda

echo 
echo $(whoami)" : "${SESSION}_${TEST_EPOCH}_${TEST_CHECKPOINT}" Finish!"
echo "Copy output pkl file to dataset folder..."
cp \
    vis/faster_rcnn_${SESSION}_${TEST_EPOCH}_${TEST_CHECKPOINT}/detections_${PHASE}.pkl \
    data/gta5_tracking/gta_${PHASE}_detections.pkl
echo "Done!!"
