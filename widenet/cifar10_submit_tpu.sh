export TPU_NAME=vit32
export DATA_DIR=gs://cifar10/cifar10/3.0.2
export MODEL_DIR=gs://cv_narrownet/widenet-base
export LOG_DIR=gs://vit_exp1/log
#export MODEL_DIR=/../checkpoint
#export LOG_DIR=/../log


python3 train.py --img_size=32 --base-lr=0.001 --batch-size=256 --epoch=100 --wd=0.001 --data_set=Cifar10 --data_dir=$DATA_DIR --model_type=ViT-MoE-B_16 --checkpoint-dir=$MODEL_DIR --log_dir=$LOG_DIR --training_option=Keras --tpu=$TPU_NAME --eval_every=10 --warmup-epochs=20 --beta2=0.999 --opt=LAMB --inception_style --use_representation --share_att --share_ffn --use_moe --num_experts=4 --num_masked_experts=1.0 --capacity_factor=1.2 --top_k=2 --aux_loss_alpha=0.01 --aux_loss_beta=0.0 --use_aux_loss
