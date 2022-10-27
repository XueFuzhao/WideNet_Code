
export TPU_NAME=vit128
export DATA_DIR=gs:///imagenet-2012-tfrecord
export MODEL_DIR=gs:///cv_narrownet/widenet-base-42
export LOG_DIR=gs:///vit_exp1/log




python3 train.py --img_size=224 --base-lr=0.01 --batch-size=4096 --epoch=300 --wd=0.1 --data_set=Imagenet --data_dir=$DATA_DIR --model_type=ViT-MoE-B_16 --checkpoint-dir=$MODEL_DIR --log_dir=$LOG_DIR --training_option=Keras --tpu=$TPU_NAME --eval_every=10 --warmup-epochs=30 --beta2=0.999 --opt=LAMB --inception_style --use_representation --use_moe --share_ffn --share_att --num_experts=4 --num_masked_experts=0.0 --capacity_factor=1.2 --top_k=2 --aux_loss_alpha=0.01 --aux_loss_beta=0.0 --use_aux_loss --seed=42




