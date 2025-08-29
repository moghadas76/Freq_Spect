export CUDA_VISIBLE_DEVICES=5
# add for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/SpectFlow_fix/traf_abl" ]; then
    mkdir ./logs/SpectFlow_fix/traf_abl
fi
seq_len=700
model_name=SpectFlow

for H_order in 10 8 5
do
for seq_len in 720 360 180 90
do
for m in 1 2
do
for seed in 114
do
for bs in 64
do



python run_longExp_F.py --is_training 1 --root_path ./dataset/ --data_path PATH/brussels.csv --model_id Brussels_720_8 --model Flow_FITS --flow_time_dim 64 --flow_hidden_multiplier 0.626 --data Brussels --features M --freq 15min --seq_len 720 --loss flow --pred_len 96 --enc_in 7 --des Exp --train_mode 2 --H_order 6 --gpu 1 --seed 114 --patience 20 --itr 1 --batch_size 4 --learning_rate 0.00267 --weight_decay 1.259841684694534e-06


wait

done
done
done
done
done