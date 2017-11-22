python3 main.py --experiment 'a|z(s)' --env Acrobot-v1 --gamma 0.99 --seed 543 \
	--num-episodes 80000 --log-interval 10 --policy-lr 0.002 --z-dim 4 --vae-lr 5e-4 \
	--buffer-capacity 50000 --batch-size 512 --vae-update-frequency 100 \
	--vae-update-times 10000 --kl-divergence --kl-weight 3e-5
