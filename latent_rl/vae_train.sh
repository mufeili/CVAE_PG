python3 vae_example_s2s.py --batch-size 512 --iterations 10000 --epochs 20 \
	--seed 543 --log-interval 10 --buffer-capacity 50000 --kl-divergence \
	--policy-dir actor_critic_20171020-082409 --kl-weight 1e-5