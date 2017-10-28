python3 actor_critic_acrobotv1.py --gamma 0.99 --seed 543 --num-episodes 120000\
	--log-interval 10 --VAE-dir VAE_kl_dim\(z\)=4_discount=3e-5 --use-cuda 
	--use-buffer --buffer-capacity 10000 --batch-size 128