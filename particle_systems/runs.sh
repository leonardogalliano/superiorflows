








###############################################################################
# OLD CLI CODE
###############################################################################

## Test baseline ~163 s
uv run particle_systems/training_particles.py \
    --nsteps 100 \
    --data-path /Users/Leonardo/Documents/PhD/Projects/ParticlesMC/data/datasets/JBB25/T1.0/N44/M100/steps1000000/seed42/trajectories \
    --model-file /Users/Leonardo/Documents/Postdoc/Projects/superiorflows/particle_systems/models/jung_biroli_berthier.json \
    --loss-type maximum_likelihood \
    --width 64 \
    --depth 4 \
    --lr 1e-3 \
    --batch-size 256 \
    --seed 42 \
    --log-freq 10 \
    --ckpt-path tmp/ckpt_particles \
    --overwrite \
    --num-checkpoints 1 \
    --temperature 1.0 \
    --ess \
    --ess-freq 50 \
    --ess-samples 512 \
    --tensorboard \
    --tensorboard-log-dir tmp/tb_logs \
    --num-workers 1 \
    --prefetch-buffer-size 4 \
    --solver tsit5 \
    --atol 1e-5 \
    --rtol 1e-5
    
## Stochastic Interpolants ~8 s
uv run particle_systems/training_particles.py \
    --nsteps 100 \
    --data-path /Users/Leonardo/Documents/PhD/Projects/ParticlesMC/data/datasets/JBB25/T1.0/N44/M100/steps1000000/seed42/trajectories \
    --model-file /Users/Leonardo/Documents/Postdoc/Projects/superiorflows/particle_systems/models/jung_biroli_berthier.json \
    --loss-type stochastic_interpolant \
    --ot \
    --use-gamma \
    --width 64 \
    --depth 4 \
    --lr 1e-3 \
    --batch-size 256 \
    --seed 42 \
    --log-freq 10 \
    --ckpt-path tmp/ckpt_particles \
    --overwrite \
    --num-checkpoints 1 \
    --temperature 1.0 \
    --ess \
    --ess-freq 50 \
    --ess-samples 512 \
    --tensorboard \
    --tensorboard-log-dir tmp/tb_logs \
    --num-workers 1 \
    --prefetch-buffer-size 4 \
    --solver tsit5 \
    --atol 1e-5 \
    --rtol 1e-5

## Stochastic Interpolants + Box Symmetry ~22 s
uv run particle_systems/training_particles.py \
    --nsteps 100 \
    --data-path /Users/Leonardo/Documents/PhD/Projects/ParticlesMC/data/datasets/JBB25/T1.0/N44/M100/steps1000000/seed42/trajectories \
    --model-file /Users/Leonardo/Documents/Postdoc/Projects/superiorflows/particle_systems/models/jung_biroli_berthier.json \
    --loss-type stochastic_interpolant \
    --ot \
    --ot-box-symmetry \
    --use-gamma \
    --width 64 \
    --depth 4 \
    --lr 1e-3 \
    --batch-size 256 \
    --seed 42 \
    --log-freq 10 \
    --ckpt-path tmp/ckpt_particles \
    --overwrite \
    --num-checkpoints 1 \
    --temperature 1.0 \
    --ess \
    --ess-freq 50 \
    --ess-samples 512 \
    --tensorboard \
    --tensorboard-log-dir tmp/tb_logs \
    --num-workers 1 \
    --prefetch-buffer-size 10 \
    --solver tsit5 \
    --atol 1e-5 \
    --rtol 1e-5


## Compare with pytorch
uv run particle_systems/training_particles.py \
    --nsteps 1250 \
    --data-path /Users/Leonardo/Documents/PhD/Projects/ParticlesMC/data/datasets/JBB25/T1.0/N44/M100/steps10000000/seed42/trajectories \
    --model-file /Users/Leonardo/Documents/Postdoc/Projects/superiorflows/particle_systems/models/jung_biroli_berthier.json \
    --loss-type stochastic_interpolant \
    --ot \
    --width 64 \
    --depth 3 \
    --lr 1e-3 \
    --batch-size 64 \
    --seed 42 \
    --log-freq 10000 \
    --ckpt-path tmp/ckpt_particles \
    --overwrite \
    --num-checkpoints 1 \
    --temperature 1.0 \
    --tensorboard \
    --tensorboard-log-dir tmp/tb_logs \
    --num-workers 1 \
    --prefetch-buffer-size 10 \
    --solver tsit5 \
    --atol 1e-5 \
    --rtol 1e-5
