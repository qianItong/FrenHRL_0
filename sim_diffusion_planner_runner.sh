cwd=$(pwd)

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

ARGS_FILE="$cwd/config/args/args.yaml"
CKPT_FILE="$cwd/checkpoint/planner.pth"
export PYTHONPATH=$(pwd):$PYTHONPATH

SPLIT="test"
SCENARIO_BUILDER="nuplan"

export NUPLAN_DATA_ROOT= # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT= # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")

PLANNER=HiFren_planner

CHALLENGES="closed_loop_nonreactive_agents closed_loop_reactive_agents" # 
for challenge in $CHALLENGES; do
    python run_simulation.py \
        +simulation=$challenge \
        planner=$PLANNER \
        planner.HiFren_planner.ckpt_path=$CKPT_FILE \
        planner.HiFren_planner.args_path=$ARGS_FILE \
        scenario_builder=$SCENARIO_BUILDER \
        scenario_filter=$SPLIT \
        distributed_mode='SINGLE_NODE' \
        worker.threads_per_node=15 \
        +number_of_gpus_allocated_per_simulation=0.1 \
        experiment_uid=$PLANNER/$CHALLENGES/$(date "+%m-%d-%H-%M-%S") \
        verbose=true \
        distributed_mode='SINGLE_NODE' \
        enable_simulation_progress_bar=true
done