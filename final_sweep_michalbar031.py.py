import wandb
YOUR_WANDB_USERNAME = "michalbar"
project = "NLP2024_PROJECT_michalbar3"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "MAB ",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {

        "seed": {"values": [3]},
        "online_simulation_factor": {"values": [4,3]},
        "prioritized_strategies": {"values": [0,1,2]},
        "MAB_selection": {"values": [True]},
        # "architecture": {"values": ["LSTM"]},#"transformer"
        # "features": {"values": ["EFs", "GPT4", "BERT"]},

    },
    "command": command
}

# Initialize a new sweep
# sweep_id = wandb.sweep(sweep=sweep_config, project=project)
# print("run this line to run your agent in a screen:")
# print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
#

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)

print("Run these lines to run your agent in a screen:")
parallel_num = 6

if parallel_num > 10:
    print('Are you sure you want to run more than 10 agents in parallel? It would result in a CPU bottleneck.')
for i in range(parallel_num):
    print(f"screen -dmS \"final_sweep_agent_{i}\" nohup wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
