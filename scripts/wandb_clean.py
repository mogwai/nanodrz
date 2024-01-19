import wandb
from wandb.wandb_run import Run

# Replace 'entity/project' with your specific entity and project name
entity_project = "harrycblum/nano-diarization"

api = wandb.Api()
runs:list[Run] = api.runs(entity_project)

for run in runs:

    if len(run.summary.keys()):
        continue
    
    if "_wandb" in run.summary:
        runtime = run.summary("_wandb").get("runtime")

        if runtime and runtime > 600:
            continue
        
        run.delete()