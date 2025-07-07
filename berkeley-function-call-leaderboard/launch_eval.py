import os
import subprocess
import re
from os.path import join
import argparse
# model_path = "meta-llama/Llama-3.1-8B"



def run_subprocess_slurm(command):
    # Execute the sbatch command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the output and error, if any
    print("command:", command)
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Get the job ID for linking dependencies
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if match:
        job_id = match.group(1)
    else:
        job_id = None

    return job_id


gpus_per_node = 2
nodes = 1
# model_path = "meta-llama/Llama-3.1-8B-Instruct-FC"
# model_path = "meta-llama/Llama-3.3-70B-Instruct-FC"
model_path = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8-FC"
result_dir = "/fsx-project/winnieyangwn/BFCL_OUTPUT"
score_dir = "/fsx-project/winnieyangwn/BFCL_OUTPUT/score"
# test_category = "single_turn"





# job_name  = f"activation_pca_moe_{model_path}"
job_name = f"evaluate_response_{model_path}"

slurm_cmd = f'''sbatch --account=genai_interns --qos=genai_interns \
    --job-name={job_name} --nodes={nodes} --gpus-per-node={gpus_per_node} \
    --time=24:00:00 --output=LOGS/{job_name}.log \
    --wrap="\
        python -m bfcl_eval.eval_checker.eval_runner \
        --model={model_path} \
        --result-dir={result_dir} \
        --score-dir={score_dir}; \

"'''
job_id = run_subprocess_slurm(slurm_cmd)
print("job_id:", job_id)


# slurm_cmd = f'''sbatch --account=genai_interns --qos=genai_interns \
#     --job-name={job_name} --nodes=1 --gpus-per-node={gpus_per_node} \
#     --time=24:00:00 --output=LOGS/{job_name}.log \
#     --wrap="\
#         python OLMoE_demo.py \
# "'''
# job_id = run_subprocess_slurm(slurm_cmd)
# print("job_id:", job_id)


# job_name  = "generate_LLMU"
# slurm_cmd = f'''sbatch --account=genai_interns --qos=lowest \
#     --job-name={job_name} --nodes=1 --gpus-per-node={gpus_per_node} \
#     --time=24:00:00 --output=LOGS/{job_name}.log \
#     --wrap="\
#         python generate_dataset_vllm_qwen_vl.py \
#         ++n_train=128 \
#         ++batch_size=64; \
# "'''
# job_id = run_subprocess_slurm(slurm_cmd)
# print("job_id:", job_id)