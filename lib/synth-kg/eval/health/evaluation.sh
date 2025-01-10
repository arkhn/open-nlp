task_name="iCliniq"
refer_model_name=(
    "gpt-3.5-turbo"
    "claude-2"
    "gpt-4"
    "text-davinci-003"
)

for model in "${refer_model_name[@]}"; do
    data_path="lib/synth-kg/datasets/health/eval/reference_outputs/${model}/"

    python gpt_eval.py \
        --model_output "$2" \
        --reference_output "$data_path" \
        --model_name "$1" \
        --refer_model_name "${model}" \
        --engine "gpt-3.5-turbo" \
        --reference_first \
        --task_name "${task_name}" \
        --batch_dir "$2"/gpt_eval_results \
        --max_test_number 1000

    python evaluation.py \
        --compared_model "$1" \
        --data_path "$2"/gpt_eval_results \
        --engine "gpt-3.5-turbo" \
        --refer_model "${model}"
done