task_name="iCliniq"
refer_model_name=(
    "gpt-3.5-turbo"
    "claude-2"
    "gpt-4"
    "text-davinci-003"
)

for model in "${refer_model_name[@]}"; do
    data_path="datasets/health/eval/reference_outputs/${model}/iCliniq_output.jsonl"
    batch_dir=$(dirname "$2")
    max_test_number=20

    python eval/health/gpt_eval.py \
        --model_output "$2" \
        --reference_output "$data_path" \
        --model_name "$1" \
        --refer_model_name "${model}" \
        --engine "gpt-3.5-turbo" \
        --reference_first \
        --task_name "${task_name}" \
        --batch_dir $batch_dir/gpt_results/ \
        --max_test_number "$max_test_number"

    python eval/health/gpt_eval.py \
        --model_output "$2" \
        --reference_output "$data_path" \
        --model_name "$1" \
        --refer_model_name "${model}" \
        --engine "gpt-3.5-turbo" \
        --task_name "${task_name}" \
        --batch_dir $batch_dir/gpt_results/ \
        --max_test_number "$max_test_number"

    python eval/health/evaluation.py \
        --compared_model "$1" \
        --data_path $batch_dir/gpt_results/ \
        --engine "gpt-3.5-turbo" \
        --refer_model "${model}" \
        --max_test_number "$max_test_number"
done