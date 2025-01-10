task_name="icliniq"
reference_outputs=(

)
refer_model_name=(
  "gpt-3.5-turbo"
)

for i in "${!reference_outputs[@]}"; do

                   data_path="./data/comparsion/refer_${refer_model_name[$i]}/${task_name}/"
                   python gpt_eval.py \
                       --model_output "$2" \
                       --reference_output "${reference_outputs[$i]}" \
                       --model_name "$1" \
                       --refer_model_name "${refer_model_name[$i]}" \
                       --engine "gpt-3.5-turbo" \
                       --reference_first \
                       --task_name "${task_name}" \
                       --batch_dir  $data_path \
                       --max_test_number 1000

                  python evaluation.py --compared_model $1 \
                                       --data_path $data_path \
                                       --engine "gpt-3.5-turbo" \
                                       --refer_model ${refer_model_name[$i]} \

done