sbatch bash/jz-slurm/submit-a100.sh \
            model.name=meta-llama/Llama-3.2-3B-Instruct \
            model.lora.target_modules='["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
            dataset.name=bio-datasets/mimic_style_transfer \
            max_steps=3 \
            score.model_name_or_path=FremyCompany/BioLORD-2023-C
