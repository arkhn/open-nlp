python style_transfer/run_rb_gen.py model.name=meta-llama/Llama-3.2-3B-Instruct \
            model.peft_config.target_modules='["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]' \
            dataset.name=bio-datasets/mimic_style_transfer \
            max_steps=3 \
            score.model.model_name_or_path=FremyCompany/BioLORD-2023-C \
            dataset.sft_ratio=0.06 \
            dataset.gen_ratio=0.7
