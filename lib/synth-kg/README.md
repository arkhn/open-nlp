# Results

| Model                                             | Text-davinci-003 | GPT-3.5-turbo | GPT-4 | Claude-2 | Avg       |
| ------------------------------------------------- | ---------------- | ------------- | ----- | -------- | --------- |
| DP-Instruct-ICL                                   | 0.382            | 0.295         | 0.187 | 0.199    | 0.266     |
| KnowledgeSG                                       | 0.776            | 0.530         | 0.457 | 0.488    | 0.562     |
| SynthKG w. Alpacare + response - sft (ows6nxyzk)  | 0.592            | 0.504         | 0.478 | 0.468    | **0.510** |
| SynthKG w. Alpacare + response - dpo-1 (ny9w8rj9) | 0.630            | 0.510         | 0.476 | 0.469    | **0.520** |
| SynthKG w. Alpacare - epoch 1 - sft (ows6nxyzk)   | 0.621            | 0.507         | 0.485 | 0.48     | **0.530** |
| SynthKG w. Alpacare - epoch 1 - dpo-1 (0kszbdra)  | 0.641            | 0.51          | 0.49  | 0.49     | **0.532** |
| SynthKG w. Alpacare - epoch 1 - dpo-2 (8rjj64f60) | 0.622            | 0.51          | 0.49  | 0.49     | **0.530** |

# Corresponding ID and experiment

| Id        | Step                 | Semscore w/ STD    | Experiment                                                                                                                                                                                                                                         |
| --------- | -------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| s0j9oljg  | sft                  | 0.594 (0.040)      | initial run with no alpacare seed                                                                                                                                                                                                                  |
| a4iitw7e  | dpo-1                | 0.560              | previous state : s0j9oljg. Vanilla with problem of stopping separator during evaluation ("\n"). So the evaluator model doesn't generate good response due to a reduced questions context                                                           |
| wwovogn1  | dpo-2                | 0.527              | previous state : a4iitw7e, s0j9oljg. Vanilla with problem of stopping separator during evaluation ("\n"). So the evaluator model doesn't generate good response due to a reduced questions context                                                 |
| cajea4rd  | sft                  | 0.491 (0.174)      | (\*) sft with really small sft (only 15 doc lol) the std of generated scored candidates are more various and the dpo learning is more stable and I see improvement in DPO when I use the learned model to generate train dataset with that (0.173) |
| umayvnmm  | dpo-1                | 0.614 (0.052)      | previous state : cajea4rd see (\*)                                                                                                                                                                                                                 |
| ftq9yrkn  | sft                  | 0.614 (0.051)      | less epoch -> only 5 (std 0.051 )                                                                                                                                                                                                                  |
| ows6nxyzk | sft_less-variability | 0.587 (0.0680)     | i used sampling params without any variations - it gives me very an mean std (0.068 vs. 0.173 for cajea4rd run) - but I have a really better score on evaluation preference                                                                        |
| tebe2eb3  | dpo-1                | 0.567 (0.120)      | previous step : ows6nxyzk I have a worst score in evaluation (-2 in global). ** I suspect that the sft step is overfitted maybe with less epoch and give less variation between candidate and give a biased dataset **                             |
| ows6nxyzk | sft                  | **0.578** (0.0689) | i used sampling params with variations                                                                                                                                                                                                             |
| ny9w8rj9  | dpo-1                | **0.595** (0.06)   | previous step : ows6nxyzk. But i used a generation with several different sampling params for each candidate (sft std 0.068 -> std 0.044)                                                                                                          |
| 4aidu9e6  | sft                  | 0.580 (0.073)      | use only one epoch for training                                                                                                                                                                                                                    |
| 0kszbdra  | dpo-1                | 0.630              | use only one epoch for training                                                                                                                                                                                                                    |
| 8rjj...   | dpo-2                | 0.630              | use only one epoch for training                                                                                                                                                                                                                    |

Notes: more variation = better score
