# Results

| Model                                            | Text-davinci-003 | GPT-3.5-turbo | GPT-4 | Claude-2 | Avg   |
|--------------------------------------------------|------------------|---------------|-------|----------|-------|
| DP-Instruct-ICL                                  | 0.382            | 0.295         | 0.187 | 0.199    | 0.266 |
| KnowledgeSG                                      | 0.776            | 0.530         | 0.457 | 0.488    | 0.562 |
| SynthKG w. Alpacare + response - sft (ows6nxyzk) | 0.592            | 0.504         | 0.478 | 0.468    |       |

# Corresponding ID and experiment

| Id        | Step                 | Experiment                                                                                                                                                                                                                                        |
|-----------|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| a4iitw7e  | dpo-1                | previous state : s0j9oljg. Vanilla with problem of stopping separator during evaluation ("\n"). So the evaluator model doesn't generate good response due to a reduced questions context                                                          |
| cajea4rd  | sft                  | (*) sft with really small sft (only 15 doc lol) the std of generated scored candidates are more various and the dpo learning is more stable and I see improvement in DPO when I use the learned model to generate train dataset with that (0.173) |
| ftq9yrkn  | sft                  | less epoch -> only 5 (std 0.051 )                                                                                                                                                                                                                 |
| ny9w8rj9  | dpo-1                | previous step : ows6nxyzk. But i used a generation with several different sampling params for each candidate  (sft std  0.068 -> std 0.044)                                                                                                       |
| ows6nxyzk | sft_less-variability | i used sampling params without any variations - it gives me very an mean std (0.068 vs. 0.173 for cajea4rd run) - but I have a really better score on evaluation preference                                                                       |
| s0j9oljg  | sft                  | initial run                                                                                                                                                                                                                                       |
| tebe2eb3  | dpo-1                | previous step : ows6nxyzk I have a worst score in evaluation (-2 in global). ** I suspect that the sft step is overfitted maybe with less epoch and give less variation between candidate and give a biased dataset **                            |
| umayvnmm  | dpo-1                | previous state : cajea4rd see (*)                                                                                                                                                                                                                 |
| wwovogn1  | dpo-2                | previous state : a4iitw7e, s0j9oljg. Vanilla with problem of stopping separator during evaluation ("\n"). So the evaluator model doesn't generate good response due to a reduced questions context                                                |                                                                                                                                                   |


