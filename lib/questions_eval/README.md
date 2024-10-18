# How to evaluate the performance of generated synthetic data?

## Datasets

@Simon updating ...

## Metrics

| Term              | Definition                                                                                                                  | Formula                                                                                                                                                            | Interpretation                                                                                                        |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Coverage score    | How comprehensively the summary covers the content of the original document.                                                | 100 − X, where X is the percentage of document generated questions that receive an "IDK" (I Don’t Know) response based on the summary.                             | A higher coverage score indicates that the summary captures more of the original details and is less generic.         |
| Conformity score  | Whether the summary avoids contradicting the document.                                                                      | It is derived by identifying the percentage of questions for which the summary’s answer is "NO" and the document’s is "YES", or vice versa, and computing 100 − X. | A higher conformity score signifies a greater alignment between the summary and the document.                         |
| Consistency score | The level of non-hallucination, is based on the accuracy of factual information in the summary as compared to the document. | 100 − X, where X is the percentage of summary derived questions that are answered with an "IDK" based on the document, indicating factual discrepancies.           | A higher consistency score suggests that the summary is more factual and contains fewer inaccuracies or fabrications. |

## Implementation

Command

```
python run.py -m model=llama3.1-405b-local samples=10 num_questions=5
```

Scripts:

```
cd ./open-nlp/lib/questions_eval
bash/experiments/super_tiny.sh
```

## References

- [SemScore: Evaluating LLMs with Semantic Similarity](https://huggingface.co/blog/g-ronimo/semscore)
- [MEDIC: Towards a Comprehensive Framework for evaluating LLMs in Clinical Applications](https://arxiv.org/pdf/2409.07314)

## Contributors

- [@simonmeoni](https://github.com/simonmeoni)
- [@honghanhh](https://github.com/honghanhh)
