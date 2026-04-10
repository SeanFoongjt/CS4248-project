# Evaluation Comparison Report

Generated on 2026-04-10 21:01:25 from the latest downloaded evaluation artifacts.

## Inputs

- Original run: `runs\eval_all_models\original_test\2026-04-10_12-07-51`
- Shuffled run: `runs\eval_all_models\shuffled_test\2026-04-10_12-07-51`
- Original metrics CSV: `runs\eval_all_models\original_test\2026-04-10_12-07-51\original_test_set_results.csv`
- Shuffled metrics CSV: `runs\eval_all_models\shuffled_test\2026-04-10_12-07-51\experiment_shuffle_description.csv`
- Original predictions CSV: `runs\eval_all_models\original_test\2026-04-10_12-07-51\original_test_set_predictions.csv`
- Shuffled predictions CSV: `runs\eval_all_models\shuffled_test\2026-04-10_12-07-51\experiment_shuffle_description_predictions.csv`

## Headline Findings

- Best original-set F1: `roberta_without_cn` at 0.9973.
- Best shuffled-set F1: `roberta_without_cn` at 0.8840.
- Largest F1 drop after description shuffling: `roberta_with_cn` at -0.1197.

## Metric Comparison

| Model | Original Acc | Shuffled Acc | Delta Acc | Original F1 | Shuffled F1 | Delta F1 | Original Loss | Shuffled Loss | Delta Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `roberta_with_cn` | 0.9961 | 0.8675 | -0.1287 | 0.9965 | 0.8768 | -0.1197 | 0.0283 | 1.8096 | +1.7812 |
| `distilbert_with_cn` | 0.9963 | 0.8656 | -0.1307 | 0.9967 | 0.8779 | -0.1188 | 0.0261 | 1.7687 | +1.7426 |
| `distilbert_without_cn` | 0.9934 | 0.8688 | -0.1245 | 0.9940 | 0.8804 | -0.1136 | 0.0435 | 1.8852 | +1.8418 |
| `roberta_without_cn` | 0.9970 | 0.8743 | -0.1227 | 0.9973 | 0.8840 | -0.1133 | 0.0267 | 2.0051 | +1.9784 |

## Prediction Stability

| Model | Changed Predictions | Changed Pred % | Correct -> Incorrect | Incorrect -> Correct | Net Example Change | Avg Confidence Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `distilbert_with_cn` | 584 / 4376 | 0.1335 | 578 | 6 | -572 | -0.0018 |
| `roberta_with_cn` | 575 / 4376 | 0.1314 | 569 | 6 | -563 | -0.0023 |
| `distilbert_without_cn` | 567 / 4376 | 0.1296 | 556 | 11 | -545 | -0.0017 |
| `roberta_without_cn` | 543 / 4376 | 0.1241 | 540 | 3 | -537 | -0.0026 |

## Example Regressions

### `distilbert_with_cn`

| Row | True Label | Original Pred | Shuffled Pred | Original Conf | Shuffled Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 66 | 1 | 1 | 0 | 1.0000 | 0.5253 | -0.4747 | reddi wip canister used as directed |
| 2904 | 1 | 1 | 0 | 1.0000 | 0.5642 | -0.4358 | thick sweater no match for determined nipples |
| 539 | 0 | 0 | 1 | 1.0000 | 0.5952 | -0.4048 | mark zuckerberg: 'i regret' rejecting idea that facebook fake news altered election |
| 186 | 0 | 0 | 1 | 1.0000 | 0.6049 | -0.3951 | celebrities celebrate fourth of july with some fun in the sun |
| 1906 | 1 | 1 | 0 | 1.0000 | 0.6770 | -0.3230 | report: gross-ass gourd all bumpy and shit |

### `roberta_with_cn`

| Row | True Label | Original Pred | Shuffled Pred | Original Conf | Shuffled Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1078 | 0 | 0 | 1 | 1.0000 | 0.5047 | -0.4953 | cbs, pbs cut ties with charlie rose following sexual misconduct allegations |
| 4112 | 0 | 0 | 1 | 1.0000 | 0.5052 | -0.4948 | a second-by-second breakdown of sean spicer's holocaust comments |
| 198 | 1 | 1 | 0 | 1.0000 | 0.5615 | -0.4385 | classic movie 'avatar' updated for today's audiences |
| 1024 | 1 | 1 | 0 | 1.0000 | 0.5865 | -0.4135 | supreme court allows corporations to run for political office |
| 1554 | 1 | 1 | 0 | 1.0000 | 0.6992 | -0.3008 | that guy from that one show in rehab |

### `distilbert_without_cn`

| Row | True Label | Original Pred | Shuffled Pred | Original Conf | Shuffled Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2059 | 1 | 1 | 0 | 1.0000 | 0.5037 | -0.4963 | protesters ignored |
| 4296 | 0 | 0 | 1 | 1.0000 | 0.5791 | -0.4209 | the glory days: iv |
| 422 | 1 | 1 | 0 | 1.0000 | 0.5873 | -0.4127 | joe wilson getting bored with no-longer-covert wife |
| 2569 | 0 | 0 | 1 | 1.0000 | 0.6250 | -0.3750 | jaden smith is all of us during kanye west's vmas speech |
| 3679 | 1 | 1 | 0 | 1.0000 | 0.6303 | -0.3697 | target range under fire from community members |

### `roberta_without_cn`

| Row | True Label | Original Pred | Shuffled Pred | Original Conf | Shuffled Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1973 | 0 | 0 | 1 | 1.0000 | 0.5689 | -0.4311 | gop lawmaker: gay rep. should have stayed in the closet |
| 3469 | 1 | 1 | 0 | 1.0000 | 0.6118 | -0.3882 | report: only 260,000 more games of 'candy crush' until you die |
| 3684 | 0 | 0 | 1 | 1.0000 | 0.6286 | -0.3714 | larry wilmore throws some serious shade at brian williams, the media |
| 3270 | 1 | 1 | 0 | 1.0000 | 0.6329 | -0.3671 | everyone at thanksgiving doing chore to get away from rest of family |
| 1836 | 1 | 1 | 0 | 1.0000 | 0.6741 | -0.3259 | ohio state hires jim tressel as head football coach |

