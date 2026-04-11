# Random Section Replacement Comparison Report

Generated on 2026-04-11 22:29:09 from downloaded evaluation artifacts.

## Inputs

- Original run: `runs\eval_all_models\2026-04-10_12-07-51`
- Random-section run: `runs\random_section_replacement\2026-04-11_22-07-01`
- Original metrics CSV: `runs\eval_all_models\2026-04-10_12-07-51\original_test_set_results.csv`
- Random-section metrics CSV: `runs\random_section_replacement\2026-04-11_22-07-01\experiment_random_section_replacement.csv`
- Original predictions CSV: `runs\eval_all_models\2026-04-10_12-07-51\original_test_set_predictions.csv`
- Random-section predictions CSV: `runs\random_section_replacement\2026-04-11_22-07-01\experiment_random_section_replacement_predictions.csv`

## Headline Findings

- Best original-set F1: `roberta_without_cn` at 0.9973.
- Best random-section F1: `roberta_with_cn` at 0.9863.
- Largest F1 drop after random section replacement: `distilbert_with_cn` at -0.0195.

## Metric Comparison

| Model | Original Acc | Random Acc | Delta Acc | Original F1 | Random F1 | Delta F1 | Original Loss | Random Loss | Delta Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distilbert_with_cn` | 0.9963 | 0.9751 | -0.0213 | 0.9967 | 0.9772 | -0.0195 | 0.0261 | 0.1762 | +0.1502 |
| `distilbert_without_cn` | 0.9934 | 0.9744 | -0.0190 | 0.9940 | 0.9765 | -0.0175 | 0.0435 | 0.2158 | +0.1723 |
| `roberta_without_cn` | 0.9970 | 0.9842 | -0.0128 | 0.9973 | 0.9856 | -0.0117 | 0.0267 | 0.1468 | +0.1201 |
| `roberta_with_cn` | 0.9961 | 0.9849 | -0.0112 | 0.9965 | 0.9863 | -0.0102 | 0.0283 | 0.1352 | +0.1069 |

## Prediction Stability

| Model | Replaced Sections | Changed Predictions | Changed Pred % | Correct -> Incorrect | Incorrect -> Correct | Net Example Change | Avg Confidence Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distilbert_with_cn` | 4162 / 4376 | 97 / 4376 | 0.0222 | 95 | 2 | -93 | -0.0018 |
| `distilbert_without_cn` | 4162 / 4376 | 99 / 4376 | 0.0226 | 91 | 8 | -83 | -0.0013 |
| `roberta_without_cn` | 4162 / 4376 | 70 / 4376 | 0.0160 | 63 | 7 | -56 | -0.0008 |
| `roberta_with_cn` | 4162 / 4376 | 57 / 4376 | 0.0130 | 53 | 4 | -49 | -0.0012 |

## Section Replacement Patterns

### `distilbert_with_cn`

Top replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | politics | 487 |
| politics | news | 400 |
| news | entertainment | 297 |
| entertainment | news | 229 |
| politics | entertainment | 139 |
| entertainment | politics | 133 |
| news | other | 91 |
| news | voices | 74 |

Top correct-to-incorrect replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | world | 7 |
| news | voices | 5 |
| news | lifestyle | 5 |
| lifestyle | news | 4 |
| news | entertainment | 4 |
| news | style | 3 |
| news | other | 3 |
| news | health | 3 |

### `distilbert_without_cn`

Top replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | politics | 487 |
| politics | news | 400 |
| news | entertainment | 297 |
| entertainment | news | 229 |
| politics | entertainment | 139 |
| entertainment | politics | 133 |
| news | other | 91 |
| news | voices | 74 |

Top correct-to-incorrect replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | entertainment | 8 |
| news | politics | 6 |
| news | voices | 5 |
| news | world | 5 |
| news | lifestyle | 5 |
| entertainment | arts and culture | 3 |
| lifestyle | news | 3 |
| news | other | 3 |

### `roberta_without_cn`

Top replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | politics | 487 |
| politics | news | 400 |
| news | entertainment | 297 |
| entertainment | news | 229 |
| politics | entertainment | 139 |
| entertainment | politics | 133 |
| news | other | 91 |
| news | voices | 74 |

Top correct-to-incorrect replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | lifestyle | 5 |
| news | voices | 5 |
| politics | news | 3 |
| news | politics | 3 |
| news | relationships | 2 |
| news | world | 2 |
| entertainment | arts and culture | 2 |
| lifestyle | news | 2 |

### `roberta_with_cn`

Top replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | politics | 487 |
| politics | news | 400 |
| news | entertainment | 297 |
| entertainment | news | 229 |
| politics | entertainment | 139 |
| entertainment | politics | 133 |
| news | other | 91 |
| news | voices | 74 |

Top correct-to-incorrect replacement pairs:

| Original Section | Replacement Section | Count |
| --- | --- | ---: |
| news | voices | 5 |
| news | lifestyle | 4 |
| news | entertainment | 4 |
| entertainment | news | 3 |
| news | relationships | 2 |
| politics | other | 2 |
| family | news | 2 |
| voices | news | 2 |

## Example Regressions

### `distilbert_with_cn`

| Row | True Label | Original Pred | Random Pred | Original Section | Replacement Section | Original Conf | Random Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 2602 | 1 | 1 | 0 | entertainment | money | 0.9995 | 0.5179 | -0.4816 | voice of patrick stewart lends air of legitimacy |
| 661 | 1 | 1 | 0 | politics | world | 1.0000 | 0.5290 | -0.4709 | trump pours milk over bowl of skittles while settling in to watch comey hearing |
| 2006 | 1 | 1 | 0 | news | other | 0.9998 | 0.6542 | -0.3456 | 15,000 brown people dead somewhere |
| 2249 | 1 | 1 | 0 | news | nature | 1.0000 | 0.6747 | -0.3253 | 'oh, was i not enough for you?' amazon echo asks couple bringing new baby home |
| 456 | 1 | 1 | 0 | news | style | 1.0000 | 0.6792 | -0.3208 | pope francis washes feet of phillie phanatic |

### `distilbert_without_cn`

| Row | True Label | Original Pred | Random Pred | Original Section | Replacement Section | Original Conf | Random Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 945 | 1 | 1 | 0 | news | world | 1.0000 | 0.6640 | -0.3360 | polar bear cub just knows he's going to be last of species |
| 3542 | 0 | 0 | 1 | religion | politics | 1.0000 | 0.7252 | -0.2748 | ashley madison and the clergy |
| 1723 | 1 | 1 | 0 | news | food and drink | 1.0000 | 0.7389 | -0.2611 | federal prison system retires mcveigh's number |
| 704 | 1 | 1 | 0 | news | family | 1.0000 | 0.7473 | -0.2527 | samuel adams apologizes for 'boston sucks' pilsner |
| 97 | 1 | 1 | 0 | news | relationships | 1.0000 | 0.8002 | -0.1998 | ape's tits incredible |

### `roberta_without_cn`

| Row | True Label | Original Pred | Random Pred | Original Section | Replacement Section | Original Conf | Random Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 3492 | 1 | 1 | 0 | politics | business | 0.9998 | 0.5417 | -0.4582 | trump, putin hold first joint press crackdown |
| 3165 | 1 | 1 | 0 | news | entertainment | 1.0000 | 0.5608 | -0.4392 | u.s. stock market soars after bernanke's reassuring comments about 'pacific rim' |
| 1145 | 1 | 1 | 0 | news | education | 1.0000 | 0.5936 | -0.4064 | actually, suicide not the easy way out for area quadriplegic |
| 2254 | 1 | 1 | 0 | politics | environment | 1.0000 | 0.6261 | -0.3739 | kavanaugh: 'i am not denying that ford was sexually assaulted in some alternate dimension, plane of existence' |
| 1187 | 1 | 1 | 0 | news | impact | 1.0000 | 0.7813 | -0.2187 | woman wakes husband up on valentine's day with hot surprise blowtorch |

### `roberta_with_cn`

| Row | True Label | Original Pred | Random Pred | Original Section | Replacement Section | Original Conf | Random Conf | Delta Conf | Headline |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 750 | 1 | 1 | 0 | news | entertainment | 0.9998 | 0.5180 | -0.4818 | dell acquired by gateway 2000 in merger of 2 biggest names in computer technology |
| 333 | 1 | 1 | 0 | politics | voices | 1.0000 | 0.6179 | -0.3821 | trump warns removing confederate statues could be slippery slope to eliminating racism entirely |
| 2843 | 1 | 1 | 0 | news | business | 1.0000 | 0.6368 | -0.3632 | vote, voter wasted |
| 1609 | 1 | 1 | 0 | politics | world | 1.0000 | 0.6867 | -0.3133 | stormy daniels '60 minutes' interview leads to spike in pornhub searches for anderson cooper |
| 339 | 1 | 1 | 0 | politics | other | 1.0000 | 0.8318 | -0.1682 | rick perry speech electrifies 1,200 scared, miserable racists |

