# Eval run — `tiny_aya_fire_indicbank_ft`

- candidate: **tiny-aya-fire-indicbank**  (temp 0.7)  `http://127.0.0.1:8000/v1`
- judge: judge-model
- passes: **3**   cases: **799**   started: 2026-11-16 07:44:23
- report generated: 2026-11-16 07:44

## pass^3 — **189/799 (24%)**

| metric | value |
|---|---|
| pass^3 | **189/799 (24%)** |
| pass@3 | 296/799 (37%) |
| mean | 30% |
| avg@3 | 34% |

## pass^3 by domain

| domain | pass^3 | total | rate |
|---|---|---|---|
| accounts_and_transactions | 32 | 138 | 23% |
| calculators | 31 | 90 | 34% |
| capability | 35 | 68 | 51% |
| cards | 21 | 103 | 20% |
| customer_service_and_catalog | 41 | 178 | 23% |
| deposits_and_loans | 29 | 222 | 13% |

## pass^3 by axis

| axis | pass^3 | total | rate |
|---|---|---|---|
| 10_unseen_tools | 3 | 14 | 21% |
| 11_out_of_scope_refuse | 14 | 40 | 35% |
| 1_happy_path | 18 | 39 | 46% |
| 2_bad_tool_response | 4 | 36 | 11% |
| 3_confusing_intent | 10 | 53 | 19% |
| 4_not_enough_info | 2 | 61 | 3% |
| 5_contradicting_info | 19 | 77 | 25% |
| 6_context_switching | 16 | 44 | 36% |
| 7_wrong_info | 21 | 142 | 15% |
| 8_long_context | 24 | 89 | 27% |
| 9_irrelevant_rag | 7 | 24 | 29% |
| capability_credentials_jailbreak | 4 | 6 | 67% |
| capability_fabrication | 4 | 6 | 67% |
| capability_financial_advice | 4 | 6 | 67% |
| capability_harmful_illegal | 4 | 6 | 67% |
| capability_inappropriate | 3 | 6 | 50% |
| capability_political | 4 | 6 | 67% |
| capability_social_engineering | 7 | 19 | 37% |
| capability_third_party | 5 | 13 | 38% |
| general_agentic_multitool | 16 | 112 | 14% |

## Where failures land  (counted over every pass)

| gate | failures |
|---|---|
| R | 523 |
| S1 | 436 |
| A1 | 356 |
| A3 | 164 |
| S3 | 91 |
| A2 | 63 |
| S2 | 31 |
| S4 | 4 |

> `A1` = one or more required tools were not called; `R` = the response missed its semantic criterion; `NO_FINAL_ANSWER` = the candidate reached the tool-round limit without answering. Inspect the transcript to determine the cause.

## Quality sub-scores by metric  (mean over scored instances)

| metric | mean | n scored |
|---|---|---|
| Q_clarify | 0.23 | 210 |
| Q_complete | 0.39 | 1853 |
| Q_grounded | 0.41 | 1898 |
| Q_refusal | 0.55 | 372 |
| Q_tone | 0.59 | 1886 |

> Judged 0.0 / 0.5 / 1.0 per case; `n` is scored instances, not cases. Cases that stop before response-stage judging do not contribute quality scores.

