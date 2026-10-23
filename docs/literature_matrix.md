# Research literature matrix: cross-patient EEG seizure detection

**Purpose.** This is a focused scoping matrix to guide the next research
decision, not a systematic review, meta-analysis, or claim of novelty. It
summarizes papers most directly relevant to the proposed direction:
distribution-shift-aware adaptation and selective prediction for unseen-patient
seizure detection.

**Search snapshot:** 16 October 2026. Targeted searches of PubMed and
Europe PMC used combinations of *EEG seizure detection*, *cross-patient*,
*cross-subject*, *domain adaptation*, *uncertainty*, *calibration*, and
*SzCORE*. Abstracts and available full text were checked for the entries
below. This search is not exhaustive; citation chasing, other databases,
preprints, and conference proceedings remain necessary before any publication
novelty claim.

## Comparison matrix

| Work | Task and data | Target-patient information | Evaluation emphasis | What it means for this project |
|---|---|---|---|---|
| Wang et al. (2024), [Cross-Subject Seizure Detection via Unsupervised Domain-Adaptation](https://doi.org/10.1142/S0129065724500552) | Cross-subject seizure **detection**; CHB-MIT and Siena; reports epoch- and event-based experiments on CHB-MIT. | Unsupervised target-domain adaptation. The abstract describes shallow feature alignment with multi-kernel MMD and deep adversarial alignment. | Feasibility of reducing cross-patient distribution differences; event evaluation is included. | Directly overlaps with “cross-patient detection + domain alignment.” A new contribution cannot be merely adding an alignment step or using an adversarial model. Compare against this class of approach under a fixed, transparent protocol. |
| Wang et al. (2023), [Unsupervised domain adaptation for cross-patient seizure classification](https://doi.org/10.1088/1741-2552/ad0859) | Cross-patient EEG seizure **classification**. | Multi-source, unsupervised adaptation; selects source domains, aligns distributions, and adapts a target subdomain. The abstract says target EEG trials are labeled by the approach, not by a clinician. | Compares the method with 13 existing approaches on two public seizure datasets. | Source selection and target adaptation already have substantial prior art. Inspect the full paper before choosing a shift score or adaptation gate; avoid presenting source selection + alignment as novel on its own. |
| Jemal et al. (2024), [Domain adaptation for EEG-based, cross-subject epileptic seizure prediction](https://doi.org/10.3389/fninf.2024.1303380) | Cross-subject **prediction** (preictal forecasting), not ongoing seizure-onset detection; CHB-MIT and Siena. | Cross-subject deep model with feature-based domain-adaptation methods; no new-patient labels are required by the cross-subject setting described. | Reports improvements in prediction classification metrics for both datasets; prediction windows and forecast horizons are task-specific. | Strongly relevant to transfer and shift, but prediction results are not interchangeable with event sensitivity, detection latency, or false alarms/hour for continuous detection. Keep task definitions separate. |
| Hu et al. (2024), [Uncertainty Estimation and Model Calibration in EEG Signal Classification for Epileptic Seizures Detection](https://doi.org/10.1109/EMBC53108.2024.10782858) | EEG seizure-signal **classification** on Temple and LEMON datasets. | Gaussian processes, Monte-Carlo dropout, and Bayesian neural networks; probability calibration is applied. | Studies uncertainty and calibration, reporting AUROC, negative log-likelihood, and Brier score. | Calibration/uncertainty for seizure-related EEG is not an untouched idea. The remaining question for this project must be evaluated at patient-held-out event level and account for alarm burden and review coverage. |
| Dan et al. (2025; first published online 2024), [SzCORE: Seizure Community Open-Source Research Evaluation framework](https://doi.org/10.1111/epi.18113) | Standardized validation framework and benchmark for EEG-based automated seizure **detection**. | Defines reproducible inputs, annotations, validation strategies, and metrics; it is an evaluation framework, not a target-adaptation policy. | Addresses heterogeneity across datasets, cross-validation, and reported performance measures. | Use SzCORE as an evaluation and reporting reference. A custom patient-grouped LOSO protocol should be documented and related to SzCORE recommendations rather than presented as a replacement standard. |
| He and Wu (2020), [Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach](https://doi.org/10.1109/TBME.2019.2913914) | Transfer learning for EEG brain-computer-interface tasks; not seizure detection. | Unlabeled covariance-based Euclidean alignment. | Aligns second-order statistics across domains for BCI transfer. | Euclidean alignment is a useful baseline, not a seizure-specific novelty claim. Measure both benefit and harm per held-out patient and do not treat alignment as evidence that all shifts are corrected. |

## Synthesis: what is and is not established

1. **Cross-patient domain adaptation for seizure tasks already exists.**
   There are seizure-detection and seizure-classification studies using target
   distribution alignment, including work that evaluates CHB-MIT and Siena.
   The proposed project therefore cannot claim novelty for simply combining
   cross-patient evaluation with Euclidean alignment, MMD, or adversarial
   alignment.
2. **Prediction and detection are different estimands.** Preictal prediction
   papers answer a different question from whether an ongoing annotated event
   is detected, how quickly it is detected, and how many false alarm episodes
   occur per hour. Compare like with like.
3. **EEG uncertainty and probability calibration have precedent.** A more
   specific contribution would need to show that a target-label-free,
   patient-held-out policy improves event-level reliability and workload, not
   only calibration or window-level AUROC.
4. **The possible gap is a hypothesis, not a literature finding.** This
   focused scan does not establish that no prior work predicts *when
   adaptation is harmful* or combines that decision with selective deferral
   for continuous unseen-patient seizure detection. A systematic review and
   full-text comparison are required before asserting that gap.
5. **The Bonn F/S pilot is not evidence for this question.** It is a real-data
   segment-discrimination software example, but its available mirror lacks
   original patient/source-recording identifiers. It cannot establish
   patient-independent or source-independent performance.

## Candidate research question and falsifiable hypotheses

**Question:** Can a fixed policy, selected using only source-patient
development folds and unlabeled calibration EEG from an unseen patient,
decide when to leave a detector unchanged, when to adapt it, or when to defer
predictions for review—and improve the event-performance/review-workload
trade-off over always-adapt and never-adapt baselines?

- **H1 — Adaptation is heterogeneous:** paired adaptation effects vary
  materially across held-out patients, including both improvement and
  degradation.
- **H2 — Unlabeled indicators transfer:** a prespecified shift or signal
  quality indicator computed from calibration EEG predicts adaptation benefit
  on patients excluded from policy development.
- **H3 — The policy adds value:** the locked shift-aware policy improves a
  prespecified event-level objective or selective risk-versus-coverage curve
  over both always-adapt and never-adapt, at a comparable review burden.

Failure to support H2 or H3 is a meaningful result: it would argue against
deploying a shift-gated policy under this protocol.

## Recommended experiment design

1. **Freeze the task and protocol.** Choose continuous onset detection or
   explicitly retrospective event detection. Specify allowed target
   calibration duration, whether calibration may contain seizures, artifact
   handling, causal versus zero-phase filtering, seizure matching, alarm
   latency, and the primary endpoint before inspecting final test results.
2. **Use patient-level outer holdout.** For each outer test patient, fit the
   detector, thresholds, calibration maps, shift thresholds, and adaptation
   policy using source patients only. Never use outer-test seizure labels to
   choose whether or how to adapt.
3. **Create patient-level pseudo-target trials inside development data.**
   Rotate source patients into pseudo-target roles to estimate adaptation
   benefit and learn/lock the policy. This inner policy evaluation must itself
   keep pseudo-target labels out of the adaptation decision; labels may be
   used only to score the outcome during policy development.
4. **Compare a small, matched set of policies:** no adaptation; always
   adapt; a prespecified unlabeled shift-gated adaptation rule; confidence-only
   deferral; and shift-aware adaptation plus deferral. Keep detector,
   preprocessing, data, threshold protocol, and target-calibration budget
   matched.
5. **Test fragile assumptions.** Repeat with short calibration periods and
   calibration windows containing seizure activity or artifacts. Explicitly
   distinguish retrospective transforms that use future samples from a
   causal streaming implementation.
6. **Report event and human-review outcomes.** At minimum report event
   sensitivity, false alarms per non-seizure hour, detection latency,
   patient-level paired differences and intervals, and selective risk versus
   EEG coverage (fraction deferred and corresponding review time). Report
   calibration metrics such as Brier score and negative log-likelihood as
   secondary measures; do not use window AUROC as the sole endpoint.
7. **Seek external validation.** Consider Siena after harmonizing data,
   annotations, montage, and preprocessing. Treat it as a distinct cohort,
   not merely extra records, and do not imply external generalization until
   the protocol has actually been run.

## Immediate project status

The current repository has an exploratory five-model Bonn F/S benchmark and a
continuous CHB-MIT pipeline, but no completed CHB-MIT benchmark output is
included. The Bonn results remain accurately caveated in the
[README](../README.md). The next defensible engineering step is to audit and
freeze the CHB-MIT event evaluation before implementing a learned adaptation
gate or a selective predictor.

## References

1. Wang S, Feng H, Lv H, et al. Cross-Subject Seizure Detection via
   Unsupervised Domain-Adaptation. *International Journal of Neural Systems*.
   2024;34(10):2450055. [doi:10.1142/S0129065724500552](https://doi.org/10.1142/S0129065724500552).
2. Wang Z, Zhang W, Li S, Chen X, Wu D. Unsupervised domain adaptation for
   cross-patient seizure classification. *Journal of Neural Engineering*.
   2023;20(6):066002.
   [doi:10.1088/1741-2552/ad0859](https://doi.org/10.1088/1741-2552/ad0859).
3. Jemal I, Abou-Abbas L, Henni K, Mitiche A, Mezghani N. Domain adaptation
   for EEG-based, cross-subject epileptic seizure prediction.
   *Frontiers in Neuroinformatics*. 2024;18:1303380.
   [doi:10.3389/fninf.2024.1303380](https://doi.org/10.3389/fninf.2024.1303380).
4. Hu J, Ur Rahman MM, Al-Naffouri T, Laleg-Kirati TM. Uncertainty Estimation
   and Model Calibration in EEG Signal Classification for Epileptic Seizures
   Detection. *2024 IEEE EMBC*. 1–5.
   [doi:10.1109/EMBC53108.2024.10782858](https://doi.org/10.1109/EMBC53108.2024.10782858).
5. Dan J, Pale U, Amirshahi A, et al. SzCORE: Seizure Community Open-Source
   Research Evaluation framework for the validation of electroencephalography-based
   automated seizure detection algorithms. *Epilepsia*. 2025;66(Suppl 3):14–24.
   [doi:10.1111/epi.18113](https://doi.org/10.1111/epi.18113).
6. He H, Wu D. Transfer learning for brain-computer interfaces: A Euclidean
   space data alignment approach. *IEEE Transactions on Biomedical
   Engineering*. 2020;67(2):399–410.
   [doi:10.1109/TBME.2019.2913914](https://doi.org/10.1109/TBME.2019.2913914).
