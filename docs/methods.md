# Methods and statistical protocol

This note defines the computation performed by the repository. It is intended
to make design choices auditable, not to imply that any particular result has
already been reproduced.

## Continuous EEG windows and labels

For recording \(r\), let \(X_r \in \mathbb{R}^{C\times N_r}\) be filtered
signals with \(C=18\) channels and sampling frequency \(f_s\). The fixed window
length is \(T=4\) seconds, so the window sample count is
\(L=\operatorname{round}(Tf_s)\). Windows are contiguous and non-overlapping:

\[
W_{r,k}=X_r[:,\,kL:(k+1)L],\qquad t_{r,k}=kT.
\]

For an annotated seizure interval \([a_j,b_j]\), a window is ictal only if
\([t_{r,k},t_{r,k}+T]\subseteq[a_j,b_j]\). A window that intersects a seizure
but is not fully contained is dropped as an onset/offset-ambiguous example.
Every other complete window is non-ictal. The label is therefore not inferred
from a model score or from nearby labels.

Signals are band-pass filtered from 0.5 to 45 Hz before windowing. This is an
offline retrospective filter; MNE's default zero-phase FIR filtering can use
future samples around a time point. The supplied experiments do not establish
causal/real-time performance.

## Feature representation

Per channel, the compact feature vector includes log standard deviation, log
RMS, log mean absolute first difference, skew, excess kurtosis, log activity,
log Hjorth mobility, and log Hjorth complexity. For centered channel signal
\(x\), Hjorth mobility is
\(\sigma_{\Delta x}/\sigma_x\), and complexity is
\((\sigma_{\Delta^2x}/\sigma_{\Delta x})/
(\sigma_{\Delta x}/\sigma_x)\). Small positive floors protect logarithms and
ratios from zero; non-finite feature values are explicitly converted to zero.

Welch power spectral density \(S_c(f)\) is integrated over delta (0.5–4 Hz),
theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–45 Hz) bands.
Both log absolute band power and power divided by the total five-band power
are included.

For each window, the sample covariance is regularized:

\[
C = \frac{(W-\bar W)(W-\bar W)^\top}{L-1}
  + \lambda I,\qquad
\lambda=\max(10^{-6}\operatorname{tr}(C_0)/C,10^{-10}),
\]

where \(C_0\) is the unregularized covariance. The matrix logarithm is computed
from its symmetric eigendecomposition. Its upper triangle is vectorized with
off-diagonal entries multiplied by \(\sqrt{2}\), preserving the matrix
Frobenius norm. These off-diagonal log-covariance terms encode channel
co-variation/connectivity; they are not a separate coherence estimate.

## Euclidean alignment

For the optional alignment ablation, divide the first 60 seconds of each
recording into four-second baseline segments \(B_i\). For each segment compute
the trace-normalized covariance
\[
\widetilde C_i=C_i\,C/\operatorname{tr}(C_i),
\]
where \(C\) is the number of channels. The record reference and transform are
\[
R=\frac{1}{m}\sum_{i=1}^{m}\widetilde C_i,\qquad
A=(R+\epsilon I)^{-1/2},\qquad
X_{\mathrm{aligned}}=AX.
\]
The inverse square root uses a symmetric eigendecomposition with a positive
eigenvalue floor. No labels are used to estimate \(R\). This uses the initial
segment of the current test record and is explicitly record-adaptive; it is
not evidence of strict causal deployment. The fixed first-60-second interval
may contain seizure activity or artifacts; neither is screened using target
annotations. The no-alignment comparator uses no target-record signal for
adaptation, but excludes the same warm-up interval from its evaluation.

For the alignment comparison, a diagnostic feature-mean distance is computed
using the source-training feature vectors and unlabeled target-record windows
fully contained within the configured reference interval. If
\(\mu_s,\sigma_s\) are the per-feature mean and standard deviation of source
training features and \(\mu_t\) is the mean of target calibration features,
the reported value is
\[
d_{\mathrm{shift}} =
\sqrt{\frac{1}{p}\sum_{j=1}^{p}
\left(\frac{\mu_{t,j}-\mu_{s,j}}{\max(\sigma_{s,j},10^{-8})}\right)^2}.
\]
It is a descriptive RMS standardized mean difference, not a full
distributional distance and not an adaptation decision rule. Values are
calculated separately under each signal configuration, since alignment
changes the feature representation. For evaluation, only complete windows
whose start time is at or after the calibration boundary are retained, for
both aligned and unaligned runs. Validation probabilities used to choose the
alarm threshold are restricted the same way. A seizure contributes to event
sensitivity only if its annotated onset is at or after that boundary; a
seizure crossing the boundary contributes its post-boundary interval to ictal
exposure and alarm overlap but is excluded from the event-sensitivity
denominator. Thus the target's warm-up interval is excluded from reported
validation and test metrics. The protocol remains retrospective because the
filter is zero-phase, and the aligned run's transform is estimated using
unlabeled initial target-record data.

## Imbalance handling and classical model selection

For a training split with \(n_0\) non-ictal and \(n_1\) ictal windows, training
only random undersampling retains
\[
n'_0=\min(n_0,\lceil \rho n_1\rceil), \quad \rho=5
\]
non-ictal examples. It is part of an imbalanced-learn estimator pipeline, so
each inner-training fold is independently sampled. Validation and test data
are never resampled. Class weights are also enabled for the classifier.

The classical baselines are logistic regression, RBF-SVM, random forest,
Extra Trees, and histogram gradient boosting. Their small prespecified
parameter grids are selected by average precision within up to four
`StratifiedGroupKFold` splits grouped by EDF. At least three independent
training recordings are required; each fold must preserve both classes. A
fold that cannot do so fails rather than falling back to a random-window
split.

The outer test unit is the patient: in each leave-one-subject-out fold, all EDF
files from one patient are withheld. A disjoint recording-grouped validation
subset of the remaining patients selects the alarm threshold. The final test
model is fit on the remaining training recordings (not the threshold
validation recordings), so the model that scores validation and test is the
same estimator. The validation split first seeks seizure-bearing and
seizure-free recordings on both sides. If a small cohort has no seizure-free
recordings, it may instead use a recording-disjoint split with both window
classes on both sides. This fallback does not synthesize control records or
split windows from one EDF across partitions. In a one-recording-per-patient
pilot, the EDF grouping also keeps the validation patient disjoint.

## Alarm rule and threshold

At threshold \(\tau\), windows with \(p_{r,k}\ge\tau\) become positive. Adjacent
positive windows, and positive runs separated by no more than the configured
gap \(g\), are merged. Runs shorter than minimum duration \(d\) are removed.
Both \(g\) and \(d\) are fixed before outer-test scoring.

On validation recordings, select the threshold maximizing event sensitivity
subject to false alarms/hour not exceeding the target. Ties prefer higher time
precision and then the higher threshold. If no threshold meets the target, the
minimum-false-alarm threshold is selected, with sensitivity/time precision as
tie-breakers, and the unmet target is recorded. No outer-test labels or
probabilities affect \(\tau\). The search evaluates up to 201 unique-score
quantiles (or every unique score when there are fewer), bounding computation
for long recordings.

## Primary and secondary estimands

Let \(\mathcal S\) be annotated seizure events and \(\mathcal A\) postprocessed
alarms:

- **Event sensitivity:** the fraction of seizures whose interval has any
  positive-duration overlap with at least one alarm.
- **False alarms/hour:** alarm runs with no overlap with any seizure, divided
  by total non-seizure recording hours. A partially overlapping alarm counts
  as a seizure-covering alarm, not a false alarm.
- **Time precision:** total alarm seconds overlapping seizure intervals
  divided by total alarm seconds.
- **Window precision, recall, F1, AUROC, and average precision:** secondary
  window-level statistics; AUROC is never used alone as a continuous
  monitoring claim.

Patient-level confidence intervals resample held-out patients with
replacement, so adjacent windows are not treated as independent replicates.
Paired model comparisons use the same held-out patients and bootstrap their
within-patient metric differences.

## Bonn segment benchmark

Bonn is an ictal/non-ictal segment discrimination task, distinct from
continuous monitoring. The manifest must associate each derived segment with
its true source recording group. Five outer grouped folds are used with four
inner grouped folds. If original source-record identity is unavailable, a
recording-grouped result cannot be reconstructed from a random segment split.
The optional random-window ablation is deliberately labelled as leaky and
invalid for unseen-recording generalization.

The compact F/S example manifest assigns one group to each text segment because
the pinned mirror does not provide original subject or source-recording IDs.
Its held-out-segment metrics are real-data software examples, but are not
source-independent or patient-independent estimates and must not be presented
as seizure-onset performance.

## EEGPT transfer experiment

EEGPT is loaded from the public Braindecode model
`braindecode/eegpt-pretrained`; it is reinitialized from that checkpoint for
each outer patient fold and fine-tuned on training recordings. A recording
disjoint validation split chooses the epoch and alarm threshold. Training
negative subsampling is capped at the configured ratio; validation/test retain
natural prevalence.

The checkpoint expects 250 Hz, four-second windows. The CHB-MIT signal is
resampled to 250 Hz for this experiment. Because the 18 inputs are bipolar
derivations while EEGPT's spatial representation is electrode-based, the code
maps each derivation to an approximate electrode midpoint and requests
Braindecode's spline channel strategy. This is a material representation
change. Report the checkpoint, library versions, input scale, and channel
mapping; compare with classical results only on identical patients,
recordings, preprocessing, and alarm settings.

## Claim boundaries

The Bonn result cannot substitute for unseen-patient continuous evaluation.
This code does not import historical pilot or benchmark numbers as output,
labels, or fitted parameters. Clinical utility, real-time causality, external
validity, and acceptable false-alarm burden require independent prospective
research and expert review.
