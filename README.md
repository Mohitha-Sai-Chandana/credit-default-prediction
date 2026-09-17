# Credit Default Prediction

**A logistic regression model that predicts credit card default and identifies the key behavioral factors driving it, trained on 30,000 real credit card accounts.**

---
## Business Context

Lenders face a constant trade-off: extend credit too freely and default rates eat into profitability, but review every account manually and the cost and delay become unsustainable at scale. A predictive model helps bridge this gap by flagging high-risk accounts early, allowing lenders to prioritize manual review, adjust credit limits, or intervene before default occurs — rather than reacting after the fact.

## Dataset

- **Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Size:** 30,000 credit card accounts, 23 features
- **Target variable:** whether the customer defaulted on payment the following month (binary: yes/no)
- **Features include:** credit limit, demographics (age, sex, education, marital status), repayment history for the past 6 months, and bill/payment amounts

**Note:** this dataset reflects 2005 Taiwan credit card market conditions. It's used here to demonstrate risk-modeling methodology, not as a claim about current market applicability.

## Key EDA Findings

- **Recent repayment delinquency is the strongest signal.** Customers 2+ months late on their most recent payment (PAY_0) default at 69%, compared to ~13% for those current on payments — a 5x lift. Predictive strength decreases for older repayment months (PAY_2 through PAY_6), indicating recency matters more than distant history.

- **Stability in repayment behavior matters independently of current status.** A custom engineered feature (`pay_trend`, the change in repayment status from 6 months ago to the most recent month) shows customers with no change in status default at the lowest rate (16%), while both worsening and improving trends carry elevated risk (24-41% in the reliable sample range) — suggesting volatility itself is a risk signal, not just direction of change.

- **Credit limit is a strong, clean, inverse predictor.** Default rate falls monotonically from 31.8% in the lowest credit limit quintile to 13.8% in the highest — consistent with credit limits reflecting the issuing bank's own prior risk assessment.

- **Credit utilization shows a mostly monotonic relationship once a customer is actively using credit**, rising from 16.9% (low utilization) to 30.1% (over-limit). An anomalous "no/negative balance" group (24.7%) likely mixes financially conservative overpayers with low-activity accounts, illustrating that zero debt doesn't always mean zero risk.

- **Age shows a mild, non-linear pattern.** Raw per-age default rates were highly volatile past age 70 due to small sample sizes (n<15 per age group). After bucketing into ranges, a clearer, modest trend emerged: lowest risk in the 31-40 range (~20.5%), rising steadily to ~27% for 60+.

- **Demographic variables (sex, education, marital status) show only mild effects** with no clear causal explanation, and were treated as minor, secondary signals rather than headline findings.


## Model & Results

A logistic regression model was trained using `class_weight='balanced'` to address the dataset's class imbalance (78% non-default / 22% default), with features scaled via `StandardScaler`.

**Performance on held-out test data (6,000 accounts):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.70 |
| Recall (defaulters) | 0.59 |
| Precision (defaulters) | 0.38 |
| Accuracy | 0.70 |

The model correctly identifies 59% of customers who actually default, at the cost of a 38% precision rate (roughly 6 in 10 flagged accounts are false alarms). This reflects a deliberate trade-off: prioritizing recall over precision is appropriate for a use case like flagging accounts for manual review, rather than automatic credit denial, where missing a genuine defaulter is costlier than an unnecessary review. Accuracy alone (70%) is not a meaningful benchmark here, since a naive model predicting "no default" for every customer would already score ~78% while catching zero actual defaulters.

## What Drives Risk

| Feature | Coefficient | Direction |
|---|---|---|
| PAY_0 (most recent repayment status) | +0.41 | Increases risk |
| pay_trend (change in repayment status) | +0.21 | Increases risk |
| PAY_6 | +0.19 | Increases risk |
| PAY_2 | +0.13 | Increases risk |
| AGE | +0.08 | Increases risk |
| LIMIT_BAL | -0.31 | Decreases risk |
| utilization | -0.20 | Decreases risk* |

Recent repayment status (PAY_0) and the engineered `pay_trend` feature are the two strongest drivers of predicted default — confirming that both the *level* and *trajectory* of repayment behavior carry more predictive signal than any demographic factor. Credit limit is the strongest protective factor.

*Utilization's negative model coefficient appears to contradict its positive relationship with default found in EDA. This is likely due to correlation with LIMIT_BAL and BILL_AMT1 (multicollinearity) — a known limitation of interpreting individual coefficients in the presence of correlated features. The univariate EDA relationship is the more directly interpretable one.

## Limitations

- **Dataset vintage:** reflects 2005 Taiwan credit card market conditions; used to demonstrate methodology, not current market applicability.
- **Multicollinearity:** several repayment status columns (PAY_0–PAY_6) and utilization/credit limit are correlated with each other, which can distort individual coefficient interpretation (see utilization note above).
- **Modest overall performance (AUC 0.70):** typical for logistic regression on this dataset; a more complex model (e.g., gradient boosting) could likely improve predictive accuracy at the cost of interpretability.
- **Undocumented category codes:** EDUCATION and MARRIAGE contained undocumented values (0, 5, 6) that were merged into "other" categories based on common practice, not official documentation.
- **Production scorecard conversion:** this model outputs raw probabilities; a production credit risk system would typically convert this into a standardized points-based scorecard for non-technical staff use.

