# 5–8 minute walkthrough script

**0:00–0:40 — Problem and scope**  
“This project predicts whether a parcel arrives later than its promised date. The dataset is synthetic, so the results are a demonstration, not a claim about a real carrier network.”

**0:40–1:30 — Data audit and label**  
“Late means actual delivery date is after promised delivery date. I checked missingness, duplicate shipment IDs, joins, and category consistency. I used a chronological split to mimic future prediction.”

**1:30–2:30 — Model comparison**  
“I compared Logistic Regression with Random Forest. Logistic Regression had higher validation PR-AUC, approximately 0.532 versus 0.487, so I selected it. I prioritized PR-AUC and top-of-queue precision because operations has limited review capacity.”

**2:30–3:30 — Results and business policy**  
“On the test set, the top 10% queue had about 65.8% precision but only 18.3% recall. It concentrates risk but misses most late parcels. The illustrative cost threshold flagged about 86.4% of validation shipments, so it was not operationally feasible under a 10% capacity assumption.”

**3:30–4:30 — Explainability and monitoring**  
“Show one flagged parcel and its plain-language explanation. Explain that feature importance is descriptive, not causal. Show the drift check and the deliberate shifted-data example.”

**4:30–5:30 — Failure and recovery**  
“I encountered a pandas nullable-value compatibility error with scikit-learn. I normalized dtypes consistently across train, validation, and test. I also rejected the impractically large threshold-based queue.”

**5:30–6:30 — Tests, limits, and next steps**  
“Show the input validation and pytest tests. Explain that these helper tests need integration with the exact model schema. The next step is point-in-time validation on real data, real cost estimates, and a prospective shadow pilot.”

**Before recording:** Run the notebook from a clean Colab runtime, show the actual outputs from that run, and do not read out metrics that differ from the executed notebook. Replace this script if your final rerun changes the results.
