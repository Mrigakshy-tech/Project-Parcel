# Project Parcel — Late Shipment Risk Prototype

## Project objective
Predict whether a shipment will arrive later than its promised delivery date, then rank parcels so a constrained operations team can review the highest-risk 10%.

**Important:** This project uses a synthetic demonstration dataset. The metrics, costs, and operational conclusions are not evidence of performance or savings in a real logistics business.

## Repository layout
```text
Project-Parcel/
├── README.md
├── requirements.txt
├── notebook/Project_Parcel_Final_Submission.ipynb
├── src/parcel_utils.py
├── tests/
│   ├── test_input_validation.py
│   ├── test_predictions.py
│   └── test_drift.py
├── reports/
│   ├── Project_Parcel_Report.pdf
│   └── Operations_Decision_Memo.pdf
├── documentation/
│   ├── FAILURES.md
│   ├── Day_1_Checkin.md
│   ├── Day_2_Checkin.md
│   └── Day_3_Checkin.md
└── walkthrough/walkthrough_script.md
```

## Run the notebook
1. Open `notebook/Project_Parcel_Final_Submission.ipynb` in Google Colab.
2. Follow the notebook's dataset-generation/download and setup cells. This prototype uses synthetic data; do not upload confidential company data.
3. Run all cells from a fresh runtime, in order.
4. Confirm the final metrics, saved model, sample predictions, drift check, and smoke checks appear.

If the notebook expects a companion synthetic-data ZIP, place that ZIP at the path requested by the notebook or update the data-loading cell to the included synthetic-data generator. Do not silently substitute a different dataset.

## Run the helper tests
Use Python 3.10+ in a local environment:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The tests cover reusable input-validation, prediction-output, and population-stability-index helpers. They are not a substitute for re-running the full Colab notebook. Before production use, connect these helpers to the exact fitted preprocessing/model pipeline and use the exact feature list and schema from the notebook.

## Label and evaluation
A shipment is labeled late when `actual_delivery_date > promised_delivery_date`. The data is split chronologically into train (5,600), validation (1,200), and test (1,200) records. A temporal split is more representative of future use than a random split because it prevents later observations from being mixed into model training.

## Model choice and operating policy
Logistic Regression was selected because its validation PR-AUC was higher than Random Forest in the recorded comparison (approximately 0.532 versus 0.487). The test top-10% ranking policy selected 120 of 1,200 shipments. Approximately 65.8% of those selected were late, but this captured only about 18.3% of all late shipments.

A cost-based threshold flagged approximately 86.4% of the validation set under illustrative assumptions, far above the assumed 10% review capacity. Therefore the top-10% ranking rule is used for the demonstration. The costs and capacity are assumptions, not measured company economics.

## Known limitations and safeguards
- Synthetic data can make patterns easier or different from real operations; external validity is unknown.
- The top-10% queue misses most late shipments; it is a prioritization aid, not a complete detector.
- Historical carrier rates and planned-load features must be proven available at dispatch time and calculated without future information.
- Drift checks on one held-out batch do not prove future stability.
- Review predictions with an operator; do not automate rerouting or customer promises from this prototype.

## What is not tested
The helper tests do not verify the full Colab environment, cloud downloads, exact model serialization compatibility, real-world data contracts, causal validity, or production latency. Those require an integration environment and representative operational data.

## AI assistance disclosure
I used ChatGPT as a coding and writing assistant to help debug pandas/scikit-learn dtype compatibility, structure documentation, and draft explanations/tests. I reviewed the outputs and remain responsible for understanding, verifying, and explaining the implementation. I will disclose any additional tools actually used.

## Reproducibility
Record the Python and package versions shown by the notebook, the notebook execution date, and the Git commit hash used for the final run. Re-run from a clean runtime before submission.
