# Open Questions for Professor

These questions need to be resolved before or shortly after the May 12 interim presentation.

## Critical (affects implementation)

1. **"Verified" software**: What does this mean in the critical infrastructure context?
   - EU AI Act compliant libraries?
   - Well-established open-source packages (pandas, scikit-learn)?
   - An explicit approved list?
   - Something with formal certification?

2. **Dataset source**: Is the intended data source the ENTSO-E Transparency Platform (via API key), or will a separate dataset file be distributed?
   - We are assuming ENTSO-E and have applied for an API key.
   - We are assuming the Germany (DE-LU) bidding zone — is that correct, or is a specific region assigned?

3. **Forecast specification**:
   - We are assuming a **24-hour horizon** — is that correct?
   - We are assuming **1-hour resolution** — is that correct?

## Important (affects evaluation)

4. **Baseline definition**: Is the baseline formally defined as "predict using the same hour from one week prior" (7-day seasonal naive), or is each team responsible for defining and justifying it?

5. **Team ranking**: How will teams be compared?
   - Same held-out test set with a fixed metric?
   - Each team defines their own evaluation setup and justifies it?
   - Which metric is primary (MAE, RMSE, MAPE)?

## Administrative

6. **Submission format**: Is there a required deliverable format (GitHub repo, written report, notebook), or is this the team's choice?
