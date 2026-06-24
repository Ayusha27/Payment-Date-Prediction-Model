# Invoice Payment Date Prediction Pipeline

An end-to-end Machine Learning regression solution designed to analyze historical B2B enterprise invoices and predict account clearing delays (`avg_delay` in seconds). This repository provides tracking mechanisms that enable organizations to forecast payment windows and enhance account receivable collections strategy.

## Repository Contents

As displayed in the workspace mapping file (`image_888930.png`), the project structure is organized as follows:

*   **dataset.csv**: The raw input dataset containing 50,000 transaction instances and 19 descriptive meta-features.
*   **Ayusha_Final_dataset.csv**: The finalized, structurally clean production-ready feature matrix built out during processing.
*   **HRC40654W_Ayusha_Nayak_Payment_Date_Prediction.ipynb**: Core Jupyter notebook documenting exploratory data analysis, currency conversions, historical splitting windows, hyperparameter grid search, and final metric comparisons.
*   **README.md**: Documentation detailing environment setups and software processing frameworks.

## Technical Workflow Details

### 1. Data Cleaning & Integrity Check
*   Calculated null percentages, uncovering that `area_business` was entirely empty (100% missing data) while 20% of `clear_date` points remained un-cleared (acting as our unseen test pool).
*   Identified and dropped constant metadata tags (`posting_id`, `document type`) along with high-cardinality tokens (`invoice_id`, `doc_id`) that don't add statistical value.
*   Removed duplicate records, reducing the dataset volume from 50,000 down to 48,839 unique records.

### 2. Conversions & Structural Engineering
*   **Robust Datetime Conversion:** Cleaned date strings stored as raw numerical float representations (e.g., transforming `20200210.0` into standardized `YYYY-MM-DD` timestamps).
*   **Unified Financial Metric Scale:** Normalized exchange fluctuations by transforming all Canadian Dollar amounts (`CAD`) into United States Dollars (`USD`) utilizing a fixed 0.7 exchange coefficient multiplier:
    $$\text{converted\_usd} = \text{total\_open\_amount} \times 0.7 \quad (\text{if Currency is CAD})$$

### 3. Chronological Train-Validation-Test Splitting
To simulate actual forecasting constraints, data splitting avoids random shuffle leaks and instead builds sequential time-based validation sets:
*   **Unseen Testing Pool:** Created from instances where `clear_date` contains missing values (9,681 customer records).
*   **Training and Target Engineering:** Built from non-null histories where historical clearance windows define our ground truth lag:
    $$\text{Delay} = \text{clear\_date} - \text{due\_in\_date}$$
*   **Aggregated Encoding Target:** Grouped variations by account name to produce our training reference label (`avg_delay`), which is converted to pure seconds to achieve fine-grained optimization.
*   The historical matrix was partitioned sequentially into training (60%), tracking validation (20%), and secondary verification windows (20%).

### 4. Predictive Modeling & Optimization Performance
The feature matrix was verified against several machine learning algorithms:
*   **Linear Regression** (Base comparison line)
*   **Support Vector Regression (SVR)**
*   **Decision Tree Regressor**
*   **Random Forest Ensemble**
*   **XGBoost Regressor** (Top Performer)

The models were optimized using an automated cross-validated randomized search grid (`RandomizedSearchCV`), adjusting learning rates, structural depth metrics, and sub-sampling ratios to produce optimized inference profiles over unknown commercial ledgers.

## Quickstart Guide

To boot up the runtime and execute inferences on the predictive ledger, run the following setup steps in your Python environment:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost