# Used Car Price Prediction - [Your Name]

**GitHub Repository:** https://github.com/faridhasan2020/used-car-price-prediction
**Live App:** https://used-car-price-prediction.streamlit.app

## Key EDA Findings
- Present_Price is the strongest predictor of Selling_Price (r ≈ 0.88), with a near-linear relationship.
- Selling_Price is right-skewed (mean 4.66, median 3.60 lakhs, skewness ≈ 2.5).
- Car age drives depreciation: ~90% of showroom price retained at 9 years, ~50% at 13 years, under ~30% at 17+ years.
- Diesel, Automatic and Dealer-sold cars have higher median prices.
- Kms_Driven and Owner carry little signal; Car_Name is high-cardinality and was dropped; 2 duplicate rows removed.

## Features Used
Present_Price, Kms_Driven, Car_Age (2026 − Year), Owner, Fuel_Type, Seller_Type, Transmission
(one-hot encoding for the three categorical features; scaling only for Linear/Ridge)

## CV Model Comparison (5-fold, training split only)
| Model | Train R2 (CV) | CV R2 | CV RMSE |
|---|---|---|---|
| XGBoost | 0.9972 | 0.9046 | 1.4462 |
| LightGBM | 0.9871 | 0.8952 | 1.5430 |
| Random Forest | 0.9830 | 0.8929 | 1.5195 |
| Ridge Regression | 0.9047 | 0.8890 | 1.6069 |
| Linear Regression | 0.9069 | 0.8850 | 1.6451 |

## Selected Model
**XGBoost** (n_estimators=300, learning_rate=0.05, max_depth=3)
**Final Test R2:** 0.8628
**Final Test RMSE:** 1.8804 lakhs

**Reason for Selection:** Highest mean CV R² and lowest CV RMSE. Boosted shallow trees capture the non-linear,
age-dependent depreciation that linear models miss. The train–CV gap (~0.09 R²) indicates some variance on this
small dataset, but CV performance is still the best. The test R² is slightly below CV, mostly due to two unusual
test cars (a 33-lakh Fortuner and a 20-year-old Camry).
