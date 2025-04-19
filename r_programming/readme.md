# Blocket Car Analysis

This project analyzes second-hand car listings from Blocket, enriched with technical data from Transportstyrelsen and additional cost estimates such as insurance and tax.

## Goals
- Predict car prices using a machine learning model (Random Forest).
- Perform statistical inference to understand what factors influence:
  - Insurance cost
  - Vehicle tax
  - Market price

## Data Sources
- **Blocket**: Used car listings (scraped or provided)
- **Transportstyrelsen**: Technical car specs
- **SCB (Statistiska centralbyrån)**: API-based vehicle registration trends
- **If Försäkring**: Insurance cost data

## Features
- Automatic data cleaning and log-transformation
- Outlier detection and removal
- Cross-validated model tuning
- Comparison of inference vs. prediction
- Residual diagnostics and variable importance analysis

## How to Run
1. Open `cars.Rmd` in RStudio.
2. Click **Knit** to generate the HTML report.
3. The final output will appear in the `docs/` folder.

## Authors
- Geisol
- Maria

---

*This project was developed as part of the course "Programmering i R".*
