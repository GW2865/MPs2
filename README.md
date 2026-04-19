# MicroFragment Atlas Pro v2

A polished scientific Streamlit application for microplastic fragmentation modelling, validation, sample-level interpretation, and regional raster prediction.

## Core workflow
1. Upload the sampling CSV and define the response variable
2. Configure optional coordinates and excluded columns
3. Train the random forest model
4. Review repeated and spatial cross-validation
5. Inspect sample-level SHAP analysis
6. Explore the driving process of any selected variable with:
   - global SHAP importance
   - value–SHAP scatter
   - smoothed driver curve
   - observed feature distribution
7. Run raster prediction and uncertainty export
8. Generate a single-variable SHAP raster after raster prediction

## Design principles
- English-only interface
- Research-oriented visual design
- Sample-level interpretation before spatial projection
- Single-variable spatial SHAP mapping after prediction
- Stable cloud deployment defaults

## Deploy
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Sample-level SHAP is optional and disabled by default to reduce memory pressure
- Raster SHAP mapping is available only after raster prediction completes
- Predictor TIFF names should match retained predictor names
