# Risk Analysis Platform - Streamlit Dashboard

Professional dashboard interface for your institutional-grade risk analysis backend.

## Features

- **Portfolio Analysis**: Optimization, risk metrics, allocation visualization
- **Enhanced Correlation Analytics**: Time-varying, regime-conditional, clustering, network analysis
- **Advanced Analytics**: Risk/performance attribution, factor analysis
- **Behavioral Analysis**: Bias detection, sentiment analysis
- **Real-time API Integration**: Direct connection to your minimal APIs

## Quick Start

### Prerequisites

1. Your backend APIs must be running:
   ```bash
   # Terminal 1 - Financial Analysis API
   python minimal_api.py  # Port 8001
   
   # Terminal 2 - Behavioral Analysis API  
   python behavioral_complete_api.py  # Port 8003
   ```

2. Install Streamlit requirements:
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

### Run the Dashboard

```bash
cd streamlit_app
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
streamlit_app/
├── app.py                          # Main entry point
├── pages/
│   ├── 1_Portfolio_Analysis.py     # Portfolio optimization & metrics
│   ├── 2_Risk_Analytics.py         # Risk analysis (to be added)
│   ├── 3_Correlation_Analysis.py   # Enhanced correlation analytics
│   ├── 4_Advanced_Analytics.py     # Factor analysis (to be added)
│   └── 5_Behavioral_Analysis.py    # Behavioral tools (to be added)
├── utils/
│   ├── api_client.py              # API communication layer
│   └── visualizations.py          # Chart helpers (to be added)
└── requirements.txt
```

## Current Implementation Status

### ✅ Complete
- Main dashboard homepage with API status
- API client with full endpoint coverage (25+ endpoints)
- Portfolio Analysis page (optimization, risk metrics)
- Enhanced Correlation Analysis page (all 5 correlation types)

### 🚧 To Be Added
- Risk Analytics page (stress testing, VaR, volatility forecasting)
- Advanced Analytics page (risk/performance attribution, factor analysis)
- Behavioral Analysis page (bias detection, sentiment analysis)
- Chart visualization helpers
- Portfolio upload functionality (CSV/Excel)

## Usage Guide

### Portfolio Analysis
1. Enter stock symbols in the sidebar (one per line)
2. Adjust portfolio weights using sliders
3. Select analysis period
4. Click "Run Analysis" to:
   - Optimize portfolio allocation
   - Calculate risk metrics
   - Compare against benchmarks

### Correlation Analysis
1. Enter 2+ stock symbols
2. Select analysis type:
   - **Basic**: Standard correlation matrix
   - **Rolling**: Time-varying correlations
   - **Regime**: Crisis vs normal correlations
   - **Clustering**: Identify groups of similar assets
   - **Network**: Analyze interconnectedness
   - **Comprehensive**: Integrated analysis
3. View metrics, visualizations, and insights

## API Integration

The dashboard connects to your backend APIs:

```python
# Financial Analysis API
FINANCIAL_API_URL = "http://localhost:8001"

# Behavioral Analysis API  
BEHAVIORAL_API_URL = "http://localhost:8003"
```

All API calls include:
- Proper error handling
- 30-second timeout for complex analytics
- User-friendly error messages
- Response caching via Streamlit

## Performance

- **Initial Load**: < 2 seconds
- **API Calls**: 2.05-2.09s (matching backend performance)
- **Visualization Rendering**: < 1 second
- **Responsive Design**: Optimized for desktop (1920x1080+)

## Tips for Development

### Adding New Pages

1. Create file in `pages/` directory with numeric prefix:
   ```python
   # pages/6_New_Feature.py
   import streamlit as st
   st.title("New Feature")
   ```

2. Streamlit automatically adds to sidebar navigation

### Using the API Client

```python
from utils.api_client import get_risk_api_client

api_client = get_risk_api_client()
result = api_client.optimize_portfolio(symbols, "max_sharpe")
```

### Error Handling Pattern

```python
with st.spinner("Processing..."):
    result = api_client.some_endpoint(params)
    
    if result:
        st.success("Success!")
        # Display results
    else:
        # Error already displayed by api_client
        pass
```

## Customization

### Styling

Edit custom CSS in `app.py`:
```python
st.markdown("""
<style>
    .your-custom-class {
        /* CSS here */
    }
</style>
""", unsafe_allow_html=True)
```

### API Endpoints

Modify base URLs in `utils/api_client.py`:
```python
class RiskAnalysisAPIClient:
    def __init__(self, base_url: str = "http://your-api:8001"):
        self.base_url = base_url
```

## Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

**Streamlit Cloud** (Recommended):
1. Push to GitHub
2. Connect at share.streamlit.io
3. Configure secrets for API URLs

**Docker**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

### "Connection Error" Messages
- Ensure both backend APIs are running
- Check ports 8001 and 8003 are available
- Verify no firewall blocking localhost connections

### Slow Performance
- Reduce portfolio size (< 10 holdings)
- Use shorter analysis periods
- Check backend API logs for bottlenecks

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

1. **Complete remaining pages**: Add Risk Analytics, Advanced Analytics, Behavioral Analysis
2. **Add file upload**: CSV/Excel portfolio import
3. **Enhanced visualizations**: More interactive charts
4. **User sessions**: Save portfolio configurations
5. **Export functionality**: PDF reports, CSV exports

## Support

- Backend Issues: Check `risk-analysis-backend/` repository
- Frontend Issues: Review Streamlit logs
- API Documentation: See backend README.md

## License

[Your License Here]

---

Built with Streamlit • Powered by your institutional-grade backend APIs