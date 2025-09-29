# Risk Analysis Backend - Refactored Architecture

A clean, scalable risk analysis backend with proper dependency injection, comprehensive testing, and real market data integration.

## Key Architectural Improvements

### Problems Solved
- ✅ **Circular Import Dependencies**: Clean dependency injection system
- ✅ **Mixed Responsibilities**: Separated API, service, and data layers  
- ✅ **Inconsistent Error Handling**: Centralized exception management
- ✅ **Testing Gaps**: Comprehensive integration test suite
- ✅ **Configuration Management**: Environment-based configuration
- ✅ **Coupling Issues**: Proper abstraction and interfaces

### Architecture Overview
```
main.py                    # FastAPI application with lifespan management
├── api/
│   └── risk_router.py     # Clean API endpoints with validation
├── services/
│   └── risk_service.py    # Business logic layer
├── models/
│   └── risk_models.py     # Pydantic data models
├── core/
│   ├── config.py          # Configuration management
│   ├── dependencies.py    # Dependency injection
│   └── exceptions.py      # Custom exceptions
├── data/providers/
│   └── fmp_integration.py # External data integration
└── tests/
    └── test_integration.py # Comprehensive test suite
```

## Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <your-repo>
cd risk-analysis-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file in project root:
```bash
# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# FMP Data Provider (optional)
FMP_API_KEY=your_fmp_api_key_here
FMP_ENABLED=true

# Risk Analysis Settings  
RISK_FREE_RATE=0.02
CONFIDENCE_LEVELS=[0.95, 0.99]
ENABLE_CACHING=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Run Application
```bash
# Development server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Risk analysis
curl -X POST "http://localhost:8000/api/risk/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT"],
    "weights": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
    "period": "1year",
    "use_real_data": false
  }'
```

## API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with service info
- `GET /health` - Global health check
- `GET /api/risk/health` - Risk service health check

### Risk Analysis
- `POST /api/risk/analyze` - Comprehensive portfolio risk analysis
- `POST /api/risk/compare` - Compare portfolio vs benchmark risk
- `POST /api/risk/legacy/analyze` - Backwards compatibility endpoint

### Cache Management
- `GET /api/risk/cache/stats` - Cache statistics
- `POST /api/risk/cache/clear` - Clear analysis cache

## Request/Response Examples

### Risk Analysis Request
```json
{
  "symbols": ["AAPL", "GOOGL", "MSFT"],
  "weights": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
  "portfolio_id": "portfolio_001",
  "period": "1year",
  "risk_analysis_type": "comprehensive", 
  "confidence_level": 0.95,
  "use_real_data": true,
  "include_stress_testing": true
}
```

### Risk Analysis Response
```json
{
  "success": true,
  "message": "Risk analysis completed successfully",
  "data_source": "FMP Real Data",
  "execution_time": 2.34,
  "timestamp": "2025-01-15T10:30:00Z",
  "portfolio_id": "portfolio_001",
  "risk_metrics": {
    "sharpe_ratio": 1.25,
    "sortino_ratio": 1.45, 
    "max_drawdown_pct": -12.5,
    "annualized_volatility": 0.18,
    "annualized_return": 0.22,
    "value_at_risk": {
      "var_95": -0.023,
      "cvar_95": -0.031
    }
  },
  "stress_test_results": {
    "scenarios": {...},
    "worst_case_scenario": "Market Crash 2008",
    "resilience_score": 72.5
  },
  "risk_insights": [
    "Good risk-adjusted returns (Sharpe > 1.0)",
    "Moderate volatility levels"
  ]
}
```

## Testing

### Run Integration Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_integration.py::TestRiskAnalysisIntegration -v

# Run with coverage
pytest tests/ --cov=services --cov=api --cov=core
```

### Test Categories
- **Health Checks**: Service availability and configuration
- **Basic Analysis**: Core risk calculation functionality
- **Validation**: Input validation and error handling
- **Performance**: Response time and concurrent request handling
- **Error Handling**: Graceful degradation and fallback behavior

## Configuration Options

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Environment: development/test/production |
| `FMP_API_KEY` | None | Financial Modeling Prep API key |
| `FMP_ENABLED` | true | Enable/disable FMP data integration |
| `RISK_FREE_RATE` | 0.02 | Risk-free rate for calculations |
| `CONFIDENCE_LEVELS` | [0.95, 0.99] | VaR confidence levels |
| `ENABLE_CACHING` | true | Enable result caching |
| `LOG_LEVEL` | INFO | Logging level |

### Data Sources
The system supports multiple data sources with automatic fallback:

1. **FMP Real Data**: Live market data via Financial Modeling Prep API
2. **Synthetic Data**: Generated data for testing and fallback scenarios

## Deployment

### Development
```bash
python main.py
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Architecture Details

### Dependency Injection
The system uses a clean dependency injection pattern:

```python
# core/dependencies.py
def get_risk_service() -> RiskAnalysisService:
    """Get risk service with all dependencies"""
    data_manager = get_data_manager() 
    return RiskAnalysisService(data_manager=data_manager)

# FastAPI endpoint
@router.post("/analyze")
async def analyze_risk(
    request: RiskAnalysisRequest,
    service: RiskAnalysisService = Depends(get_risk_service_dependency)
):
    return await service.analyze_portfolio_risk(request)
```

### Error Handling
Centralized exception handling with proper HTTP status codes:

```python
# Custom exceptions
class RiskAnalysisError(Exception): pass
class DataProviderError(Exception): pass

# Graceful error responses
try:
    result = await service.analyze_portfolio_risk(request)
    return result
except RiskAnalysisError as e:
    raise HTTPException(status_code=400, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Data Validation
Strong typing with Pydantic models:

```python
class RiskAnalysisRequest(BaseModel):
    symbols: List[str]
    weights: Optional[Dict[str, float]] = None
    period: AnalysisPeriod = AnalysisPeriod.ONE_YEAR
    
    @validator("weights")
    def validate_weights(cls, v, values):
        if v and abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        return v
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'services'**
   - Ensure project root is in Python path
   - Check that `__init__.py` files exist in directories

2. **FMP API key errors**
   - Set `FMP_ENABLED=false` to use synthetic data
   - Check API key validity and quota limits

3. **Test failures**
   - Ensure test environment variables are set
   - Run `pytest` from project root directory

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python main.py
```

## Next Steps

### Potential Enhancements
1. **Database Integration**: Add PostgreSQL/SQLite for result persistence
2. **Redis Caching**: Implement distributed caching
3. **Async Task Queue**: Add Celery for background processing
4. **Authentication**: JWT-based authentication system
5. **Monitoring**: Prometheus metrics and health checks
6. **Documentation**: OpenAPI/Swagger documentation auto-generation

### Performance Optimization
- Connection pooling for external APIs
- Database query optimization  
- Result caching strategies
- Async processing for large portfolios

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit pull request

## License

MIT License - see LICENSE file for details