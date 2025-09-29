# models/correlation_models.py
"""
Correlation Models - Enhanced Correlation Analysis
=================================================

Pydantic models for enhanced correlation analysis capabilities including:
- Rolling correlation analysis with stability metrics
- Regime-conditional correlation matrices
- Hierarchical clustering analysis
- Network analysis with centrality measures
- Comprehensive multi-method correlation analysis

Follows the split models architecture pattern for maintainability.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base_models import BaseRequestModel, BaseResponseModel, AnalysisPeriod

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CorrelationMethod(str, Enum):
    """Available correlation calculation methods"""
    PEARSON = "pearson"
    SPEARMAN = "spearman" 
    KENDALL = "kendall"

class ClusteringMethod(str, Enum):
    """Available clustering methods for correlation analysis"""
    WARD = "ward"
    COMPLETE = "complete"
    AVERAGE = "average"
    SINGLE = "single"

class NetworkCentrality(str, Enum):
    """Network centrality measures"""
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"

# =============================================================================
# REQUEST MODELS
# =============================================================================

class RollingCorrelationRequest(BaseRequestModel):
    """Request model for rolling correlation analysis"""
    symbols: List[str] = Field(..., min_items=2, max_items=20)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR)
    window_days: int = Field(default=60, ge=20, le=252)
    correlation_method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON)
    use_real_data: bool = Field(default=True)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        """Validate symbol list"""
        if len(set(v)) != len(v):
            raise ValueError("Duplicate symbols not allowed")
        return [s.upper().strip() for s in v]

class RegimeCorrelationRequest(BaseRequestModel):
    """Request model for regime-conditional correlation analysis"""
    symbols: List[str] = Field(..., min_items=2, max_items=20)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR)
    correlation_method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON)
    regime_detection_method: str = Field(default="hmm", pattern="^(hmm|volatility|returns)$")
    use_real_data: bool = Field(default=True)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        return [s.upper().strip() for s in v]

class CorrelationClusteringRequest(BaseRequestModel):
    """Request model for hierarchical correlation clustering"""
    symbols: List[str] = Field(..., min_items=3, max_items=20)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR)
    clustering_method: ClusteringMethod = Field(default=ClusteringMethod.WARD)
    max_clusters: Optional[int] = Field(default=None, ge=2, le=10)
    correlation_method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON)
    use_real_data: bool = Field(default=True)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        return [s.upper().strip() for s in v]
    
    @validator('max_clusters')
    def validate_max_clusters(cls, v, values):
        if v is not None and 'symbols' in values and v >= len(values['symbols']):
            raise ValueError("max_clusters must be less than number of symbols")
        return v

class CorrelationNetworkRequest(BaseRequestModel):
    """Request model for correlation network analysis"""
    symbols: List[str] = Field(..., min_items=3, max_items=20)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR)
    correlation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    correlation_method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON)
    centrality_measures: List[NetworkCentrality] = Field(default=[NetworkCentrality.DEGREE])
    use_real_data: bool = Field(default=True)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        return [s.upper().strip() for s in v]

class ComprehensiveCorrelationRequest(BaseRequestModel):
    """Request model for comprehensive correlation analysis combining all methods"""
    symbols: List[str] = Field(..., min_items=3, max_items=15)
    period: AnalysisPeriod = Field(default=AnalysisPeriod.ONE_YEAR)
    window_days: int = Field(default=60, ge=20, le=252)
    correlation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    correlation_method: CorrelationMethod = Field(default=CorrelationMethod.PEARSON)
    clustering_method: ClusteringMethod = Field(default=ClusteringMethod.WARD)
    include_rolling: bool = Field(default=True)
    include_regime: bool = Field(default=True)
    include_clustering: bool = Field(default=True)
    include_network: bool = Field(default=True)
    use_real_data: bool = Field(default=True)
    
    @validator('symbols')
    def validate_symbols(cls, v):
        return [s.upper().strip() for s in v]

# =============================================================================
# RESPONSE DATA MODELS
# =============================================================================

class StabilityMetrics(BaseModel):
    """Correlation stability metrics"""
    avg_correlation_change: float = Field(..., description="Average correlation change between windows")
    max_correlation_change: float = Field(..., description="Maximum correlation change observed")
    correlation_volatility: float = Field(..., description="Standard deviation of correlation changes")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Overall stability score (0-1)")

class RegimeSensitivity(BaseModel):
    """Regime sensitivity metrics"""
    bull_bear_correlation_difference: float = Field(..., description="Difference in avg correlation between bull/bear")
    crisis_correlation_multiplier: float = Field(..., description="Crisis correlation multiplier vs normal times")
    regime_sensitivity_score: float = Field(..., ge=0.0, description="Overall regime sensitivity")

class ClusterQualityMetrics(BaseModel):
    """Clustering quality metrics"""
    optimal_clusters: int = Field(..., ge=1, description="Optimal number of clusters")
    clustering_efficiency: float = Field(..., description="Clustering quality score")
    total_explained_correlation: float = Field(..., description="Correlation explained by clustering")
    silhouette_score: Optional[float] = Field(None, description="Silhouette score if available")

class NetworkHealthMetrics(BaseModel):
    """Network health and connectivity metrics"""
    network_density: float = Field(..., ge=0.0, le=1.0, description="Network density (0-1)")
    average_clustering: float = Field(..., ge=0.0, le=1.0, description="Average clustering coefficient")
    network_efficiency: float = Field(..., ge=0.0, le=1.0, description="Network efficiency score")
    total_connections: int = Field(..., ge=0, description="Total network connections")
    connectivity_score: float = Field(..., ge=0.0, le=1.0, description="Connectivity strength (0-1)")
    diversification_potential: float = Field(..., ge=0.0, le=1.0, description="Diversification potential (0-1)")

class DegreeCentralityMetrics(BaseModel):
    """Degree centrality metrics for individual assets"""
    degree: int = Field(..., ge=0, description="Number of connections")
    normalized_degree: float = Field(..., ge=0.0, le=1.0, description="Normalized degree centrality")
    centrality_rank: int = Field(..., ge=1, description="Centrality ranking (1=highest)")

class SystemicImportanceMetrics(BaseModel):
    """Systemic importance metrics for individual assets"""
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Systemic importance score")
    systemic_rank: int = Field(..., ge=1, description="Systemic importance ranking")
    risk_contribution: float = Field(..., ge=0.0, description="Risk contribution to system")

# =============================================================================
# RESPONSE MODELS
# =============================================================================

class RollingCorrelationResponse(BaseResponseModel):
    """Response model for rolling correlation analysis"""
    analysis_type: str = Field(default="rolling_correlations")
    window_days: int = Field(..., description="Rolling window size in days")
    total_windows: int = Field(..., description="Total number of rolling windows analyzed")
    stability_metrics: StabilityMetrics = Field(..., description="Correlation stability metrics")
    pair_correlations: Dict[str, Dict[str, Any]] = Field(..., description="Pairwise correlation time series")
    correlation_breakpoints: List[Dict[str, Any]] = Field(..., description="Significant correlation changes")
    latest_correlation_matrix: Optional[List[List[float]]] = Field(None, description="Most recent correlation matrix")
    symbols: List[str] = Field(..., description="Analyzed symbols")
    data_source: str = Field(..., description="Data source used")

class RegimeCorrelationResponse(BaseResponseModel):
    """Response model for regime-conditional correlation analysis"""
    analysis_type: str = Field(default="regime_conditional_correlations")
    market_regime_correlations: Dict[str, Dict[str, Any]] = Field(..., description="Correlations by market regime")
    volatility_regime_correlations: Dict[str, Dict[str, Any]] = Field(..., description="Correlations by volatility regime")
    regime_sensitivity: RegimeSensitivity = Field(..., description="Regime sensitivity metrics")
    symbols: List[str] = Field(..., description="Analyzed symbols")
    total_periods: int = Field(..., description="Total periods analyzed")
    data_source: str = Field(..., description="Data source used")

class CorrelationClusteringResponse(BaseResponseModel):
    """Response model for hierarchical correlation clustering"""
    analysis_type: str = Field(default="hierarchical_correlation_clustering")
    optimal_clusters: int = Field(..., description="Optimal number of clusters")
    cluster_assignments: List[int] = Field(..., description="Cluster assignment for each symbol")
    cluster_analysis: Dict[str, Dict[str, Any]] = Field(..., description="Analysis of each cluster")
    dendrogram_data: Dict[str, Any] = Field(..., description="Dendrogram visualization data")
    quality_metrics: ClusterQualityMetrics = Field(..., description="Clustering quality metrics")
    symbols: List[str] = Field(..., description="Analyzed symbols")
    data_source: str = Field(..., description="Data source used")

class CorrelationNetworkResponse(BaseResponseModel):
    """Response model for correlation network analysis"""
    analysis_type: str = Field(default="correlation_network_analysis")
    correlation_threshold: float = Field(..., description="Correlation threshold used")
    network_health: NetworkHealthMetrics = Field(..., description="Network health metrics")
    degree_centrality: Dict[str, DegreeCentralityMetrics] = Field(..., description="Degree centrality by symbol")
    clustering_coefficients: Dict[str, float] = Field(..., description="Clustering coefficients by symbol")
    systemic_importance: Dict[str, SystemicImportanceMetrics] = Field(..., description="Systemic importance by symbol")
    adjacency_matrix: List[List[int]] = Field(..., description="Network adjacency matrix")
    symbols: List[str] = Field(..., description="Analyzed symbols")
    data_source: str = Field(..., description="Data source used")

class ComprehensiveCorrelationResponse(BaseResponseModel):
    """Response model for comprehensive correlation analysis"""
    analysis_type: str = Field(default="comprehensive_correlation_analysis")
    rolling_analysis: Optional[Dict[str, Any]] = Field(None, description="Rolling correlation results")
    regime_analysis: Optional[Dict[str, Any]] = Field(None, description="Regime correlation results")
    clustering_analysis: Optional[Dict[str, Any]] = Field(None, description="Clustering analysis results")
    network_analysis: Optional[Dict[str, Any]] = Field(None, description="Network analysis results")
    synthesized_insights: List[str] = Field(..., description="Key insights from combined analysis")
    symbols: List[str] = Field(..., description="Analyzed symbols")
    analyses_completed: int = Field(..., description="Number of analyses completed")
    data_source: str = Field(..., description="Data source used")

# =============================================================================
# EXAMPLE REQUESTS
# =============================================================================

CORRELATION_EXAMPLE_REQUESTS = {
    "rolling_correlation": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "period": "1year",
        "window_days": 60,
        "correlation_method": "pearson",
        "use_real_data": True
    },
    "regime_correlation": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "period": "1year",
        "correlation_method": "pearson",
        "regime_detection_method": "hmm",
        "use_real_data": True
    },
    "correlation_clustering": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"],
        "period": "1year",
        "clustering_method": "ward",
        "max_clusters": 3,
        "correlation_method": "pearson",
        "use_real_data": True
    },
    "correlation_network": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META"],
        "period": "1year",
        "correlation_threshold": 0.5,
        "correlation_method": "pearson",
        "centrality_measures": ["degree"],
        "use_real_data": True
    },
    "comprehensive_correlation": {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"],
        "period": "1year",
        "window_days": 60,
        "correlation_threshold": 0.5,
        "correlation_method": "pearson",
        "clustering_method": "ward",
        "include_rolling": True,
        "include_regime": True,
        "include_clustering": True,
        "include_network": True,
        "use_real_data": True
    }
}

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CorrelationMethod", "ClusteringMethod", "NetworkCentrality",
    
    # Request Models
    "RollingCorrelationRequest", "RegimeCorrelationRequest",
    "CorrelationClusteringRequest", "CorrelationNetworkRequest",
    "ComprehensiveCorrelationRequest",
    
    # Response Data Models
    "StabilityMetrics", "RegimeSensitivity", "ClusterQualityMetrics",
    "NetworkHealthMetrics", "DegreeCentralityMetrics", "SystemicImportanceMetrics",
    
    # Response Models
    "RollingCorrelationResponse", "RegimeCorrelationResponse",
    "CorrelationClusteringResponse", "CorrelationNetworkResponse",
    "ComprehensiveCorrelationResponse",
    
    # Examples
    "CORRELATION_EXAMPLE_REQUESTS"
]