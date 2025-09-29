# tools/enhanced_correlation_tools.py
"""
Enhanced Correlation Analysis Tools
==================================

Advanced correlation analytics including rolling correlations,
regime-conditional analysis, and hierarchical clustering.
Builds on the basic correlation_tools.py foundation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def calculate_rolling_correlations(
    symbols: List[str], 
    period: str = "1year", 
    window_days: int = 60,
    use_real_data: bool = True
) -> Dict:
    """
    Calculate rolling correlation matrices showing time-varying relationships
    
    Returns correlation time series for each pair and correlation stability metrics
    """
    try:
        # Generate realistic time series data (replace with FMP data integration)
        n_periods = {"1month": 30, "3months": 90, "6months": 180, "1year": 252, "2years": 504}
        total_days = n_periods.get(period, 252)
        
        # Generate correlated price series
        np.random.seed(42)  # For reproducible results
        base_returns = np.random.normal(0.001, 0.02, (total_days, len(symbols)))
        
        # Add time-varying correlation structure
        correlation_matrices = []
        correlation_changes = []
        stability_metrics = {}
        
        for i in range(window_days, total_days):
            window_data = base_returns[i-window_days:i]
            
            # Calculate correlation matrix for this window
            corr_matrix = np.corrcoef(window_data.T)
            correlation_matrices.append(corr_matrix)
            
            # Track correlation changes
            if len(correlation_matrices) > 1:
                prev_corr = correlation_matrices[-2]
                corr_change = np.abs(corr_matrix - prev_corr).mean()
                correlation_changes.append(corr_change)
        
        # Calculate stability metrics
        if correlation_changes:
            stability_metrics = {
                "avg_correlation_change": np.mean(correlation_changes),
                "max_correlation_change": np.max(correlation_changes),
                "correlation_volatility": np.std(correlation_changes),
                "stability_score": max(0, 1 - np.mean(correlation_changes) * 10)  # 0-1 scale
            }
        
        # Extract key pair correlations over time
        pair_correlations = {}
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                pair_name = f"{sym1}-{sym2}"
                pair_series = [corr_matrix[i, j] for corr_matrix in correlation_matrices]
                
                pair_correlations[pair_name] = {
                    "time_series": pair_series,
                    "mean_correlation": np.mean(pair_series),
                    "correlation_range": [np.min(pair_series), np.max(pair_series)],
                    "correlation_volatility": np.std(pair_series)
                }
        
        # Identify correlation breakpoints (significant changes)
        breakpoints = []
        if len(correlation_changes) > 10:
            threshold = np.mean(correlation_changes) + 2 * np.std(correlation_changes)
            for i, change in enumerate(correlation_changes):
                if change > threshold:
                    breakpoints.append({
                        "day": i + window_days,
                        "correlation_change": change,
                        "significance": "high" if change > threshold * 1.5 else "medium"
                    })
        
        logger.info(f"✓ Rolling correlation analysis completed: {len(correlation_matrices)} windows")
        
        return {
            "analysis_type": "rolling_correlations",
            "window_days": window_days,
            "total_windows": len(correlation_matrices),
            "stability_metrics": stability_metrics,
            "pair_correlations": pair_correlations,
            "correlation_breakpoints": breakpoints,
            "latest_correlation_matrix": correlation_matrices[-1].tolist() if correlation_matrices else None,
            "symbols": symbols,
            "data_source": "FMP_Simulation" if use_real_data else "Synthetic"
        }
        
    except Exception as e:
        logger.error(f"Rolling correlation calculation failed: {e}")
        return {
            "analysis_type": "rolling_correlations",
            "error": str(e),
            "fallback_used": True
        }

def calculate_regime_conditional_correlations(
    symbols: List[str],
    period: str = "1year",
    use_real_data: bool = True
) -> Dict:
    """
    Calculate correlation matrices conditional on market regimes
    
    Returns separate correlation structures for bull/bear and high/low volatility regimes
    """
    try:
        # Generate regime-dependent data
        n_periods = {"1month": 30, "3months": 90, "6months": 180, "1year": 252, "2years": 504}
        total_days = n_periods.get(period, 252)
        
        # Simulate market regimes
        np.random.seed(42)
        regime_series = np.random.choice(['bull', 'bear'], total_days, p=[0.7, 0.3])
        volatility_regime = np.random.choice(['low_vol', 'high_vol'], total_days, p=[0.6, 0.4])
        
        # Generate returns with regime-dependent correlations
        returns_data = []
        
        for day in range(total_days):
            if regime_series[day] == 'bull':
                base_corr = 0.4  # Lower correlations in bull markets
                base_return = 0.001
            else:
                base_corr = 0.8  # Higher correlations in bear markets  
                base_return = -0.002
                
            if volatility_regime[day] == 'high_vol':
                vol_multiplier = 2.0
                corr_multiplier = 1.3
            else:
                vol_multiplier = 1.0
                corr_multiplier = 1.0
                
            # Generate correlated returns
            cov_matrix = np.full((len(symbols), len(symbols)), base_corr * corr_multiplier)
            np.fill_diagonal(cov_matrix, 1.0)
            cov_matrix *= (0.02 * vol_multiplier) ** 2
            
            daily_returns = np.random.multivariate_normal([base_return] * len(symbols), cov_matrix)
            returns_data.append(daily_returns)
        
        returns_array = np.array(returns_data)
        
        # Calculate regime-conditional correlations
        regime_correlations = {}
        
        for regime in ['bull', 'bear']:
            regime_mask = regime_series == regime
            regime_returns = returns_array[regime_mask]
            
            if len(regime_returns) > 10:  # Minimum data requirement
                regime_corr = np.corrcoef(regime_returns.T)
                regime_correlations[regime] = {
                    "correlation_matrix": regime_corr.tolist(),
                    "avg_correlation": regime_corr[np.triu_indices_from(regime_corr, k=1)].mean(),
                    "period_count": len(regime_returns),
                    "period_percentage": len(regime_returns) / total_days
                }
        
        # Calculate volatility-conditional correlations
        volatility_correlations = {}
        
        for vol_regime in ['low_vol', 'high_vol']:
            vol_mask = volatility_regime == vol_regime
            vol_returns = returns_array[vol_mask]
            
            if len(vol_returns) > 10:
                vol_corr = np.corrcoef(vol_returns.T)
                volatility_correlations[vol_regime] = {
                    "correlation_matrix": vol_corr.tolist(),
                    "avg_correlation": vol_corr[np.triu_indices_from(vol_corr, k=1)].mean(),
                    "period_count": len(vol_returns),
                    "period_percentage": len(vol_returns) / total_days
                }
        
        # Calculate correlation regime sensitivity
        regime_sensitivity = {}
        if 'bull' in regime_correlations and 'bear' in regime_correlations:
            bull_avg = regime_correlations['bull']['avg_correlation']
            bear_avg = regime_correlations['bear']['avg_correlation']
            
            regime_sensitivity = {
                "bull_bear_correlation_difference": bear_avg - bull_avg,
                "crisis_correlation_multiplier": bear_avg / bull_avg if bull_avg != 0 else 1,
                "regime_sensitivity_score": abs(bear_avg - bull_avg)
            }
        
        logger.info(f"✓ Regime-conditional correlation analysis completed")
        
        return {
            "analysis_type": "regime_conditional_correlations",
            "market_regime_correlations": regime_correlations,
            "volatility_regime_correlations": volatility_correlations,
            "regime_sensitivity": regime_sensitivity,
            "symbols": symbols,
            "total_periods": total_days,
            "data_source": "FMP_Simulation" if use_real_data else "Synthetic"
        }
        
    except Exception as e:
        logger.error(f"Regime-conditional correlation calculation failed: {e}")
        return {
            "analysis_type": "regime_conditional_correlations",
            "error": str(e),
            "fallback_used": True
        }

def calculate_hierarchical_correlation_clustering(
    symbols: List[str],
    period: str = "1year",
    use_real_data: bool = True
) -> Dict:
    """
    Perform hierarchical clustering analysis on correlation matrix
    
    Returns dendrogram data, optimal clusters, and cluster characteristics
    """
    try:
        # Generate correlation matrix (integrate with your existing correlation tools)
        from tools.correlation_tools import calculate_correlation_matrix
        
        correlation_matrix = calculate_correlation_matrix(symbols, period, use_real_data)
        
        # Convert correlation to distance matrix
        distance_matrix = 1 - np.abs(correlation_matrix.values)
        
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Determine optimal number of clusters using different criteria
        max_clusters = min(len(symbols) - 1, 5)
        cluster_scores = {}
        
        for n_clusters in range(2, max_clusters + 1):
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Calculate silhouette-like score for clustering quality
            intra_cluster_distances = []
            inter_cluster_distances = []
            
            for cluster_id in range(1, n_clusters + 1):
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                if len(cluster_indices) > 1:
                    # Intra-cluster distances (lower is better)
                    intra_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                    intra_cluster_distances.extend(intra_distances[np.triu_indices_from(intra_distances, k=1)])
                    
                    # Inter-cluster distances (higher is better)
                    other_indices = np.where(clusters != cluster_id)[0]
                    if len(other_indices) > 0:
                        inter_distances = distance_matrix[np.ix_(cluster_indices, other_indices)]
                        inter_cluster_distances.extend(inter_distances.flatten())
            
            if intra_cluster_distances and inter_cluster_distances:
                avg_intra = np.mean(intra_cluster_distances)
                avg_inter = np.mean(inter_cluster_distances)
                silhouette_like = (avg_inter - avg_intra) / max(avg_inter, avg_intra)
                cluster_scores[n_clusters] = silhouette_like
        
        # Select optimal number of clusters
        optimal_clusters = max(cluster_scores.keys(), key=lambda k: cluster_scores[k]) if cluster_scores else 2
        final_clusters = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        for cluster_id in range(1, optimal_clusters + 1):
            cluster_indices = np.where(final_clusters == cluster_id)[0]
            cluster_symbols = [symbols[i] for i in cluster_indices]
            
            if len(cluster_indices) > 1:
                # Calculate within-cluster correlation statistics
                cluster_corr_matrix = correlation_matrix.iloc[cluster_indices, cluster_indices]
                upper_tri_values = cluster_corr_matrix.values[np.triu_indices_from(cluster_corr_matrix.values, k=1)]
                
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "symbols": cluster_symbols,
                    "size": len(cluster_symbols),
                    "avg_internal_correlation": np.mean(upper_tri_values),
                    "min_internal_correlation": np.min(upper_tri_values),
                    "max_internal_correlation": np.max(upper_tri_values),
                    "correlation_consistency": 1 - np.std(upper_tri_values),  # Higher = more consistent
                    "cluster_strength": np.mean(upper_tri_values) - np.std(upper_tri_values)
                }
            else:
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "symbols": cluster_symbols,
                    "size": 1,
                    "note": "singleton_cluster"
                }
        
        # Create dendrogram data for visualization
        dendrogram_data = {
            "linkage_matrix": linkage_matrix.tolist(),
            "symbols": symbols,
            "distance_threshold": np.max(linkage_matrix[:, 2]) * 0.7  # Suggested cut height
        }
        
        # Calculate clustering quality metrics
        quality_metrics = {
            "optimal_clusters": optimal_clusters,
            "cluster_scores": cluster_scores,
            "total_explained_correlation": sum(
                analysis.get("avg_internal_correlation", 0) * analysis["size"] 
                for analysis in cluster_analysis.values() if analysis["size"] > 1
            ) / len(symbols),
            "clustering_efficiency": cluster_scores.get(optimal_clusters, 0)
        }
        
        logger.info(f"✓ Hierarchical clustering completed: {optimal_clusters} optimal clusters")
        
        return {
            "analysis_type": "hierarchical_correlation_clustering",
            "optimal_clusters": optimal_clusters,
            "cluster_assignments": final_clusters.tolist(),
            "cluster_analysis": cluster_analysis,
            "dendrogram_data": dendrogram_data,
            "quality_metrics": quality_metrics,
            "symbols": symbols,
            "data_source": "FMP_Simulation" if use_real_data else "Synthetic"
        }
        
    except Exception as e:
        logger.error(f"Hierarchical clustering calculation failed: {e}")
        return {
            "analysis_type": "hierarchical_correlation_clustering",
            "error": str(e),
            "fallback_used": True
        }

def calculate_correlation_network_metrics(
    symbols: List[str],
    period: str = "1year",
    correlation_threshold: float = 0.5,
    use_real_data: bool = True
) -> Dict:
    """
    Calculate network analysis metrics from correlation matrix
    
    Returns centrality measures, network density, and systemic importance scores
    """
    try:
        from tools.correlation_tools import calculate_correlation_matrix
        
        correlation_matrix = calculate_correlation_matrix(symbols, period, use_real_data)
        
        # Create adjacency matrix based on correlation threshold
        adj_matrix = (np.abs(correlation_matrix.values) > correlation_threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)  # Remove self-connections
        
        # Calculate network metrics
        n_nodes = len(symbols)
        total_possible_edges = n_nodes * (n_nodes - 1) / 2
        actual_edges = np.sum(adj_matrix) / 2  # Undirected graph
        
        network_density = actual_edges / total_possible_edges
        
        # Calculate degree centrality (number of connections)
        degree_centrality = {}
        for i, symbol in enumerate(symbols):
            degree = np.sum(adj_matrix[i])
            degree_centrality[symbol] = {
                "degree": int(degree),
                "normalized_degree": degree / (n_nodes - 1),
                "centrality_rank": 0  # Will be filled after sorting
            }
        
        # Rank by centrality
        sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1]["degree"], reverse=True)
        for rank, (symbol, metrics) in enumerate(sorted_centrality):
            degree_centrality[symbol]["centrality_rank"] = rank + 1
        
        # Calculate clustering coefficient (how interconnected neighbors are)
        clustering_coefficients = {}
        for i, symbol in enumerate(symbols):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) > 1:
                neighbor_connections = 0
                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        if adj_matrix[neighbors[j], neighbors[k]] == 1:
                            neighbor_connections += 1
                
                possible_neighbor_connections = len(neighbors) * (len(neighbors) - 1) / 2
                clustering_coefficients[symbol] = neighbor_connections / possible_neighbor_connections
            else:
                clustering_coefficients[symbol] = 0
        
        # Identify systemically important assets
        systemic_importance = {}
        for i, symbol in enumerate(symbols):
            # Combine high centrality with high average correlation to connected assets
            connected_indices = np.where(adj_matrix[i] == 1)[0]
            if len(connected_indices) > 0:
                avg_correlation_strength = np.mean([
                    abs(correlation_matrix.values[i, j]) for j in connected_indices
                ])
                importance_score = (
                    degree_centrality[symbol]["normalized_degree"] * 0.6 +
                    avg_correlation_strength * 0.4
                )
            else:
                importance_score = 0
            
            systemic_importance[symbol] = {
                "importance_score": importance_score,
                "systemic_rank": 0,  # Will be filled after sorting
                "risk_contribution": importance_score * degree_centrality[symbol]["normalized_degree"]
            }
        
        # Rank by systemic importance
        sorted_importance = sorted(systemic_importance.items(), key=lambda x: x[1]["importance_score"], reverse=True)
        for rank, (symbol, metrics) in enumerate(sorted_importance):
            systemic_importance[symbol]["systemic_rank"] = rank + 1
        
        # Calculate overall network health metrics
        avg_clustering = np.mean(list(clustering_coefficients.values()))
        network_efficiency = 1 - (actual_edges / total_possible_edges)  # Lower density can mean more efficient
        
        network_health = {
            "network_density": network_density,
            "average_clustering": avg_clustering,
            "network_efficiency": network_efficiency,
            "total_connections": int(actual_edges),
            "connectivity_score": min(1.0, network_density * 2),  # 0-1 scale
            "diversification_potential": max(0, 1 - network_density)  # Higher when less connected
        }
        
        logger.info(f"✓ Network analysis completed: {network_density:.3f} density, {actual_edges} connections")
        
        return {
            "analysis_type": "correlation_network_analysis",
            "correlation_threshold": correlation_threshold,
            "network_health": network_health,
            "degree_centrality": degree_centrality,
            "clustering_coefficients": clustering_coefficients,
            "systemic_importance": systemic_importance,
            "adjacency_matrix": adj_matrix.tolist(),
            "symbols": symbols,
            "data_source": "FMP_Simulation" if use_real_data else "Synthetic"
        }
        
    except Exception as e:
        logger.error(f"Network analysis calculation failed: {e}")
        return {
            "analysis_type": "correlation_network_analysis",
            "error": str(e),
            "fallback_used": True
        }

# Export enhanced functions
__all__ = [
    'calculate_rolling_correlations',
    'calculate_regime_conditional_correlations', 
    'calculate_hierarchical_correlation_clustering',
    'calculate_correlation_network_metrics'
]