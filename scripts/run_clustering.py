# qusa/scripts/run_clustering.py

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.analysis.clustering import ClusterAnalyzer
from qusa.utils.config import load_config


def confirm_directory(path):
    """
    Confirm that a directory exists, creating it if necessary.

    Parameters:
        1) path (str): The directory path to confirm.
    """

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    return


def plot_elbow_curve(optimal_results, paths): 
    """  
    Plots the elbow curve for clustering analysis.

    Parameters:
        1) optimal_results (dict): Dictionary containing the 
            optimal number of clusters and associated metrics.
        2) paths (dict): Dictionary containing paths for saving figures.
    """

    # inertia plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # inertia plot
    ax1.plot(
        optimal_results['k'], 
        optimal_results['inertia'], 
        'bo-'
    )
    ax1.axvline(  
        x=optimal_results['optimal_k'], 
        color='firebrick', 
        linestyle='--',
        label=f"Optimal Clusters: {optimal_results['optimal_k']}"
    )
    ax1.set_xlabel(
        'Number of Clusters (k)', 
        fontsize=12
    )
    ax1.set_ylabel(
        'Inertia', 
        fontsize=12
    )
    ax1.set_title(
        'Elbow Method for Optimal k', 
        fontsize=14,
        fontweight='bold'
    )
    ax1.legend()

    # silhouette score plot
    ax2.plot(
        optimal_results['k'], 
        optimal_results['silhouette_score'], 
        'go-'
    )
    ax2.axvline(
        x=optimal_results['optimal_k'], 
        color='firebrick', 
        linestyle='--',
        label=f"Optimal Clusters: {optimal_results['optimal_k']}"
    )
    ax2.set_xlabel(
        'Number of Clusters (k)', 
        fontsize=12
    )
    ax2.set_ylabel(
        'Silhouette Score', 
        fontsize=12
    )
    ax2.set_title(
        'Silhouette Scores for Different k', 
        fontsize=14,
        fontweight='bold'
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()

    fig_path = os.path.join(paths["figures_dir"], "elbow_curve.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    return 


def plot_pca_clusters(data, pca_X, pca_model, paths): 
    """
    Plot clusters in PC space. 

    Parameters:
        1) data (pd.DataFrame): The original dataset with cluster labels.
        2) pca_X (np.ndarray): The PCA-transformed data.
        3) pca_model (PCA): The fitted PCA model.
        4) paths (dict): Dictionary containing paths for saving figures.
    """

    # copy data for plotting
    data_filtered = data.loc[data['cluster'] != -1].copy()
    
    # plotting code 
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        pca_X[:, 0], 
        pca_X[:, 1], 
        c=data_filtered['cluster'].values, 
        cmap='tab10', 
        alpha=0.7
    )
    cbar = plt.colorbar(scatter, ax=ax)  # add colorbar 
    cbar.set_label('Cluster Label', fontsize=12)

    var1 = pca_model.explained_variance_ratio_[0] * 100  # percentage variance for PC1
    var2 = pca_model.explained_variance_ratio_[1] * 100  # percentage variance for PC2

    ax.set_xlabel(
        f'Principal Component 1 ({var1:.2f}% Variance)', 
        fontsize=12
    )
    ax.set_ylabel(
        f'Principal Component 2 ({var2:.2f}% Variance)', 
        fontsize=12
    )
    ax.set_title(
        'PCA Clustering Visualization', 
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()
    
    fig_path = os.path.join(paths["figures_dir"], "pca_clusters.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


    return 


def plot_cluster_profiles(analyzer, paths): 
    """
    Plot heatmaps of cluster profiles.

    Parameters:
        1) analyzer (ClusterAnalyzer): ClusterAnalyzer instance 
            with clustering results.
        2) paths (dict): Dictionary containing paths for saving figures.
    """

    profiles = analyzer.cluster_profiles

    # select feature columns
    feature_cols = [
        col for col in profiles.columns 
        if col.endswith('_mean') and col not in ['count', 'percent']
    ]

    # prepare data for heatmap
    heatmap_data = profiles[feature_cols].T
    heatmap_data.columns = [
        f"Cluster {int(col)}" for col in profiles['cluster']
    ]

    # clean feature names 
    heatmap_data.index = [
        col.replace('_mean', '').replace('_', ' ').title() 
        for col in heatmap_data.index
    ]

    # plotting code 
    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.matshow(heatmap_data, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xlabel('Clusters', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title(
        'Cluster Profiles Heatmap', 
        fontsize=14, 
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    fig_path = os.path.join(paths["figures_dir"], "cluster_profiles_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")

    return


def plot_cluster_time_series(data, paths): 
    """
    Plot time series of cluster distributions.

    Parameters:
        1) data (pd.DataFrame): The original dataset with cluster labels and timestamps.
        2) paths (dict): Dictionary containing paths for saving figures.
    """
    
    # copy data for plotting
    data_filtered = data.loc[data['cluster'] != -1].copy()

    # convert timestamp to datetime if not already
    data_filtered['date'] = pd.to_datetime(data_filtered['date'])

    # plotting code
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ## plot 1: overnight delta vs cluster
    scatter = ax1.scatter(
        data_filtered['date'], 
        data_filtered['overnight_delta'], 
        c=data_filtered['cluster'], 
        cmap='tab10', 
        alpha=0.7
    )
    cbar1 = plt.colorbar(scatter, ax=ax1)  # add colorbar for clusters 
    cbar1.set_label('Cluster Label', fontsize=12)

    ax1.axhline(
        y=0, 
        color='gray', 
        linestyle='--', 
        linewidth=1
    )

    ax1.set_xlabel('date', fontsize=12)
    ax1.set_ylabel('Overnight Delta', fontsize=12)
    ax1.set_title(
        'Overnight Delta vs Time by Cluster', 
        fontsize=14,
        fontweight='bold'
    )

    ## plot 2: cluster distribution over time
    data_filtered['month'] = data_filtered['date'].dt.to_period('M')  # label months 
    cluster_counts = data_filtered.groupby(  # group by month and cluster
        ['month', 'cluster']
    ).size().unstack(fill_value=0)
    cluster_props = cluster_counts.div(  # normalize to get proportions
        cluster_counts.sum(axis=1), 
        axis=0
    )

    cluster_props.plot(
        kind='area', 
        stacked=True, 
        ax=ax2, 
        colormap='tab10', 
        alpha=0.7
    )
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Days', fontsize=12)
    ax2.set_title(
        'Cluster Distribution Over Time', 
        fontsize=14,
        fontweight='bold'
    )
    ax2.legend(
        title='Cluster', 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left'
    )

    plt.tight_layout()
    plt.show()

    fig_path = os.path.join(paths["figures_dir"], "cluster_time_series.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


    return


def analyze_clusters(data, analyzer): 
    """
    Analyze characteristics of each cluster. 

    Parameters:
        1) data (pd.DataFrame): The original dataset with cluster labels.
        2) analyzer (ClusterAnalyzer): ClusterAnalyzer instance 
            with clustering results.
    """

    interpretations = analyzer.interpret_clusters(data)

    print("\n" + "="*80)
    print("DETAILED CLUSTER ANALYSIS")
    print("="*80)


    # iterate across clusters 
    for cluster_label, interpretation in interpretations.items():
        # filter data for the current cluster
        current_cluster_data = data.loc[data['cluster'] == cluster_label]

        # skip empty clusters
        if len(current_cluster_data) == 0:
            continue  

        print(f"\n{'='*80}")
        print(f"CLUSTER {cluster_label}: \"{interpretation}\"")
        print(f"{'='*80}")

        print(f"\nSize: {len(current_cluster_data)} days ({len(current_cluster_data)/len(data)*100:.1f}%)")
        
        print("\nKey Statistics:")
        print(f"  Overnight Delta:")
        print(f"    Mean: {current_cluster_data['overnight_delta_pct'].mean():.2f}%")
        print(f"    Median: {current_cluster_data['overnight_delta_pct'].median():.2f}%")
        print(f"    Std Dev: {current_cluster_data['overnight_delta_pct'].std():.2f}%")
        
        print(f"\n  Volume:")
        print(f"    Mean Ratio: {current_cluster_data['volume_ratio'].mean():.2f}x")
        print(f"    Spikes (>2x): {(current_cluster_data['volume_spike']==True).sum()} days")
        
        print(f"\n  RSI:")
        print(f"    Mean: {current_cluster_data['rsi'].mean():.1f}")
        print(f"    Oversold (<30): {(current_cluster_data['rsi']<30).sum()} days")
        print(f"    Overbought (>70): {(current_cluster_data['rsi']>70).sum()} days")


        if 'abnormal' in current_cluster_data.columns:
            abnormal_rate = current_cluster_data['abnormal'].mean() * 100
            print(f"\n  Abnormal Overnight Moves: {abnormal_rate:.1f}%")

        if 'day_of_week' in current_cluster_data.columns:
            # count distribution of days of the week
            dow_counts = current_cluster_data['day_of_week'].value_counts(normalize=True) * 100

            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
            print(f"\n  Day of Week Distribution:")

            for day_num, count in dow_counts.items():
                if day_num < 5:
                    print(f"    {days[day_num]}: {count} days ({count/len(current_cluster_data)*100:.1f}%)")


def main(): 
    """
    Main function to run clustering analysis and visualizations.
    """

    config = load_config("~/Projects/qusa/config.yaml")

    data_cfg = config["data"]
    paths = data_cfg["paths"]


    print("="*80)
    print("CLUSTERING ANALYSIS FOR OVERNIGHT TRADING")
    print("="*80)
    
    # Load processed data
    print("\n1. Loading processed data...")
    try: 
        processed_dir = paths["processed_data_dir"]
        ticker = data_cfg["tickers"][0]

        data_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        data = pd.read_csv(data_path)
        print("   Data loaded successfully.")
    except FileNotFoundError: 
        print("   ERROR: Processed data file not found.")
        print("   Run 'python scripts/run_FE_pipeline.py' first")
        return 1
    
    # confirm directory for figures
    confirm_directory(os.path.join(paths["figures_dir"], "dummy.txt"))


    # confirm datetime conversion
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    # find optimal clusters and fit model
    print("\n2. Performing clustering analysis...")
    analyzer = ClusterAnalyzer(n_clusters=4, algorithm='kmeans')
    optimal_results = analyzer.find_optimal_clusters(
        data, 
        feature_cols=None, 
        max_k=8
    )

    print(f"   ✓ Optimal k: {optimal_results['optimal_k']}")
    print(f"   → Using k={analyzer.n_clusters} clusters")
    
    # plot elbow curve
    print("\n3. Generating elbow curve...")
    plot_elbow_curve(optimal_results, paths)

    # fit clustering model
    print("\n4. Fitting clustering model...")
    data_clustered = analyzer.fit_clusters(data, feature_cols=None)
    print(f"   ✓ Clustered {len(data_clustered[data_clustered['cluster']>=0])} days")

    # perform PCA for visualization
    print("\n5. Performing PCA for visualization...")
    pca_X, pca_model = analyzer.perform_pca(
        data_clustered, 
        feature_cols=analyzer.feature_columns
    )
    print(f"   ✓ Explained variance: {pca_model.explained_variance_ratio_.sum():.1%}")

    # visualize 
    print("\n6. Generating visualizations...")
    plot_pca_clusters(data_clustered, pca_X, pca_model, paths)
    plot_cluster_profiles(analyzer, paths)
    plot_cluster_time_series(data_clustered, paths)

    # print summary 
    print("\n7. Cluster Summary:")
    analyzer.print_summary(data_clustered)

    # detailed analysis
    print("\n8. Detailed Cluster Analysis:")
    analyze_clusters(data_clustered, analyzer)

    # save clustered data
    print("\n9. Saving clustered data...")
    clustered_path = os.path.join(
        paths["processed_data_dir"],
        "AMZN_processed_clustered.csv"
    )
    data_clustered.to_csv(clustered_path, index=False)

    print("   ✓ Clustered data saved.")

    # Final summary
    print("\n" + "="*80)
    print("✓ CLUSTERING ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - data/processed/AAPL_clustered.csv")
    print("  - data/processed/cluster_profiles.csv")
    print("  - data/processed/elbow_curve.png")
    print("  - data/processed/pca_clusters.png")
    print("  - data/processed/cluster_profiles.png")
    print("  - data/processed/cluster_time_series.png")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
