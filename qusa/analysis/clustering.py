# QUSA/qusa/analysis/clustering.py

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class ClusterAnalyzer: 
    """ 
    Performs clustering analysis on financial time series data.
    """

    def __init__(self, n_clusters=4, algorithm='kmeans', random_state=42): 
        """
        Class constructor.
        
        Parameters: 
            1) n_clusters (int): Number of clusters for clustering algorithms that require it.
            2) algorithm (str): Clustering algorithm to use ('kmeans' or 'dbscan').
            3) random_state (int): Random state for reproducibility.
        """

        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.random_state = random_state

        # model objects initialized after fitting model 
        self.scaler = None 
        self.pca = None
        self.model = None

        # feature information 
        self.feature_columns = None
        self.cluster_profiles = None

    
    def prepare_features(self, df, feature_cols): 
        """ 
        Prepare features for clustering.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.
            2) feature_cols (list): List of column names to use as features.

        Returns; 
            1) tuple: (scaled_features, feature_names, valid_indices)
        """

        # filter to available columns 
        available_cols = [col for col in feature_cols if col in df.columns]

        # handle case of no available indicator columns
        if len(available_cols) == 0: 
            raise ValueError("No valid feature columns found in DataFrame.")
        
        # store feature names as attribute
        self.feature_columns = available_cols

        # remove rows with NaNs in feature columns
        df_features = df[available_cols].dropna().reset_index()

        # check if enough data points remain after NaN removal
        if len(df_features) < self.n_clusters: 
            raise ValueError(
                f"Not enough data points NaN remove (({len(df_features)})"
                f"for {self.n_clusters} clusters."
            )
        
        # standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_features)

        return X_scaled, available_cols, df_features['index'].values
    

    def find_optimal_clusters(self, df, feature_cols, max_k=10): 
        """ 
        Find the optimal number of clusters using 
        elbow method and silhouette score.

        Parameters:
            1) X (np.ndarray): Scaled feature array.
            3) feature_cols (list): List of feature column names.
            3) max_k (int): Maximum number of clusters to test.


        Returns:
            1) best_k (int): Optimal number of clusters.
        """

        # call method to prepare features
        X, _, _ = self.prepare_features(df, feature_cols)

        results = {
            'k': [],
            'inertia': [],
            'silhouette_score': []
        }

        # iterate across available cluster counts (k)
        for k in range(2, max_k + 1): 
            # perform k-means clustering with current k 
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)

            # store metrics from current k 
            results['k'].append(k)
            results['inertia'].append(kmeans.inertia_)
            results['silhouette_score'].append(score)

        # apply elbow method 
        inertias = np.array(results['inertia'])
        delta = np.diff(inertias)
        delta_ratio = delta[:-1] / delta[1:]
        elbow_k = np.argmax(delta_ratio) + 2  # +2 to adjust for k=2 start 

        results['optimal_k_elbow'] = elbow_k
        
        return results 
    

    def fit_clusters(self, df, feature_cols):
        """ 
        Fit clustering model to the data and add labels to clusters.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.
            2) feature_cols (list): List of column names to use as features.
        
        Returns:
            1) df_mod (pd.DataFrame): DataFrame with an additional 'Cluster_Label' column.
        """

        # prepare features for clustering
        X_scaled, feature_names, valid_indices = self.prepare_features(df, feature_cols)

        # fit clustering model based on selected algorithm
        if self.algorithm == 'kmeans': 
            model = KMeans(
                n_clusters=self.n_clusters, 
                random_state=self.random_state
            )
        elif self.algorithm == 'dbscan': 
            model = DBSCAN(eps=0.5, min_samples=5)
        else: 
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

        # fit model and predict cluster labels
        cluster_labels = model.fit_predict(X_scaled)

        # store fitted objects and labels as attributes
        self.scaler = StandardScaler().fit(X_scaled)
        self.model = model
        self.cluster_labels = cluster_labels

        # create modified DataFrame with cluster labels
        df_mod = df.copy()
        df_mod['Cluster_Label'] = -1  # default label for NaN rows
        df_mod.loc[valid_indices, 'Cluster_Label'] = cluster_labels

        return df_mod
    

    def _calculate_cluster_profiles(self, df): 
        """ 
        Calculate mean feature values for each cluster.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Cluster_Label' column.
        """ 

        # filter rows with valid cluster labels 
        df_valid = df[df['Cluster_Label'] != -1]

        # calculate mean feature values per cluster
        cluster_profiles = df_valid.groupby('Cluster_Label')[self.feature_columns].mean()

        profiles = []
        for cluster_label in sorted(df_valid['Cluster_Label'].unique()): 
            # filter data to current cluster
            cluster_data = df_valid[df_valid['Cluster_Label'] == cluster_label]

            # store current cluster element count and proportion 
            profile = {
                'Cluster_Label': cluster_label, 
                'count': len(cluster_data), 
                'proportion': len(cluster_data) / len(df_valid)
            }

            # store mean element count per feature for current cluster 
            for feature in self.feature_columns: 
                if feature in cluster_profiles.columns:
                    profile[f'mean_{feature}'] = cluster_profiles.loc[cluster_label, feature]

            profiles.append(profile)

        # store cluster profiles as attribute
        self.cluster_profiles = pd.DataFrame(profiles)

        return 
    

    def apply_pca(self, df, feature_cols, n_components=2): 
        """ 
        Apply PCA to reduce feature dimensions.

        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data.
            2) feature_cols (list): List of feature column names.
            3) n_components (int): Number of PCA components to keep.
        
        Returns:
            1) df_mod (pd.DataFrame): DataFrame with PCA components added.
        """

        # prepare features for PCA
        X_scaled, _, _ = self.prepare_features(df, self.feature_columns)

        # fit PCA model
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)
        
        return X_pca, self.pca 
    

    def get_cluster_summary(self): 
        """ 
        Get the cluster profiles summary.

        Returns:
            1) cluster_profiles (pd.DataFrame): DataFrame summarizing cluster profiles.
        """

        # handle case where cluster profiles have not been calculated
        if self.cluster_profiles is None: 
            raise ValueError("Cluster profiles have not been calculated yet.")
        
        return self.cluster_profiles
    

    def interpret_clusters(self, df): 
        """
        Interpret clusters by characteristics. 
        
        Parameters:
            1) df (pd.DataFrame): DataFrame containing stock data with 'Cluster_Label' column.
        
        Returns:
            1) interpretations (dict): Dictionary interpreting each cluster.
        """

        # handle case where cluster profiles have not been calculated
        if self.cluster_profiles is None: 
            self._calculate_cluster_profiles(df)

        interpretations = {}


        for _, row in self.cluster_profiles.iterrows():
            cluster_label = int(row['Cluster_Label'])

            # extract key characteristics for current cluster
            overnight = row.get('mean_overnight_delta_pct', 0)
            volume = row.get('volume_ratio_mean', 0)
            rsi = row.get('rsi_mean', 0)

            # interpret with logic rules using feature thresholds
            if (volume > 1.0) and (abs(overnight) > 2.0):
                interpretation = "High Volume Spike with Significant Overnight Change"
            elif (abs(overnight) < 0.5) and (volume < 1.2): 
                interpretation = "Low Volatility, Stable Trading Day"
            elif (overnight > 1.0) and (rsi > 70): 
                interpretation = "Momentum Up"
            elif (overnight < -1.0) and (rsi < 30): 
                interpretation = "Momentum Down"
            elif (rsi > 70): 
                interpretation = "Overbought Conditions"
            elif (rsi < 30): 
                interpretation = "Oversold Conditions"
            else: 
                interpretation = "Standard Trading Day"

            interpretations[cluster_label] = interpretation 
        
