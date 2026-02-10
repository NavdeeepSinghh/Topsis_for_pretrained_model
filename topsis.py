"""
TOPSIS Implementation for Pre-trained NLP Model Selection
Author: Your Name
Date: February 10, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TOPSIS:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    Implementation for multi-criteria decision making.
    """
    
    def __init__(self, data: pd.DataFrame, weights: List[float], impacts: List[str]):
        """
        Initialize TOPSIS with decision matrix, weights, and impacts.
        
        Args:
            data: DataFrame with models as rows and criteria as columns
            weights: List of weights for each criterion (must sum to 1)
            impacts: List of '+' or '-' for each criterion (+ for beneficial, - for non-beneficial)
        """
        self.data = data.copy()
        self.model_names = data.index.tolist()
        self.criteria_names = data.columns.tolist()
        self.weights = np.array(weights)
        self.impacts = impacts
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate the inputs for TOPSIS."""
        if len(self.weights) != len(self.criteria_names):
            raise ValueError("Number of weights must match number of criteria")
        if len(self.impacts) != len(self.criteria_names):
            raise ValueError("Number of impacts must match number of criteria")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1")
        for impact in self.impacts:
            if impact not in ['+', '-']:
                raise ValueError("Impacts must be '+' or '-'")
    
    def normalize(self) -> pd.DataFrame:
        """
        Normalize the decision matrix using vector normalization.
        
        Returns:
            Normalized decision matrix
        """
        matrix = self.data.values
        # Calculate sum of squares for each column
        sum_of_squares = np.sqrt(np.sum(matrix**2, axis=0))
        # Normalize
        normalized = matrix / sum_of_squares
        return pd.DataFrame(normalized, index=self.model_names, columns=self.criteria_names)
    
    def apply_weights(self, normalized_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Apply weights to normalized matrix.
        
        Args:
            normalized_matrix: Normalized decision matrix
            
        Returns:
            Weighted normalized matrix
        """
        weighted = normalized_matrix.values * self.weights
        return pd.DataFrame(weighted, index=self.model_names, columns=self.criteria_names)
    
    def get_ideal_solutions(self, weighted_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ideal best and ideal worst solutions.
        
        Args:
            weighted_matrix: Weighted normalized matrix
            
        Returns:
            Tuple of (ideal_best, ideal_worst)
        """
        ideal_best = []
        ideal_worst = []
        
        for i, impact in enumerate(self.impacts):
            col = weighted_matrix.iloc[:, i]
            if impact == '+':
                ideal_best.append(col.max())
                ideal_worst.append(col.min())
            else:
                ideal_best.append(col.min())
                ideal_worst.append(col.max())
        
        return np.array(ideal_best), np.array(ideal_worst)
    
    def calculate_distances(self, weighted_matrix: pd.DataFrame, 
                           ideal_best: np.ndarray, 
                           ideal_worst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Euclidean distance from ideal best and ideal worst.
        
        Args:
            weighted_matrix: Weighted normalized matrix
            ideal_best: Ideal best solution
            ideal_worst: Ideal worst solution
            
        Returns:
            Tuple of (distance_to_best, distance_to_worst)
        """
        matrix = weighted_matrix.values
        
        # Distance to ideal best
        distance_to_best = np.sqrt(np.sum((matrix - ideal_best)**2, axis=1))
        
        # Distance to ideal worst
        distance_to_worst = np.sqrt(np.sum((matrix - ideal_worst)**2, axis=1))
        
        return distance_to_best, distance_to_worst
    
    def calculate_topsis_score(self, distance_to_best: np.ndarray, 
                               distance_to_worst: np.ndarray) -> np.ndarray:
        """
        Calculate TOPSIS score (performance score).
        
        Args:
            distance_to_best: Distance to ideal best solution
            distance_to_worst: Distance to ideal worst solution
            
        Returns:
            TOPSIS scores
        """
        return distance_to_worst / (distance_to_best + distance_to_worst)
    
    def run(self) -> pd.DataFrame:
        """
        Run complete TOPSIS analysis.
        
        Returns:
            DataFrame with model rankings
        """
        # Step 1: Normalize
        normalized = self.normalize()
        
        # Step 2: Apply weights
        weighted = self.apply_weights(normalized)
        
        # Step 3: Get ideal solutions
        ideal_best, ideal_worst = self.get_ideal_solutions(weighted)
        
        # Step 4: Calculate distances
        dist_best, dist_worst = self.calculate_distances(weighted, ideal_best, ideal_worst)
        
        # Step 5: Calculate TOPSIS score
        scores = self.calculate_topsis_score(dist_best, dist_worst)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Model': self.model_names,
            'TOPSIS_Score': scores,
            'Distance_to_Best': dist_best,
            'Distance_to_Worst': dist_worst
        })
        
        # Sort by TOPSIS score (descending)
        results = results.sort_values('TOPSIS_Score', ascending=False)
        results['Rank'] = range(1, len(results) + 1)
        
        # Store for later use
        self.results = results
        self.normalized = normalized
        self.weighted = weighted
        self.ideal_best = ideal_best
        self.ideal_worst = ideal_worst
        
        return results
    
    def print_summary(self):
        """Print a summary of the TOPSIS analysis."""
        print("\n" + "="*80)
        print("TOPSIS ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nNumber of Models: {len(self.model_names)}")
        print(f"Number of Criteria: {len(self.criteria_names)}")
        print(f"\nCriteria: {', '.join(self.criteria_names)}")
        print(f"Weights: {', '.join([f'{w:.3f}' for w in self.weights])}")
        print(f"Impacts: {', '.join(self.impacts)}")
        print("\n" + "-"*80)
        print("RANKINGS")
        print("-"*80)
        print(self.results.to_string(index=False))
        print("\n" + "="*80)
        print(f"Best Model: {self.results.iloc[0]['Model']} (Score: {self.results.iloc[0]['TOPSIS_Score']:.4f})")
        print("="*80 + "\n")


def create_sample_data(task_type: str) -> Tuple[pd.DataFrame, List[float], List[str]]:
    """
    Create sample data for different NLP tasks.
    
    Args:
        task_type: Type of NLP task
        
    Returns:
        Tuple of (data, weights, impacts)
    """
    
    # Define models
    models = ['BERT', 'GPT-2', 'T5', 'RoBERTa', 'ALBERT', 'DistilBERT']
    
    if task_type == 'summarization':
        # Criteria: ROUGE-1, ROUGE-2, ROUGE-L, Inference_Time(ms), Model_Size(MB), BLEU
        data = pd.DataFrame({
            'ROUGE-1': [0.42, 0.38, 0.48, 0.41, 0.39, 0.36],
            'ROUGE-2': [0.19, 0.16, 0.24, 0.18, 0.17, 0.15],
            'ROUGE-L': [0.38, 0.34, 0.44, 0.37, 0.35, 0.32],
            'Inference_Time': [45, 52, 68, 48, 42, 28],
            'Model_Size': [440, 550, 780, 470, 180, 260],
            'BLEU': [0.28, 0.24, 0.32, 0.27, 0.25, 0.22]
        }, index=models)
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        impacts = ['+', '+', '+', '-', '-', '+']
        
    elif task_type == 'generation':
        # Criteria: Perplexity, BLEU, Diversity, Inference_Time(ms), Model_Size(MB), Coherence
        data = pd.DataFrame({
            'Perplexity': [12.5, 8.2, 10.3, 13.1, 14.2, 15.8],
            'BLEU': [0.32, 0.42, 0.38, 0.31, 0.29, 0.28],
            'Diversity': [0.68, 0.78, 0.72, 0.66, 0.64, 0.62],
            'Inference_Time': [48, 55, 72, 51, 45, 32],
            'Model_Size': [440, 550, 780, 470, 180, 260],
            'Coherence': [0.72, 0.82, 0.76, 0.71, 0.69, 0.66]
        }, index=models)
        weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.20]
        impacts = ['-', '+', '+', '-', '-', '+']
        
    elif task_type == 'classification':
        # Criteria: Accuracy, F1-Score, Precision, Recall, Inference_Time(ms), Model_Size(MB)
        data = pd.DataFrame({
            'Accuracy': [0.89, 0.85, 0.91, 0.90, 0.88, 0.86],
            'F1-Score': [0.87, 0.83, 0.90, 0.88, 0.86, 0.84],
            'Precision': [0.88, 0.84, 0.91, 0.89, 0.87, 0.85],
            'Recall': [0.86, 0.82, 0.89, 0.87, 0.85, 0.83],
            'Inference_Time': [35, 40, 55, 38, 32, 22],
            'Model_Size': [440, 550, 780, 470, 180, 260]
        }, index=models)
        weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
        impacts = ['+', '+', '+', '+', '-', '-']
        
    elif task_type == 'similarity':
        # Criteria: Cosine_Similarity, Pearson_Correlation, Spearman_Correlation, Inference_Time(ms), Model_Size(MB), Accuracy
        data = pd.DataFrame({
            'Cosine_Similarity': [0.82, 0.78, 0.86, 0.84, 0.81, 0.77],
            'Pearson_Correlation': [0.79, 0.75, 0.84, 0.81, 0.78, 0.74],
            'Spearman_Correlation': [0.77, 0.73, 0.82, 0.79, 0.76, 0.72],
            'Inference_Time': [38, 42, 58, 41, 35, 25],
            'Model_Size': [440, 550, 780, 470, 180, 260],
            'Accuracy': [0.85, 0.81, 0.88, 0.86, 0.83, 0.80]
        }, index=models)
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        impacts = ['+', '+', '+', '-', '-', '+']
        
    elif task_type == 'conversational':
        # Criteria: Response_Quality, Context_Understanding, Fluency, Inference_Time(ms), Model_Size(MB), Engagement
        data = pd.DataFrame({
            'Response_Quality': [0.78, 0.85, 0.82, 0.77, 0.75, 0.72],
            'Context_Understanding': [0.82, 0.88, 0.86, 0.81, 0.79, 0.76],
            'Fluency': [0.85, 0.92, 0.88, 0.84, 0.82, 0.80],
            'Inference_Time': [50, 58, 75, 53, 48, 35],
            'Model_Size': [440, 550, 780, 470, 180, 260],
            'Engagement': [0.76, 0.84, 0.80, 0.75, 0.73, 0.70]
        }, index=models)
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        impacts = ['+', '+', '+', '-', '-', '+']
    
    return data, weights, impacts


if __name__ == "__main__":
    # Example usage
    print("TOPSIS for Pre-trained NLP Model Selection")
    print("="*80)
    
    task = 'classification'
    data, weights, impacts = create_sample_data(task)
    
    topsis = TOPSIS(data, weights, impacts)
    results = topsis.run()
    topsis.print_summary()
