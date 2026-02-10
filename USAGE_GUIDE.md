# TOPSIS NLP Model Selection - Usage Instructions

## Quick Start

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/topsis-nlp-model-selection.git
   cd topsis-nlp-model-selection
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

**Option 1: Run complete analysis for all tasks**
```bash
python main.py
```

This will:
- Analyze all 5 NLP tasks
- Generate visualizations for each task
- Create CSV files with detailed results
- Generate a comprehensive README report

**Option 2: Run analysis for a specific task**
```python
from topsis import TOPSIS, create_sample_data

# Choose task: 'summarization', 'generation', 'classification', 'similarity', 'conversational'
task = 'classification'

# Get data for the task
data, weights, impacts = create_sample_data(task)

# Run TOPSIS analysis
topsis = TOPSIS(data, weights, impacts)
results = topsis.run()

# Print results
topsis.print_summary()
```

---

## Understanding the Code

### 1. TOPSIS Class (`topsis.py`)

The main TOPSIS implementation with methods:

```python
class TOPSIS:
    def __init__(self, data, weights, impacts):
        # Initialize with decision matrix, weights, and impacts
        
    def normalize(self):
        # Normalize the decision matrix
        
    def apply_weights(self, normalized_matrix):
        # Apply weights to criteria
        
    def get_ideal_solutions(self, weighted_matrix):
        # Calculate ideal best and worst solutions
        
    def calculate_distances(self, weighted_matrix, ideal_best, ideal_worst):
        # Calculate Euclidean distances
        
    def calculate_topsis_score(self, distance_to_best, distance_to_worst):
        # Calculate final TOPSIS scores
        
    def run(self):
        # Execute complete TOPSIS analysis
```

### 2. Data Structure

Each task uses a decision matrix with 6 models and various criteria:

```python
data = pd.DataFrame({
    'Criterion1': [values],
    'Criterion2': [values],
    ...
}, index=['BERT', 'GPT-2', 'T5', 'RoBERTa', 'ALBERT', 'DistilBERT'])
```

**Weights**: Importance of each criterion (must sum to 1)
```python
weights = [0.25, 0.20, 0.15, ...]  # Sum = 1.0
```

**Impacts**: Whether higher is better (+) or lower is better (-)
```python
impacts = ['+', '+', '-', ...]  # '+' for beneficial, '-' for non-beneficial
```

---

## Customizing the Analysis

### Adding Your Own Data

1. **Create your decision matrix**:
```python
import pandas as pd

data = pd.DataFrame({
    'Accuracy': [0.89, 0.85, 0.91],
    'Speed': [45, 52, 38],
    'Size': [440, 550, 780]
}, index=['Model_A', 'Model_B', 'Model_C'])
```

2. **Define weights and impacts**:
```python
weights = [0.5, 0.3, 0.2]  # Must sum to 1
impacts = ['+', '-', '-']   # + for beneficial, - for non-beneficial
```

3. **Run TOPSIS**:
```python
from topsis import TOPSIS

topsis = TOPSIS(data, weights, impacts)
results = topsis.run()
print(results)
```

### Modifying Evaluation Criteria

Edit the `create_sample_data()` function in `topsis.py`:

```python
def create_sample_data(task_type):
    if task_type == 'my_custom_task':
        data = pd.DataFrame({
            'Criterion1': [...],
            'Criterion2': [...],
        }, index=['Model1', 'Model2', ...])
        
        weights = [0.6, 0.4]  # Adjust weights
        impacts = ['+', '-']   # Adjust impacts
        
        return data, weights, impacts
```

---

## Output Files

### CSV Files (in `results/` folder)

Each CSV contains:
- Model names
- TOPSIS scores
- Distance to ideal best
- Distance to ideal worst
- Rankings

### Visualizations (in `visualizations/` folder)

For each task:
1. **Main Analysis Chart** (`{task}_analysis.png`):
   - TOPSIS scores bar chart
   - Rankings visualization
   - Distance comparison
   - Decision matrix heatmap
   - Weighted matrix heatmap
   - Criteria weights pie chart

2. **Radar Chart** (`{task}_radar.png`):
   - Top 3 models comparison across all criteria

3. **Overall Comparison** (`overall_comparison.png`):
   - Best model for each task

---

## Understanding TOPSIS Scores

**TOPSIS Score Range**: 0 to 1
- **Higher is better**
- Score closer to 1 = Closer to ideal solution
- Score closer to 0 = Farther from ideal solution

**Interpretation**:
- Score > 0.7: Excellent choice
- Score 0.5 - 0.7: Good choice
- Score 0.3 - 0.5: Moderate choice
- Score < 0.3: Poor choice

---

## Task-Specific Details

### 1. Text Summarization
**Criteria**: ROUGE-1, ROUGE-2, ROUGE-L, Inference Time, Model Size, BLEU
**Best for**: Document summarization, article condensation

### 2. Text Generation
**Criteria**: Perplexity, BLEU, Diversity, Inference Time, Model Size, Coherence
**Best for**: Creative writing, content generation

### 3. Text Classification
**Criteria**: Accuracy, F1-Score, Precision, Recall, Inference Time, Model Size
**Best for**: Sentiment analysis, topic classification

### 4. Text Sentence Similarity
**Criteria**: Cosine Similarity, Pearson Correlation, Spearman Correlation, Inference Time, Model Size, Accuracy
**Best for**: Semantic search, duplicate detection

### 5. Text Conversational
**Criteria**: Response Quality, Context Understanding, Fluency, Inference Time, Model Size, Engagement
**Best for**: Chatbots, dialogue systems

---

## Troubleshooting

### Common Issues

**1. Module not found error**
```bash
pip install -r requirements.txt
```

**2. Weights don't sum to 1**
Ensure weights sum exactly to 1.0:
```python
weights = [0.25, 0.25, 0.25, 0.25]  # Sum = 1.0
```

**3. Invalid impact values**
Use only '+' or '-':
```python
impacts = ['+', '+', '-']  # Correct
impacts = ['positive', 'negative']  # Wrong
```

**4. Visualization not saving**
Ensure `visualizations/` folder exists:
```bash
mkdir visualizations
```

---

## Advanced Usage

### Sensitivity Analysis

Test how results change with different weights:

```python
weight_scenarios = [
    [0.4, 0.3, 0.2, 0.1],  # Scenario 1
    [0.25, 0.25, 0.25, 0.25],  # Scenario 2
    [0.1, 0.2, 0.3, 0.4],  # Scenario 3
]

for i, weights in enumerate(weight_scenarios):
    topsis = TOPSIS(data, weights, impacts)
    results = topsis.run()
    print(f"\nScenario {i+1}:")
    print(results)
```

### Batch Processing

Analyze multiple datasets:

```python
datasets = {
    'dataset1': (data1, weights1, impacts1),
    'dataset2': (data2, weights2, impacts2),
}

all_results = {}
for name, (data, weights, impacts) in datasets.items():
    topsis = TOPSIS(data, weights, impacts)
    all_results[name] = topsis.run()
```

---

## References

1. **TOPSIS Method**:
   - Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications

2. **NLP Metrics**:
   - ROUGE: Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries
   - BLEU: Papineni, K., et al. (2002). BLEU: A Method for Automatic Evaluation

3. **Pre-trained Models**:
   - BERT: Devlin et al. (2019)
   - GPT-2: Radford et al. (2019)
   - T5: Raffel et al. (2020)
   - RoBERTa: Liu et al. (2019)

---

## License

This project is available for educational purposes.

---

## Contact

For questions or issues, please create an issue on GitHub or contact the repository owner.

---

**Happy Analyzing! 📊**
