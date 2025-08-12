# AdaBoost from Scratch

A complete implementation of the AdaBoost (Adaptive Boosting) algorithm built from scratch using Python and NumPy. This project demonstrates the core concepts of ensemble learning and boosting algorithms without relying on external machine learning libraries.

## Overview

AdaBoost is a powerful ensemble learning method that combines multiple weak learners (decision stumps) to create a strong classifier. This implementation includes:

- **Decision Stump**: A simple weak learner that makes predictions based on a single feature threshold
- **AdaBoost Algorithm**: The main boosting algorithm that iteratively trains weak learners and combines them
- **Breast Cancer Classification**: Complete example using the scikit-learn breast cancer dataset

## Features

- ✅ Pure NumPy implementation of AdaBoost
- ✅ Custom Decision Stump weak learner
- ✅ Comprehensive evaluation with accuracy metrics and confusion matrix
- ✅ Visualization of results
- ✅ Well-documented code with clear variable names
- ✅ Jupyter notebook for interactive exploration

## Algorithm Details

### Decision Stump
Each decision stump:
- Uses a single feature and threshold for classification
- Can have positive or negative polarity
- Weighted by an alpha value based on its performance

### AdaBoost Process
1. Initialize uniform weights for all training samples
2. For each iteration:
   - Find the best decision stump (feature + threshold combination)
   - Calculate the stump's alpha weight based on weighted error
   - Update sample weights (increase weights for misclassified samples)
   - Add the weighted stump to the ensemble
3. Final prediction is the weighted majority vote of all stumps

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hghaemi/AdaBoost_from_scratch.git
cd AdaBoost_from_scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook
```bash
jupyter notebook adaboost.ipynb
```

### Using the Classes
```python
from adaboost import AdaBoost, DecisionStump
import numpy as np

# Create and train the model
clf = AdaBoost(n_clf=30)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

## Results

When tested on the breast cancer dataset, the implementation achieves:
- **Accuracy**: ~90-95% (depending on random seed)
- **Robust Performance**: Consistent results across different train/test splits
- **Fast Training**: Efficient implementation suitable for educational purposes

## Project Structure

```
AdaBoost_from_scratch/
├── adaboost.ipynb          # Main implementation and demonstration
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── .gitignore            # Git ignore rules
```

## Dependencies

- `numpy>=1.21.0` - Core numerical computations
- `scikit-learn>=1.0.0` - Dataset loading and evaluation metrics
- `matplotlib>=3.5.0` - Plotting and visualization
- `jupyter>=1.0.0` - Interactive notebook environment

## Educational Value

This implementation is ideal for:
- **Learning ensemble methods**: Understanding how weak learners combine to form strong classifiers
- **Algorithm implementation**: Seeing the mathematical concepts translated to code
- **Machine learning education**: Clear, documented code for teaching purposes
- **Research baseline**: Starting point for AdaBoost variations and improvements

## Key Concepts Demonstrated

- **Weighted sampling**: How AdaBoost focuses on difficult examples
- **Ensemble learning**: Combining multiple simple models
- **Bias-variance tradeoff**: How boosting reduces bias while managing variance
- **Weak learner theory**: Using simple decision stumps effectively

## Mathematical Foundation

The implementation follows the classic AdaBoost algorithm:
- Weight update: `w_i = w_i * exp(-α_t * y_i * h_t(x_i))`
- Alpha calculation: `α_t = 0.5 * ln((1-ε_t)/ε_t)`
- Final prediction: `H(x) = sign(Σ α_t * h_t(x))`

Where:
- `w_i` are sample weights
- `α_t` is the weight of weak learner t
- `ε_t` is the weighted error of weak learner t
- `h_t(x)` is the prediction of weak learner t

## Contributing

Contributions are welcome! Areas for improvement:
- Additional weak learner types
- Performance optimizations
- More comprehensive testing
- Additional datasets and examples

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the original AdaBoost algorithm by Freund and Schapire (1997)
- Uses the breast cancer dataset from scikit-learn
- Inspired by educational materials on ensemble learning

## References

- Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences, 55(1), 119-139.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction. Springer Science & Business Media.