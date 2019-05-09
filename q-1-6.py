Explain how decision tree is suitable handle missing values(few attributes missing in test samples) in data.

Decision Tree is suitable in handling missing values:

1. Predictive value imputation (PVI):  
	- Aim at estimating the missing value and impute them within both the training and the test set.
	- The simplest imputation consists of replacing the missing values with the mean for numerical predictors or the mode for categorical predictors.

2. The Separate Class (SC):
	- Replaces the missing value with a new value or a new class for all observations.
	- For categorical predictors we can simply define missing value as a category on its own
	- For continuous predictors any value out of the interval of observed value can be used.

3. Distribution-based imputation (DBI):
	- When selecting the predictor to split upon, only the observations with known values are considered.
	- After choosing the best predictor to split upon, observations with known values are split as usual.
	- Observations with unknown values are distributed among the two child nodes proportionately to the split on observed values.
	- Similarly, for prediction, a new test observation with missing value is split intro branches according to the portions of training example falling into those branches.
	- The prediction is then based upon a weighted vote among possible leaves.

4. Discarding Missing Data:
	- Simple approach
	- Do not use row/data having missing values
	- Used earlier
	- Not considered as good approach

5. Marking Null Method:
	- Another simple approach
	- Simply mark missing values as Null
	- Consider Null as another class
	- Variation of Separate Class approach

Referenced from: https://arxiv.org/pdf/1804.10168.pdf