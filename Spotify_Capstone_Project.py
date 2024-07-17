import numpy as np
import pandas as pd

data = pd.read_csv("/Users/jiaqi/Downloads/spotify52kData.csv").dropna()
#Checking for NaN's 
for column in data.columns:
    # Check if the column contains any NaN values
    has_nan = data[column].isna().any()
    
    # Print the column name and whether it has NaN values
    print("Column:", column)
    print("Has NaN values:", has_nan)
    print()

#Checking for 0's 
columns_to_check = ['duration', 'loudness', 'tempo', 'time_signature', 'valence']

# Loop over each column
for column in columns_to_check:
    # Check if the column contains any zeros
    has_zeros = (data[column] == 0).any()
    
    # Print the column name and whether it has any zeros
    print("Column:", column)
    print("Has zeros:", has_zeros)
    print()
    
has_ones = (data['valence'] == 1).any()
print("Has ones:", has_zeros)

    
netid=19343204 #use this for seeding 

pdata = data[(data['tempo'] != 0) & (data['time_signature'] != 0) & ((data['valence'] != 0) & (data['valence'] != 1))]
#%% Question 1
import matplotlib.pyplot as plt

# Defining the 10 song features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Create a 2x5 grid of subplots for histograms
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# Flatten the axes for easier iteration
axes = axes.flatten()

# Plot histograms for each feature
for i, feature in enumerate(features):
    ax = axes[i]
    ax.hist(pdata[feature], bins=260, color='skyblue', edgecolor='black')
    ax.set_title(feature)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

#%%Question 2
from scipy.stats import spearmanr

plt.figure(figsize=(10, 6))
plt.scatter(pdata['duration'], pdata['popularity'], alpha=0.5, color='skyblue')
plt.title('Song Length vs Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.grid(True)
plt.show()

rho, p_value = spearmanr(pdata['duration'], pdata['popularity'])

print("Spearman correlation coefficient:", rho)
print("p-value:", p_value)
#%% Question 3
from scipy import stats
from scipy.stats import mannwhitneyu

# Filter explicitly rated songs
explicit_songs = pdata[pdata['explicit'] == True]['popularity']

# Filter non-explicitly rated songs
non_explicit_songs = pdata[pdata['explicit'] == False]['popularity']

#Plot 
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(explicit_songs, bins=20, alpha=0.5, color='blue')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Popularity for Explicit Songs')

plt.subplot(1, 2, 2)
plt.hist(non_explicit_songs, bins=20, alpha=0.5, color='green')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.title('Distribution of Popularity for Non-Explicit Songs')

plt.tight_layout()
plt.show()

# Mann-Whitney U test
statistic1, p_value1 = stats.mannwhitneyu(explicit_songs, non_explicit_songs)
print("Mann-Whitney U statistic 1:", statistic1)
print("Mann-Whitney U p-value1:", p_value1)

#Testing for which one is larger 
statistic, p_value = stats.mannwhitneyu(explicit_songs, non_explicit_songs, alternative="greater")
print("Mann-Whitney U statistic:", statistic)
print("Mann-Whitney U p-value:", p_value)

#%% Question 4

major_key = pdata[pdata['mode'] == 1]['popularity']
minor_key = pdata[pdata['mode'] == 0]['popularity']

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(major_key, bins=20, alpha=0.5, color='blue')
plt.xlabel('Popularity')
plt.ylabel('Major Key Songs')
plt.title('Distribution of Popularity for Major Key Songs')

plt.subplot(1, 2, 2)
plt.hist(minor_key, bins=20, alpha=0.5, color='green')
plt.xlabel('Popularity')
plt.ylabel('Minor Key Songs ')
plt.title('Distribution of Popularity for Minor Key Songs')

plt.tight_layout()
plt.show()

# Mann-Whitney U test
statistic1, p_value1 = mannwhitneyu(minor_key, major_key)
statistic, p_value = mannwhitneyu(minor_key, major_key, alternative="greater")
# Testing for siginficance
print("Mann-Whitney U statistic 1:", statistic1)
print("Mann-Whitney U p-value 1:", p_value1)
#Testing for which one is larger 
print("Mann-Whitney U statistic:", statistic)
print("Mann-Whitney U p-value:", p_value)
#%% Question 5
# First plot 
loudness = pdata['loudness']
energy = pdata['energy']

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(loudness, energy, alpha=0.5)
plt.title('Loudness vs Energy')
plt.xlabel('Loudness (dB)')
plt.ylabel('Energy')
plt.grid(True)
plt.show()

#find Spearman's correlation 
correlation_coefficient, p_value = spearmanr(loudness, energy)

# Print results
print("Spearman's correlation coefficient:", correlation_coefficient)
#%% Question 6 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

X = pdata[features]
y = pdata['popularity']

# Initialize dictionaries to store MSE and R-squared values
mse_dict = {}
r2_dict = {}

# Fit linear regression models and evaluate performance for each feature
for feature in features:
    # Reshape feature array to 2D array for single feature
    X_feature = X[[feature]]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=netid)
    
    # Fit linear regression model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = regression_model.predict(X_test)
    
    # Calculate mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    mse_dict[feature] = mse
    
    # Calculate R-squared
    r2 = r2_score(y_test, y_pred)
    r2_dict[feature] = r2

# Print MSE and R-squared values for each feature
print("Mean Squared Error (MSE):")
for feature, mse in mse_dict.items():
    print(f"{feature}: {mse}")

print("\nR-squared:")
for feature, r2 in r2_dict.items():
    print(f"{feature}: {r2}")
    
#%% Question 7 
import statsmodels.api as sm

# Select features and target variable
X = pdata[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = pdata['popularity']

# Add constant for the intercept term
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()

# Print summary of the model
print(model.summary())
#%% Question 8 
from sklearn.decomposition import PCA
from scipy import stats


X = pdata[['duration', 'danceability', 'energy', 'loudness', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# Standardize the features (z-score)
zscoredData = stats.zscore(X)

# Fit PCA
pca = PCA()
pca.fit(zscoredData)

#Loadings
loadings = pca.components_*-1

# Proportion of variance explained by each component
eigVals = pca.explained_variance_

    
kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))

print('Number of factors selected by elbow criterion: 1') 

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_, marker='o', linestyle='-')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (Eigenvalue=1)')
plt.title('Principal Component vs Eigenvalue with Kaiser Criterion')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xticks(range(1, len(pca.explained_variance_) + 1))
plt.legend()
plt.grid(True)
plt.show()

n_components = len(pca.explained_variance_ratio_)

for whichPrincipalComponent in range(0, 3):  # Loop through three principal components index at 0 for 
    plt.figure()
    x = np.linspace(1, n_components, n_components)
    plt.bar(x, loadings[whichPrincipalComponent, :] * -1)
    plt.xlabel('Feature Index')
    plt.ylabel('Loading')
    plt.title(f'Principal Component {whichPrincipalComponent} Loadings')
    for i, val in enumerate(loadings[whichPrincipalComponent, :]):
        print(f'Feature Index: {i+1}, Loading: {val:.3f}')
    plt.show()
    
varExplained = eigVals/sum(eigVals)*100
print("\nCumulative proportion of variance explained by components:")
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))
    
cumulative_variance = varExplained[0] + varExplained[1] + varExplained[2]
print("Cumulative variance explained by the first three principal components:", cumulative_variance)


#%% Question 9 
from sklearn.linear_model import LogisticRegression
from scipy.special import expit # this is the logistic sigmoid function
from sklearn.metrics import roc_auc_score

X = pdata[['valence']]
y = pdata['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=netid)


# Fit model:
model = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
    

#Format the data
x1 = np.linspace(X_train.min(), X_train.max(), 500)
y1 = x1 * model.coef_ + model.intercept_
sigmoid = expit(y1)

#Scatter to get an idea of what the graph looks like 
plt.scatter(X_train,y_train,color='black', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.xlabel('Valence')
plt.ylabel('Mode?')
plt.yticks(np.array([0,1]))
plt.show()

# Plot:
plt.plot(x1,sigmoid.ravel(),color='red',linewidth=3, label='Logistic Sigmoid') # the ravel function returns a flattened array
plt.scatter(X_train,y_train,color='black', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Testing data')
plt.xlabel('Valence')
plt.ylabel('Mode')
plt.yticks(np.array([0,1]))
plt.show()

# Making predictions on the testing data
y_pred = model.predict(X_test)

# Calculate accuracy or other relevant metrics on the testing data
accuracy = model.score(X_test, y_test)
print("Accuracy on testing data:", accuracy)

# Calculate AUC
y_probs = model.predict_proba(X_test)[:, 1]  # Probability of 'mode' being 1
auc = roc_auc_score(y_test, y_probs)
print("AUC:", auc)

#Trying other features 

# Define the list of features
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Initialize lists to store evaluation results
aucs = []

# Iterate over each feature
for feature in features:
    # Select the current feature along with 'mode' as target variable
    X = pdata[[feature]]
    y = pdata['mode']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=netid)
    
    # Fit logistic regression model
    model = LogisticRegression().fit(X_train, y_train)
    
    # Evaluate model performance
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    # Append results to the lists
    aucs.append(auc)
    
    # Print results for each feature
    print("Feature:", feature)
    print("AUC:", auc)
    print()

# Sort features based on AUC values in descending order
sorted_features = sorted(zip(features, aucs), key=lambda x: x[1], reverse=True)

# Select the best performing feature (the one with the highest AUC)
best_feature = sorted_features[0][0]

print("The best performing feature based on AUC is:", best_feature)

#%%Question 10 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

yOutcomes = np.where(pdata['track_genre'] == "classical", 1, 0)

# Select predictors
predictors = pdata[['danceability', 'liveness', 'loudness']].to_numpy()

# Train Random Forest classifier
numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(predictors, yOutcomes)

# Use model to make predictions
predictions = clf.predict(predictors)

# Assess model accuracy
modelAccuracy = accuracy_score(yOutcomes, predictions)
print('Random forest model accuracy:', modelAccuracy)

# Compute ROC curve and AUC for predictors
probs = clf.predict_proba(predictors)[:, 1]
fpr, tpr, thresholds = roc_curve(yOutcomes, probs)
auc_value = roc_auc_score(yOutcomes, probs)
print("AUC Value for Components:", auc_value)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Principal Components')
plt.legend(loc="lower right")
plt.show()

# Testing duration
predictor_duration = pdata[['duration']].to_numpy()

# Train Random Forest classifier
clf_duration = RandomForestClassifier(n_estimators=numTrees).fit(predictor_duration, yOutcomes)

# Use model to make predictions
predictions_duration = clf_duration.predict(predictor_duration)

# Assess model accuracy
modelAccuracy_duration = accuracy_score(yOutcomes, predictions_duration)
print('Random forest model accuracy using duration as predictor:', modelAccuracy_duration)

# Compute ROC curve and AUC for duration
probs_duration = clf_duration.predict_proba(predictor_duration)[:, 1]
fpr_duration, tpr_duration, thresholds_duration = roc_curve(yOutcomes, probs_duration)
auc_value_duration = roc_auc_score(yOutcomes, probs_duration)
print("AUC Value for duration", auc_value_duration)

# Plot ROC curve for duration
plt.figure()
plt.plot(fpr_duration, tpr_duration, color='blue', label=f'ROC curve (area = {auc_value_duration:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Duration')
plt.legend(loc="lower right")
plt.show()
#%%Extra Credit
import pandas as pd
from scipy.stats import mannwhitneyu

explicit_songs = pdata[pdata['explicit'] == True]['loudness']
non_explicit_songs = pdata[pdata['explicit'] == False]['loudness']

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histogram for explicit songs
axs[0].hist(explicit_songs, bins=20, color='blue', alpha=0.5)
axs[0].set_title('Explicit Songs')
axs[0].set_xlabel('Loudness (dB)')
axs[0].set_ylabel('Frequency')

# Histogram for non-explicit songs
axs[1].hist(non_explicit_songs, bins=20, color='orange', alpha=0.5)
axs[1].set_title('Non-Explicit Songs')
axs[1].set_xlabel('Loudness (dB)')
axs[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Perform Mann-Whitney U test
u_statistic, p_value = mannwhitneyu(explicit_songs, non_explicit_songs)

# Print the results
print("Mann-Whitney U statistic:", u_statistic)
print("P-value:", p_value)

u_statistic, p_value = mannwhitneyu(explicit_songs, non_explicit_songs, alternative='greater')

# Print the result of the statistical test
print("Mann-Whitney U statistic:", u_statistic)
print("P-value:", p_value)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. Explicit songs tend to have louder loudness compared to non-explicit songs.")
else:
    print("Fail to reject the null hypothesis. There is no evidence that explicit songs tend to have louder loudness compared to non-explicit songs.")

