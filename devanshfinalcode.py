import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc, mean_squared_error
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings


Spotify_data = pd.read_csv("/Users/devansh/Downloads/New_Spspotify_chart_song_ranks.csv") #The data we got from Spotify
kaggle_data = pd.read_csv("/Users/devansh/Downloads/Drake_Spotify_Data.csv") #The data we got from Kaggle

#Upon looking at the data, we saw that there were a lot of duplicates. So, we remove
#all the duplicate tracks based on 'uri' and 'artist_names', then count the number of artists per song.

artist_per_song_count = Spotify_data.drop_duplicates(subset=['uri', 'artist_names']) \
    .groupby('uri') \
    .size() \
    .reset_index(name='count') \
    .sort_values(by='count', ascending=False)

#We now need to filter the Spotify data for tracks by Drake, and again remove duplicates based on 'uri',
drake_chart_spotify = Spotify_data[Spotify_data['artist_names'] == "Drake"] \
    .drop_duplicates(subset='uri') \
    .sort_values(by='WeekDate')

#As the trends are always changing we wanted to make sure that we are focusing
#on the later trends only
#So, we filter the Kaggle data for tracks released in or after 2020 and remove duplicates based on 'track_name'.

kaggle_data_2020 = kaggle_data[kaggle_data['album_release_year'] >= 2020] \
    .drop_duplicates(subset='track_name')

#We need to join the filtered Kaggle dataset and Drake's Spotify chart data
#on their respective track URIs. This merges data for tracks found in both datasets.
inner_join_result = pd.merge(kaggle_data_2020, drake_chart_spotify, left_on='track_uri', right_on='uri', how='inner')

left_join_data = pd.merge(kaggle_data_2020, drake_chart_spotify, left_on='track_uri', right_on='uri', how='left')


#Now we need to select a subset of columns for analysis, focusing on musical attributes, track names etc.
#This prepares the data for further analysis on the correlation between these attributes and chart performance.
new_data = left_join_data[['tempo', 'valence', 'liveness', 'instrumentalness', 'acousticness', 
                      'speechiness', 'loudness', 'energy', 'danceability', 'WeekDate', 
                      'track_name_x']]



with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
    new_data['on_chart'] = np.where(new_data['WeekDate'].isna(), 0, 1)

new_data = new_data.sort_values(by='WeekDate')


new_data.drop(columns='WeekDate', inplace=True)
new_data.rename(columns={'track_name_x': 'track_name'}, inplace=True)


new_data.columns

np.random.seed(45)
#For an initial EDA we are checking the correlation between each col of the data
correlation_matrix_full = new_data.corr(numeric_only=True)
print(correlation_matrix_full)

#We are now focusing on a few features we want to check how they affect the success rate
features_to_compare = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness', 'speechiness', 'instrumentalness','liveness']

X = new_data[features_to_compare]
y = new_data['on_chart'] #Target Binary Variable

#Splitting the data at 80% - 20% ratio split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
test_data_combined = X_test.copy()
test_data_combined['on_chart'] = y_test
mean_on_chart = test_data_combined['on_chart'].mean()
print("Mean of 'on_chart' in the test dataset:", mean_on_chart)


#We now scale the data using the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Performing Logistic Regression
logistic = LogisticRegression()
logistic.fit(X_train_scaled, y_train)
predict_test_logistic = logistic.predict(X_test_scaled)
prob_logistic = logistic.predict_proba(X_test_scaled)[:,1]
accuracy = accuracy_score(y_test, predict_test_logistic)
print(f"Logistic Regression Test Accuracy: {accuracy}")
print(classification_report(y_test, predict_test_logistic))
test_error_logistic = mean_squared_error(y_test, predict_test_logistic)
print("This is the MSE for Logistic" , test_error_logistic)


#Performing ROC for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_logistic)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.figure()
lw = 2
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix for Logistic Regression
confusion = confusion_matrix(y_test, predict_test_logistic)
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
predict_test_rf = rf.predict(X_test_scaled)
prob_rf = rf.predict_proba(X_test_scaled)[:,1]
accuracy = accuracy_score(y_test, predict_test_rf)
print(f"Random Forest Test Accuracy: {accuracy}")
print(classification_report(y_test, predict_test_rf))
test_error_rf = mean_squared_error(y_test, predict_test_rf)
print("This is the MSE for Random Forest" , test_error_rf)

#ROC for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Random Forest')
plt.legend(loc="lower right")
plt.show()

#Confusion Matrix for Random Forest
confusion = confusion_matrix(y_test, predict_test_rf)
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - Random Forest')
plt.show()

#Perform Lasso regression to see whic hcef results in 0
lasso = LassoCV(cv=5).fit(X_train_scaled, y_train)
lasso_coefficients = pd.Series(lasso.coef_, index=features_to_compare)
lasso_predict = lasso.predict(X_test_scaled)
test_mse_lasso = mean_squared_error(y_test, lasso_predict)
print("Test MSE for Lasso:", test_mse_lasso)
print("Lasso Coefficients:")
print(lasso_coefficients)

#check for a non-linear relationship
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)
pred_test_qda = qda.predict(X_test_scaled)
accuracy = accuracy_score(y_test, pred_test_qda)
print(f"QDA Test Accuracy: {accuracy}")
print(classification_report(y_test, pred_test_qda))
test_error_qda = mean_squared_error(y_test, pred_test_qda)
print("This is the MSE for QDA" , test_error_qda)



avg_features_by_chart_status = new_data.groupby('on_chart')[features_to_compare].mean()
avg_features_by_chart_status.T.plot(kind='bar', figsize=(12, 6))
plt.title('Average Feature Values: Charted vs Not Charted')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.legend(title='Charted', labels=['Not Charted', 'Charted'])
plt.show()



