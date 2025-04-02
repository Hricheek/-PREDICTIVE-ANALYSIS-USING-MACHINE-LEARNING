**# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING**

# Company:CODTECH IT SOLUTIONS

# Name:Hricheek Bhattacharjee

# Intern ID:CT04WB108

# Domain:Data Analysis

# Mentor:Neela Santosh

# Description:
So, I recently worked on a project analyzing the Titanic dataset using Python, where I performed data preprocessing and built a machine-learning model to predict survival. I used Pandas for handling data, Matplotlib for visualization, and Scikit-learn for machine learning. Here's how I approached the entire process.Loading the Data
First, I loaded the Titanic dataset from a CSV file into a Pandas DataFrame. The dataset contained 891 rows and 12 columns, including details like passenger class, age, fare, and whether they survived. Once the data was loaded, I checked for missing values, which is always an essential step in data analysis.
Handling Missing Data:When I checked for missing values, I found that the dataset had some missing values in the Age, Cabin, and Embarked columns. The Cabin column had too many missing values (687 out of 891), so it wasn’t useful for analysis, and I removed it along with Name and Ticket, which didn’t add much predictive power.For the Age column, instead of dropping missing values, I filled them with the mean age of the passengers, which helped retain valuable data. For the Embarked column, since it only had two missing values, I replaced them with the most common embarkation port in the dataset.
Converting Categorical Variable:Since machine learning models work with numerical data, I had to convert categorical variables like Pclass (Passenger Class), Sex, and Embarked into numerical form. I first converted Pclass into a string type and then used one-hot encoding (get_dummies) to create separate binary columns for each category.For example, the Sex column was converted into Sex_female and Sex_male, and the Embarked column became Embarked_C, Embarked_Q, and Embarked_S. This transformation was essential for the machine-learning model to understand categorical features.
Splitting the Data:After preprocessing, I separated the Survived column as my target variable and kept the rest as features. Then, I split the dataset into training (75%) and testing (25%) sets using train_test_split from Scikit-learn.Building the Machine Learning Model:For this classification problem, I decided to use a Random Forest Classifier, which is an ensemble learning method that builds multiple decision trees and combines their outputs. I trained the model using the training data, which allowed it to learn patterns between different features and the survival status of passengers.
Evaluating Model Performance:Once the model was trained, I made predictions on the test data and evaluated its performance using the ROC-AUC score. The AUC score came out to be 0.826, which meant the model was performing well in distinguishing between survivors and non-survivors.
Hyperparameter Tuning:To further improve the model’s performance, I experimented with different numbers of decision trees (estimators) in the Random Forest. I tested values ranging from 1 to 200 and plotted the AUC score for both training and testing sets. As expected, increasing the number of trees improved accuracy up to a certain point, but after that, the performance leveled off.
Conclusion:This project was a great learning experience because it covered essential machine learning concepts like data preprocessing, feature engineering, model training, and evaluation. The Titanic dataset was relatively small, but the steps I followed—handling missing data, converting categorical variables, and hyperparameter tuning—are applicable to real-world problems where data is much larger and messier.
Overall, I found Random Forest to be a powerful classifier, and tuning hyperparameters played an important role in optimizing model performance. I also realized how crucial data preprocessing is; a well-prepared dataset can significantly impact model accuracy. This project reinforced my understanding of classification models, feature engineering, and model evaluation, and I’m looking forward to applying these techniques to more complex datasets in the future!

# Output:






















