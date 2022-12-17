# fraud_detection
# Finding Fraud Faster 
## < Nicole An > 
# Executive Summary 
## Overview
For large financial institutions, losses to fraud can reach to more than 10% of their total spending. Such massive losses push companies to utilize machine learning models for new solutions to prevent, detect, and eliminate fraud in the payment stream. Predictive models are promising weapons to fight against fraud transaction and set thresholds on prediction scores that align with companies’ operational goals. 
## Business Problem
The objective is to operate at 5% false positive rate. That is, out of 100 legit transactions, the fraud detection system wrongly categorizes 5 transactions as suspicious, shutting down the payment or closing the account completely. The error happens when a non-fraudulent transaction triggers the system to interrupt the transaction, causing both the financial institution and merchants to lose revenue. It also damages customer rapport and reputation to e-commerce businesses for losing legit sales. The trade-off here is that by addressing false positives, the true positive rate also decreases. In other words, the system’s ability to correctly detect frauds from fraudulent transactions declines. Missing a fraud erodes trusts and generates legal issues and huge loss in profit for the institution. We want to reduce the costs made by errors from both sides by setting an appropriate threshold for the prediction score to detect fraud. 
The goal is to create machine learning models to be better at separating legit transactions from fraudulent ones. The best performing model ensures fraud scoring is dynamic with features that are important influencers of the fraud rate. 
Explanatory Analysis on Variables
## Explore the Target
 
Event_Label	n
fraud	6785
legit	118215

The target variable is event label, which has two levels indicating whether the event is a legit or a fraud transaction. Fraud is the positive class. The bar graph above shows the default fraud rate to be 5.4% while the legit rate to be 94.6%. That is, out of all 125,000 events from the dataset, 6785 transactions were labelled as fraud. The default accuracy rate for the target variable is 94.6%, the rate of the majority class, the legit class. The two levels are relatively unbalanced in terms of sizes, so accuracy of the model will tend to be very high for correctly predicting the legit class. Therefore, it alone is not a good evaluation of the model. When selecting the best predictive model, not only should the model accuracy rate be higher than the default accuracy, but the model should also have relatively high precision and recall as indication of generating correct predictions for the positive class. 
## Explore the Predictors
      
The charts above indicate that these predictor variables seem to obtain strong relationship to fraud. For categorical variables cvv type, transaction type, and transaction environment, there is significant variation in fraud rate between different levels. For numerical variables account ages, transaction amount, and transaction adjustment amount, the mean and 50% quartile vary between fraud legit events. Therefore, these influential variables may determine whether the transaction is a fraud.

The histograms and the table above discuss the important variables that the company believe may influence fraud rate.
•	Billing postal: Since billing postal has 11065 unique levels, the graph above shows the number of levels against fraud rate. The graph is right skewed with majority levels having fraud rates under the default fraud rate, 5.4%. However, there are outliers that have 100% fraud rate, that is areas with only 1 transaction and the transaction is a fraud. With the high variability in fraud rate between different levels, billing postal can be an important predictor.
•	Email domain: Since email domain has 6992 unique levels, the graph above shows the number of levels against fraud rate. The graph is right skewed with majority levels having fraud rates under the default fraud rate, 5.4%. Email domain also has outliers with 100% fraud rate. It is worth investigating those levels of email domains so email domain can be an important predictor.
•	User agent browser: User agent browser is parsed from user agent, indicating the type of browser used when the transaction took place. There are 3 levels, Mozilla, Opera, and missing value (nan). Since all levels have fraud rates close to the default fraud rate 5.4%, user agent browser may be irrelevant factor to fraud. Therefore, it may not be an important predictor. 
## Methodology 
1.	Data partitioning
•	The holdout dataset is splitted randomly by 80/20 where 80% were training data and 20% were testing data.
2.	Data preprocessing
•	Formula for Random Forest Model
i.	Event_label ~ account_age_days, transaction_amt, transaction_adj_amt, card_bin_freq, billing_state, currency, cvv, transaction_type, transaction_env
•	Formula for Reduced Logistic Model
i.	Event_label ~ account_age_days, transaction_amt, transaction_adj_amt, email_domain_fraud_rate, card_bin_freq, billing_state, currency, cvv, transaction_type, transaction_env
•	Target Variable Pre-Processing
i.	Removed rows to make the occurrence of levels in the target variables so that the distribution has a ratio of 1:3 (fraud:legit).
•	Numeric Predictor Pre-Processing 
i.	Replaced missing numeric variables with median 
ii.	Removed variables with low variances (almost all of the same number)
•	Categorical Predictor Pre-Processing 
i.	Replaced missing categorical variables with a new level called “unknown”
ii.	Assigned a novel factor level to potential new levels in testing dataset
iii.	Encoded categorical variables with dummy variables of 0s and 1s with one-hot encoded
iv.	Pooled rarely occurring levels together and created a level called “other”
v.	Target encoded email domain and billing postal with mean fraud rates
vi.	Target encoded card bin with frequency
3.	Model specification  
•	Trained 2 Random Forest Models
i.	Variables are selected based on explanatory analysis of all predictor variables and business requirements from stakeholders
ii.	Hyperparameters are tuned with CV model
•	Trained 1 Reduced Logistic Model 
## Key Insights
•	The chart below shows that transaction amount varies in average between legit and fraud class. Fraud transactions have lower transaction adjustment amount compared to legit transactions. 
 
•	Both email domain and billing postal seem to be important predictors from explanatory analysis with mean encoding. However, they were not good predictors with random forest models because they yield a testing AUC of 0.91, which is significantly lower than the best performing random forest model. The influence may be overfitting the model, and the high number of levels with 0 fraud rate decreases the influence of these two variables.
•	Card bin has 6332 unique levels, and the chart below shows that the different levels yield varied fraud rate. The frequency of card bin is an important variable of the best performing random forest model, indicating that the number of transactions made from a certain card type/issuing bank is influential to fraud rate.
 
## Recommendations 
•	Since the model’s precision rate is only moderate (0.471), the dataset can be improved with more relatable variables such as location, order, e-commerce site so that the model can make more accurate predictions. 
•	 The financial institution could lower the costs of inspection and send emails to confirm with the customers of suspicious activities instead of shutting down the account to lower customer friction when making false positive mistakes. This way, the financial institution could operate in a higher false positive rate and drive true positive rate higher, reducing the probability of missing an actual fraud. With a false positive rate of 0.09, the model can correctly identify more than 892 from 1000 fraud transactions. That is 50 more transactions from operating at 0.05 false positive rate.
•	The financial institution can further investigate on the card type and issuing bank of fraud transactions because this information seems to be important influencers of fraud rate according to the best performing random forest model. Each 1 increase in the number of transactions made from a certain card type/issuing bank, the fraud rate increases by 0.02.

## Model Analysis  
<tables comparing your model performance > 
Model 	Partition	AUC	Precision	Recall
Model 1 – Reduced Logistic Model	train	0.96165	0.89919	0.70975
Model 2 – Random Forest (tree=130)	train	0.99924	0.98903	0.79373
Model 3 – Random Forest (tree=100;mtry=17)	train	0.99949	0.98811	0.82307
				
Model 	Partition	AUC	Precision	Recall
Model 1 – Reduced Logistic Model	test	0.93995	0.84433	0.61533
Model 2 – Random Forest (tree=130)	test	0.94100	0.91150	0.61908
Model 3 – Random Forest (tree=100;mtry=17) 	test	0.94442	0.88458	0.65064
				

### ROC Chart by Model 
   
### Feature Importance by Model 
   
### Model Selection
Based on the overall performance, the best performing model is the random forest model with tree=100;mtry=17. This model has the highest testing AUC out of the 3 models, indicating that it performs best when making predictions. Model 3 is selected because a 0.65064 testing recall rate is significantly higher than the recall rates of the 2 models. A model with the highest recall rate means that it will make the best predictions for the positive class. Although it does not have the best testing precision rate, it is not very far from the highest precision rate, down by only 0.02. Also, we value recall rate more because missing a fraud can cause legal issues. Making a false positive mistake cause less in terms of revenue and reputation assumably. 
Selected Model Operating Ranges 

fpr
<dbl>	threshold
<dbl>	tpr
<dbl>		
0.00	0.490	0.000		
0.01	0.291	0.658		
0.02	0.198	0.770		
0.03	0.153	0.820		
0.04	0.119	0.835		
0.05	0.100	0.860		
0.06	0.084	0.867		
0.07	0.072	0.878		
0.08	0.062	0.889		
0.09	0.056	0.892		

## Operational Business Rules w. Expected Performance (Precision & Recall) 
   
The operational rule is to detect fraudulent transactions at a 5% false positive rate (FPR) or lower. The best performing model allows operating at 5% FPR to leverage the tradeoff between FPR and recall (true positive rate) and finds the sweet spot that minimizes damage of making errors from both sides. The ROC chart for the best performing model above shows that false positive rate and recall are positively related to each other. The costing of missing a fraud transaction is huge, so the recall should be as high as possible. Rising recall will cause the FPR to increase, and the costs of inspection increase as well. The random forest model sets the prediction score threshold to be 0.109. In other words, if a transaction has a 10.9% or higher probability of being fraudulent, the model marks the transaction to be suspicious and askes the fraud detection system to terminate the payment stream. There is a 5% chance that this suspicious transaction is a legitimate transaction. 
By operating at 5% FPR, the chart threshold vs TPR (recall) indicates that a threshold of 10.9% yields a 0.854 recall. That is, out of 1000 fraud transactions, the model will correctly classify 854 of them. The precision rate is 0.471, displayed in the chart precision vs recall. That is, out of 1000 transactions predicted as fraudulent, 471 of them are correct classification. The chart also indicates that precision and recall are negatively related to each other. Therefore, the model yields a moderate precision rate when the recall rate is very high. 
