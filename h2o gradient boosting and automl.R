##########################################
# Trying Gradient Boosting and AutoML
##########################################
rm(list = ls(all.names = TRUE)) 

library(h2o)
library(readr)
library(dplyr)
library(tidyverse)
library(caret)
library(glmnet)
library(e1071)
h2o.init()

h2o.no_progress()
h2o.removeAll()
options(max.print = 10000000)

# Read in datasets
train <- h2o.importFile("C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\machine_learning_datasets\\train_split.csv")
valid <- h2o.importFile("C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\machine_learning_datasets\\valid_split.csv")
test <- h2o.importFile("C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\machine_learning_datasets\\MLProjecttest.csv")

#Prep both datasets --> both targets are encoded as 
h2o.describe(train)
h2o.describe(valid)

# convert target 1 & 2 to factors within train and valid
train[, 150] <- as.factor(train[, 150])
train[, 151] <- as.factor(train[, 151])

valid[, 150] <- as.factor(valid[, 150])
valid[, 151] <- as.factor(valid[, 151])

# Starting with target 1

response1 <- "target1"
predictors1 <- setdiff(names(train[, !names(train) %in% c("target2")]), response1)

# Starting Gradient Boosting Target 1

grade1 <- h2o.gbm(x= predictors1,
                  y= response1,
                  training_frame=train,
                  validation_frame = valid,
                  nfolds = 3)

grade1

# # GBM Metrics on training data
# Model Summary: 
#   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
# 1              50                       50               21899         5
# max_depth mean_depth min_leaves max_leaves mean_leaves
# 1         5    5.00000         18         32    30.18000
# 
# 
# H2OBinomialMetrics: gbm
# ** Reported on training data. **
#   
#   MSE:  0.09246694
# RMSE:  0.3040838
# LogLoss:  0.3016738
# Mean Per-Class Error:  0.2353965
# AUC:  0.8726284
# pr_auc:  0.5976343
# Gini:  0.7452567
# R^2:  0.2979869

gbmpred1 <- h2o.predict(object = grade1, newdata = test)
gbmpred1

h2o.metric(gbmpred1)
gbmperform1 <- h2o.performance(model = grade1, newdata = test)


my_cm <- function(model_selected) {
  pred <- h2o.predict(object = model_selected, newdata = test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, test, positive = "1") %>% 
    return()
}

lapply(list(grade1), my_cm)

# Then XGBoost 
boosted1 <- h2o.xgboost(x=predictors1,
                        y=response1,
                        training_frame = train,
                        nfolds = 3)

h2o.xgboost.available()
##############################################
# Target 2
##############################################

# Then target 2!!

response2 <- "target2"
predictors2 <- setdiff(names(train[, !names(train) %in% c("target1")]), response2)

grade2 <- h2o.gbm(x= predictors2,
                  y= response2,
                  training_frame=train,
                  nfolds = 3)

grade2

h2o.performance(grade2, test)

my_cm <- function(model_selected) {
  pred <- h2o.predict(model_selected, test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, valid$target2, positive = "1") %>% 
    return()
}

lapply(list(grade2), my_cm)

######################################################
# AutoML and Ensemble Models
######################################################
h2o.init()

# For this we can use the same response and predictor columns as before
# Starting with target 1, running 10 models 

aml1 <- h2o.automl(y = response1,
                   x = predictors1,
                   training_frame = train,
                   validation_frame = valid,
                   max_models = 10,
                   seed = 1)

aml2 <- h2o.automl(y = response2,
                   x = predictors2,
                   training_frame = train,
                   validation_frame = valid,
                   max_models = 10,
                   seed = 1)

lb <- aml1@leaderboard
lb2 <- aml2@leaderboard
print(lb, n=nrow(lb))
print(lb2, nrow(lb2))

# Separating out the top model: distributed random forests for both targets

# Target 1 Metrics on training data

drf1 <- aml1@leader
drf1

# #   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
# 1              49                       49             6659990        20
# max_depth mean_depth min_leaves max_leaves mean_leaves
# 1        20   20.00000       9640      12016 10822.83700
# Performance on training data
# MSE:  0.03279025
# RMSE:  0.1810808
# LogLoss:  0.126031
# Mean Per-Class Error:  0.07259693
# AUC:  0.982976
# pr_auc:  0.9068082
# Gini:  0.9659519
# R^2:  0.751055


# Target 2 Metrics on training data 

drf2 <- aml2@leader
drf2

# Model Summary: 
#   number_of_trees number_of_internal_trees model_size_in_bytes min_depth
# 1              49                       49             6592418        20
# max_depth mean_depth min_leaves max_leaves mean_leaves
# 1        20   20.00000       9306      12499 10714.87800
# MSE:  0.0368666
# RMSE:  0.1920068
# LogLoss:  0.144439
# Mean Per-Class Error:  0.06213847
# AUC:  0.9838373
# pr_auc:  0.9173892
# Gini:  0.9676745
# R^2:  0.7555568

# Get prediction objects on test dataset
pred1 <- h2o.predict(drf1, test)
pred2 <- h2o.predict(drf2, test)

# Making Model metrics object to see performance on test data

perform1 <- h2o.performance(drf1, newdata=valid)
perform1

# MSE:  0.03170171
# RMSE:  0.1780497
# LogLoss:  0.1221891
# Mean Per-Class Error:  0.07220032
# AUC:  0.9852525
# pr_auc:  0.935896
# Gini:  0.9705051
# R^2:  0.7602379

# Gains/Lift table 
h2o.gainsLift(perform1) 

# Gains/Lift Table: Avg response rate: 15.68 %, avg score: 15.68 %
#   group cumulative_data_fraction lower_threshold     lift
# 7      7               0.15000519        0.424950 4.808620
h2o.auc(perform1) #0.9852525

# Now for target 2 
perform2 <- h2o.performance(drf2, newdata=valid)
perform2
# Gains/Lift table 
h2o.gainsLift(perform2)

# Gains/Lift Table: Avg response rate: 18.60 %, avg score: 18.55 %
# group cumulative_data_fraction lower_threshold     lift
# 7      7               0.15000519        0.565286 4.800132

h2o.auc(perform2) # 0.9862133

#####################################
# Write to csv
#####################################

pred1 <- as.data.frame(pred1)
pred2 <- as.data.frame(pred2)

write.csv(pred1,
          'C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\Modeling Competition\\pred1.csv',
          row.names = TRUE)
write.csv(pred2,
          'C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\Modeling Competition\\pred2.csv',
          row.names = TRUE)

###################################
# Accuracy statistics 

# Running function to get confusion matrix and all the accuracy stats
my_cm <- function(model_selected) {
  pred <- h2o.predict(model_selected, test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, valid$target2, positive = "1") %>% 
    return()
}

lapply(list(drf2), my_cm)

# Gain and Lift Charts + summary stats

# target 1
summary(drf1)
h2o.gainsLift(drf1, test)

# target 2
summary(drf2)
h2o.gainsLift(drf2, test)



# References
# AutoML (https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/R/automl_binary_classification_product_backorders.Rmd#L27)
# DRF (http://h2o-release.s3.amazonaws.com/h2o/rel-xu/3/docs-website/h2o-docs/data-science/drf.html#xrt)
# Medium (https://medium.com/analytics-vidhya/gentle-introduction-to-automl-from-h2o-ai-a42b393b4ba2)


