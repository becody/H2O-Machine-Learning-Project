####################################
# Logistic, Lasso, Ridge in H2O
####################################

# clear objects in evironment
rm(list = ls(all.names = TRUE)) 

# Load packages
library(tidyverse)
library(caret)
library(glmnet)
library(readr)
library(dplyr)
library(e1071)
library(h2o)

localH2O = h2o.init()

#remove H2O code
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Read in datasets
train <- read_csv("C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\machine_learning_datasets\\train_split.csv")
valid <- read_csv("C:\\Users\\17046\\OneDrive\\Documents\\MSA 20\\Machine Learning\\machine_learning_datasets\\valid_split.csv")

#Prep both datasets
train <- na.omit(train)
train <- as.data.frame(train)

valid <- na.omit(valid)
valid <- as.data.frame(valid)

train$target1 <- as.factor(train$target1)
valid$target1 <- as.factor(valid$target1)

# convert to h2o frames and then identify inputs and outputs for target 1
train <- as.h2o(train)
test <- as.h2o(valid)

response1 <- "target1"
predictors1 <- setdiff(names(train[, !names(train) %in% c("target2")]), response1)

###############################################
# Pure, Lasso, and Ridge Logistic for Target 1
###############################################

# Train logistic model
pure_log1 <- h2o.glm(family="binomial",
                     x = predictors1,
                     y = response1,
                     lambda = 0,
                     training_frame = train)

# shows coefficients table
show_coeffs <- function(model_selected) {
  model_selected@model$coefficients_table %>% 
    as.data.frame() %>% 
    mutate_if(is.numeric, function(x) {round(x, 3)}) %>% 
    filter(coefficients != 0) %>% 
    knitr::kable()
}

# Use this function: 
show_coeffs(pure_log1)

# LASSO logistic model

lasso1 <- h2o.glm(family = "binomial",
                  alpha = 1,
                  seed = 1,
                  x = predictors1,
                  y = response1,
                  training_frame = train)

show_coeffs(lasso1)

# Ridge Logistic Model: 
ridge1 <- h2o.glm(family = "binomial",
                  alpha = 0,
                  seed = 1,
                  x = predictors1,
                  y = response1,
                  training_frame = train)
show_coeffs(ridge1)

# model performance on test data 

my_cm <- function(model_selected) {
  pred <- h2o.predict(model_selected, test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, valid$target1, positive = "1") %>% 
    return()
}

pred1 <- h2o.predict(pure_log1, test)
lapply(list(pure_log1, lasso1, ridge1), my_cm)

########################################################
# Trying this on target 2
########################################################

train$target2 <- as.factor(train$target2)
valid$target2 <- as.factor(valid$target2)

response2 <- "target2"
predictors2 <- setdiff(names(train[, !names(train) %in% c("target1")]), response1)


# Train logistic model
pure_log2 <- h2o.glm(family="binomial",
                     x = predictors2,
                     y = response2,
                     lambda = 0,
                     training_frame = train)

# shows coefficients table
show_coeffs <- function(model_selected) {
  model_selected@model$coefficients_table %>% 
    as.data.frame() %>% 
    mutate_if(is.numeric, function(x) {round(x, 3)}) %>% 
    filter(coefficients != 0) %>% 
    knitr::kable()
}

# Use this function: 
show_coeffs(pure_log2)

# LASSO logistic model

lasso2 <- h2o.glm(family = "binomial",
                  alpha = 1,
                  seed = 1,
                  x = predictors2,
                  y = response2,
                  training_frame = train)

show_coeffs(lasso2)

# Ridge Logistic Model: 
ridge2 <- h2o.glm(family = "binomial",
                  alpha = 0,
                  seed = 1,
                  x = predictors2,
                  y = response2,
                  training_frame = train)

show_coeffs(ridge2)

# model performance on test data 

my_cm <- function(model_selected) {
  pred <- h2o.predict(model_selected, test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, valid$target2, positive = "0") %>% 
    return()
}

lapply(list(pure_log2, lasso2, ridge2), my_cm)









