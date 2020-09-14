library(tidyverse)
library(rio)
library(randomForest)
library(party)
library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rm(list = ls())
options(digits=4, scipen=999)

#############
# load data #
#############

data  <- import('../../local-data/output/titanic_train_clean.csv', setclass = "tibble")

cols <- c("Survived", "Pclass", "Sex", "FamilySize", "hasCabin", "isChild")
dat <- data %>%
  mutate_at(cols, factor)

dat$PassengerId <- NULL # remove original
head(dat)


##############
# Split Data #
##############

# init params and data structures
set.seed(1238)
N <- nrow(dat)

# randomly split original data
train_idx <- sample(1:N, size = 0.75*N)
test_idx <- setdiff(1:N, train_idx)
Xtr <- dat[train_idx, ]
Xte <- dat[test_idx, ] # exclude from k-fold cross validation


#################
# Model fitting #
#################

# init folds
N2 <- nrow(Xtr)
K <- 10
folds <- rep( 1:K, ceiling(N2/K) )
folds <- sample(folds)             # random permute
folds <- folds[1:N2]                # ensure we got N data points

acc <- matrix(NA, K, 1) # for each fold store accuracy pairs
best_feature_tmp <- matrix(NA, K, 3) # store k best features for current replication

# k-fold cross validation
for ( k in 1:K ) {
  
  # split current replication training data into train/val
  train_idx <- which(folds != k)
  val_idx <- setdiff(1:N2, train_idx)

  # fit random forests
  fitrf <- randomForest(Survived ~ .,
                        data = Xtr,
                        subset = train_idx,
                        importance = T)

  
  # validate random forests
  predrf <- predict(fitrf,
                    type = "class",
                    newdata = Xtr[val_idx, ])
  
  
  # compute accuracy and store result random forests
  tabrf <- table(Xtr$Survived[val_idx], predrf)
  acc[k, 1] <- sum(diag(tabrf))/sum(tabrf)

  feat_imp_df <- importance(fitrf) %>%
    data.frame() %>%
    mutate(feature = row.names(.)) %>%
    arrange(desc(MeanDecreaseAccuracy)) %>%
    slice(1:3)
  for (i in seq(1:3))
    best_feature_tmp[k, i] <- feat_imp_df$feature[i]
  
} # END fold k

# view stats
acc # acc stores accuracies for each of the k folds
best_feature_tmp

# compute avergae accuracy of all k-folds
avg_k_folds <- apply(acc, 2, mean) # MARGIN=2, it works over columns
avg_k_folds <- c(`Random Forests` = avg_k_folds[1]) # add names
avg_k_folds

# Predict X_te
predTestRf <- predict(fitrf,
                      type = "class",
                      newdata = Xte)


# view confusion matrix for predictions on labelled test set
tabTestRf <- table(Xte$Survived, predTestRf)
tabTestRf

# compute accuracy from confusion matrix
accBest <- sum(diag(tabTestRf))/sum(tabTestRf)


# view confusion matrix for predictions on labelled train set
# shows the model predictions of the kth (i.e. last) iteration
fitrf


#############################
# Apply to Titanic test set #
#############################

# load data
titanic_test_data  <- import('../../local-data/output/titanic_test_clean.csv', setclass = "tibble")

cols <- c("Pclass", "Sex", "FamilySize", "hasCabin", "isChild")
T_te <- titanic_test_data %>%
  mutate_at(cols, factor)
head(T_te)

T_te$Survived <- predTitanicTestRf <- predict(fitrf,
                                              type = "class",
                                              newdata = T_te[, -1])

submission <- T_te %>% select(PassengerId, Survived)
currentDate <- Sys.Date()
csvFileName <- paste("../../local-data/output/submission_",currentDate,".csv",sep="")
write.csv(submission, csvFileName, row.names = FALSE)
csvFileName <- paste("../../local-data/output/testdata_predictions_",currentDate,".csv",sep="")
write.csv(T_te, csvFileName, row.names = FALSE)



