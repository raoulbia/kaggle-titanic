library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rm(list = ls())

library(xgboost)
library(caret)
# trellis.par.set(caretTheme())

library(lattice)  # for plotting
library(rio) # for importing data
library(dplyr) # for piping

# Run all subsequent models in parallel
# library(doParallel)
# n.cores <- detectCores(all.tests = T, logical = T) 
# cl <- makePSOCKcluster(n.cores)
# doParallel::registerDoParallel(cl)


#############
# load data #
#############

data  <- import('../../local-data/output/titanic_train_clean.csv', setclass = "tibble")
cols <- c("Survived", "Pclass", "Sex", "FamilySize", "hasCabin", "isChild")
dat <- data %>%
  mutate_if(is.double, as.integer) 
# mutate_at(cols, as.numeric(factor))

dat$PassengerId <- NULL # remove original
dat$hasCabin <- NULL # high correlation with Pclass

dat$Survived <- as.factor(dat$Survived)
# dat <- dat[sample(nrow(dat), 50), ]
head(dat)
# TODO is scaling applicable here? don't think so

res <- cor(dat[, -1])
round(res, 2)

##############
# Split Data #
##############

# init params and data structures
set.seed(1238)
# dat[,-1] <- scale(dat[,-1])  # standardize 

# split data into training, validation and test
N <- nrow(dat)
train_idx <- sort( sample(1:N, size = N*0.7) )
val_idx <- setdiff(1:N, train_idx)

# randomly split into test and validation
# SKIP THIS AS WE DONT DO MODEL SELECTION HERE
# test_idx <- sort( sample(val_idx, length(val_idx)/2) )
# val_idx <- setdiff(val_idx, test_idx)
# length(train_idx)+length(test_idx)+length(val_idx) # sanity check

# Setup grid search
ctrl <- trainControl(method="repeatedcv", number=10, repeats=5)

# ~1.6 hrs on laptop
system.time({
rf.fit        <- train(Survived~., data=dat[train_idx,], method="rf", trControl=ctrl);
knn.fit       <- train(Survived~., data=dat[train_idx,], method="knn", trControl=ctrl);
svm.fit       <- train(Survived~., data=dat[train_idx,], method="svmRadialWeights", trControl=ctrl);
adabag.fit    <- train(Survived~., data=dat[train_idx,], method="AdaBag", trControl=ctrl); # SLOW
adaboost.fit  <- train(Survived~., data=dat[train_idx,], method="adaboost", trControl=ctrl)
})


# close multi-core cluster
# doParallel::stopCluster(cl) 
# rm(cl)

# summary of model differences
results <- resamples(list(RF=rf.fit, kNN=knn.fit, SVM=svm.fit, 
                          Bag=adabag.fit,
                          Boost=adaboost.fit))
summary(results)


# Plot Accuracy Summaries
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)                 # Box plots of accuracy


# Predict on validation data

predval <- predict(rf.fit, newdata = dat[val_idx,])
tab <- table(dat$Survived[val_idx], predval)
sum(diag(tab))/sum(tab)

predval <- predict(svm.fit, newdata = dat[val_idx,])
tab <- table(dat$Survived[val_idx], predval)
sum(diag(tab))/sum(tab)

predval <- predict(adabag.fit, newdata = dat[val_idx,])
tab <- table(dat$Survived[val_idx], predval)
sum(diag(tab))/sum(tab)


#############################
# Apply to Titanic test set #
#############################

# load data
titanic_test_data  <- import('../../local-data/output/titanic_test_clean.csv', setclass = "tibble")
T_te <- titanic_test_data %>%
  mutate_if(is.double, as.integer) 
head(T_te)

T_te$Survived <- predict(svm.fit,
                         newdata = T_te[, -1] # rm PassengerId
                         )

submission <- T_te %>% dplyr::select(PassengerId, Survived)
currentDate <- Sys.Date()
csvFileName <- paste("../../local-data/output/submission_",currentDate,".csv",sep="")
write.csv(submission, csvFileName, row.names = FALSE)

# add preds to test data file
# csvFileName <- paste("../../local-data/output/testdata_predictions_",currentDate,".csv",sep="")
# write.csv(T_te, csvFileName, row.names = FALSE)