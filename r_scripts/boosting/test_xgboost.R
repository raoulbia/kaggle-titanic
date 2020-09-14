library(MBCbook)
library(adabag)
library(rstudioapi)
library(rpart)
library(fastAdaboost)
library(xgboost)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rm(list = ls())

#############
# load data #
#############

data  <- import('../../local-data/output/titanic_train_clean.csv', setclass = "tibble")
cols <- c("Survived", "Pclass", "Sex", "FamilySize", "hasCabin", "isChild")
dat <- data %>%
  mutate_if(is.double, as.integer) 
  # mutate_at(cols, as.numeric(factor))

dat$PassengerId <- NULL # remove original
dat$Survived <- as.factor(dat$Survived)
head(dat)
# TODO is scaling applicable here? don't think so

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
test_idx <- sort( sample(val_idx, length(val_idx)/2) )
val_idx <- setdiff(val_idx, test_idx)

# sanity check
length(train_idx)+length(test_idx)+length(val_idx)

# xgboost fitting with arbitrary parameters
xgb_params_1 = list(
  # objective = "linear",                                               # binary classification
  eta = 0.01,                                                                  # learning rate
  max.depth = 3,                                                               # max tree depth
  eval_metric = "error"                                                          # evaluation/loss metric
)

# fit the model with the arbitrary parameters specified above
xgb_1 = xgboost(data = as.matrix(dat[train_idx, -1]),
                label = as.matrix(dat[train_idx, 1]),
                params = xgb_params_1,
                nrounds = 100,                                                 # max number of trees to build
                verbose = TRUE,                                         
                # print.every.n = 1,
                # early.stop.round = 10                                          # stop if no improvement within 10 trees
)