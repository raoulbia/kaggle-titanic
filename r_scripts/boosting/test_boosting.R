library(rstudioapi)
library(xgboost)
library(caret)
trellis.par.set(caretTheme())
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
# dat <- dat[sample(nrow(dat), 50), ]
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
# SKIP THIS AS WE DONT DO MODEL SELECTION HERE
# test_idx <- sort( sample(val_idx, length(val_idx)/2) )
# val_idx <- setdiff(val_idx, test_idx)
# length(train_idx)+length(test_idx)+length(val_idx) # sanity check

# Setup grid search
ctrl <- trainControl(method="repeatedcv", 
                     number=10, 
                     repeats=5,
                     selectionFunction = "oneSE"
                   )

# boosting
fit <- train(Survived~., data=dat[train_idx,], 
             method="adaboost",
             # niter=5000,
             trControl=ctrl
             )
fit; #summary(fit)

# plotting
plot(fit) 

# predict on validation data
predval <- predict(fit, newdata = dat[val_idx,])
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

T_te$Survived <- predict(fit,
                         newdata = T_te[, -1] # rm PassengerId
                         )

submission <- T_te %>% dplyr::select(PassengerId, Survived)
currentDate <- Sys.Date()
csvFileName <- paste("../../local-data/output/submission_",currentDate,".csv",sep="")
write.csv(submission, csvFileName, row.names = FALSE)

# add preds to test data file
# csvFileName <- paste("../../local-data/output/testdata_predictions_",currentDate,".csv",sep="")
# write.csv(T_te, csvFileName, row.names = FALSE)