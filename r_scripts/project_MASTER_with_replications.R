library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rm(list = ls())
library(tidyverse)
library(rio)
library(randomForest)
options(digits=4, scipen=999)

# load data
dat  <- import('../local-data/output/titanic_train_clean.csv', setclass = "tibble")

##################
# Pre-processing #
##################

cols <- c("Survived" "Pclass", "Sex", "FamilySize", "Cabin", "Embarked")
dat <- data %>%
  mutate_at(cols, factor)
head(dat)

dat$PassengerId <- NULL # remove original

#################
# Model fitting #
#################

# init params and data structures
set.seed(1238)
N <- nrow(dat)
R <- 100 # nbr replications
out <- matrix(NA, R, 4)
colnames(out) <- c("cvscore_rf", "cvscore_lr", "best", "test")
out <- as.data.frame(out) # to ensure doubles are not stored as characters ...
best_feature <- matrix(NA, R, 3) # store best feature for each replication

# replications
for (r in 1:R){
  print(r)
  
  # randomly split original data
  train_r_idx <- sample(1:N, size = 0.75*N)
  test_r_idx <- setdiff(1:N, train_r_idx)
  Xtr <- dat[train_r_idx, ]
  Xte <- dat[test_r_idx, ] # exclude from k-fold cross validation
  
  # init folds
  N2 <- nrow(Xtr)
  K <- 10
  folds <- rep( 1:K, ceiling(N2/K) )
  folds <- sample(folds)             # random permute
  folds <- folds[1:N2]                # ensure we got N data points

  acc <- matrix(NA, K, 2) # for each fold store accuracy pairs
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
                          importance = T,
                          ntree = 100)

    # fit logreg
    fitlr <- glm(Survived ~ .,
                 data = Xtr,
                 subset = train_idx,
                 family = "binomial")
    
    # Optimal Discrimination Threshold Tau
    # predObj <- prediction(predictions = fitted(fitlr),
    #                       labels = Xtr$Survived[train_idx])
    # sens <- performance(predObj, "sens")
    # spec <- performance(predObj, "spec")
    # cutoffs <- sens@x.values[[1]]
    # sum_sens_spec <- sens@y.values[[1]] + spec@y.values[[1]]
    # best_idx <- which.max(sum_sens_spec)
    # tau = format(cutoffs[best_idx],3)
    tau <- 0.5
    
    # validate random forests
    predrf <- predict(fitrf,
                      type = "class",
                      newdata = Xtr[val_idx, ])


    # validate logreg
    predlr_prob <- predict(fitlr,
                           type = "response",
                           newdata = Xtr[val_idx, ])
    predlr <- ifelse(predlr_prob > tau, 1, 0)

    # compute accuracy and store result random forests
    tabrf <- table(Xtr$Survived[val_idx], predrf)
    acc[k, 1] <- sum(diag(tabrf))/sum(tabrf)

    # compute accuracy and store result logreg
    tablr <- table(Xtr$Survived[val_idx], predlr)
    acc[k, 2] <- sum(diag(tablr))/sum(tablr)

    # feat_imp_df <- importance(fitrf) %>%
    #   data.frame() %>%
    #   mutate(feature = row.names(.)) %>%
    #   arrange(desc(MeanDecreaseAccuracy)) %>%
    #   slice(1:3)
    # for (i in seq(1:3))
    #   best_feature_tmp[k, i] <- feat_imp_df$feature[i]
  
    } # END fold k
  
  # top3 features
  for (i in seq(1:3))
    best_feature[r, i] <- max(best_feature_tmp[, i])
  
  # avg k-folds
  avg_k_folds <- apply(acc, 2, mean) # avg k-fold accuracy 
  avg_k_folds <- c(`Random Forests` = avg_k_folds[1], 
                   `Logistic Regression` = avg_k_folds[2]) # add names
  out[r,1] <- avg_k_folds[1]
  out[r,2] <- avg_k_folds[2]


  # fit models using test set of current replication
  best <- names( which.max(avg_k_folds) )
  switch(best,
         `Random Forests` = {
           predTestRf <- predict(fitrf,
                                 type = "class",
                                 newdata = Xte)
           tabTestRf <- table(Xte$Survived, predTestRf)
           accBest <- sum(diag(tabTestRf))/sum(tabTestRf)
         },
         `Logistic Regression` = {
           predTestLr_prob <- predict(fitlr,
                                 type = "response",
                                 newdata = Xte)
           predTestLr <- ifelse(predTestLr_prob > tau, 1, 0)
           tabTestLr <- table(Xte$Survived, predTestLr)
           accBest <- sum(diag(tabTestLr))/sum(tabTestLr)
         }
  )
  out[r,3] <- best
  out[r,4] <- accBest
  
} # END Replication r


###########  
# Results #
###########

# tabulate rf vs. lr winnings
wins <- table(out[,3])
colnames(wins) <- c("Method", "Best")
print(wins)

# stats for test set
sm <- tapply(out[,4], out[,3], summary)
sm <- cbind(`Logistic Regression` = sm$`Logistic Regression`, 
            `Random Forests` = sm$`Random Forests`)
sm <- t(sm[c(1,4,6), ])
sm <- round(sm, 3)
print(sm)

# best features for replications 1 to 10
best_feature <- best_feature %>%
  as.data.frame(.) %>%
  slice(1:10) %>%
  mutate(Replication = seq(1:10)) %>%
  select(Replication,  everything()) %>%
  rename(`Top 1` = V1, `Top 2` = V2, `Top 3` = V3)

# top 3 features
top3 <- apply(best_feature, 2, function(x) names(which.max(table(x))))
print(top3)

# boxplot rf vs lr accuracy
boxplot(out$test ~ out$best)
stripchart(out$test ~ out$best, add = TRUE, vertical = TRUE,
           method = "jitter", pch = 19, col = adjustcolor("magenta3", 0.2))


############
# Plotting #
############

# line plot rf vs lr accuracy by replication
meanAcc <- colMeans(out[, 1:2])
sdAcc <- apply(out[, 1:2], 2, sd)/sqrt(R) # estimated mean accuracy standard deviation
matplot(out[, 1:2], type="l", lty=c(2,3), col = c("darkorange2", "deepskyblue4"),
        xlab = "Replications", ylab = "Accuracy")
# add confidence intervals
bounds1 <- rep( c(meanAcc[1] - 2*sdAcc[1], meanAcc[1] + 2*sdAcc[1]), each = R )
bounds2 <- rep( c(meanAcc[2] - 2*sdAcc[2], meanAcc[2] + 2*sdAcc[2]), each = R )
polygon(c(1:R, R:1), bounds1, col = adjustcolor("darkorange2", 0.2), border = FALSE)
polygon(c(1:R, R:1), bounds2, col = adjustcolor("deepskyblue4", 0.2), border = FALSE)
#
# add estimated mean line
abline(h = meanAcc, col = c("darkorange2", "deepskyblue4"))
#
# add legend
legend("bottomleft", fill = c("darkorange2", "deepskyblue4"),
       legend = c("random forests", "logistic regression"), bty = "n")  
  
save(list = ls(all = TRUE), file= "all.rda")
