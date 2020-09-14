#
# =============================== Boosting - Breast cancer classification
#


# load packages
library(MBCbook)
library(adabag)


# Load data
data(wdbc)
?wdbc
dat <- wdbc[,-1] # remove ID col
dat[,-1] <- scale(dat[,-1])  # standardize
dim(dat)

# split data into training, validation and test
N <- nrow(dat)
train <- sort( sample(1:N, size = N*0.7) )
val <- setdiff(1:N, train)
# randomly split into test and validation
test <- sort( sample(val, length(val)/2) )
val <- setdiff(val, test)


# boosting
# look at the effect of options: coeflearn, boos, etc.
fit <- boosting(Diagnosis ~. , data = dat[train,], coeflearn = "Freund", boos = FALSE)


# investigate overfitting and number of iterations/trees
?errorevol
err_train <- errorevol(fit, dat[train,])$error
err_val <- errorevol(fit, dat[val,])$error

# plot error paths  ---> there is evidence of overfitting
mat <- cbind(err_train, err_val)
cols <- c("deepskyblue4", "darkorange3")
matplot(mat, type = "l", lty = 1:2, col = cols,
        lwd = 2, xlab = "Number of trees", ylab = "Classification error")
legend("topright", cex = 0.75,
       legend = c("Train", "Test"), 
       lty = 1:2, col = cols, lwd = 2, bty = "n")
points(apply(mat, 2, which.min), apply(mat, 2, min), col = cols,
       pch = rep(c(15, 17), each = 2), cex = 1.5)


# prune tree
?predict.boosting
pred_val <- predict.boosting(fit, newdata = dat[val,])  # this uses all trees and it might overfit
pred_val
#
opt <- which.min(err_val)   # find smallest number of trees leading to minimum error
pred_val <- predict.boosting(fit, newdata = dat[val,], newmfinal = opt)
pred_val


# apply optimal model to test data for evaluation
pred_test <- predict.boosting(fit, newdata = dat[test,], newmfinal = opt)
pred_test
