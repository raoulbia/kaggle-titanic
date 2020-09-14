#
# =============================== Boosting - Simulated data example
#


# load packages
library(MASS)
library(adabag)


# generate training data from a bivariate Normal 
# distribution with mean (mu1, mu2) and variance (s1, s2)
set.seed(7891)
N <- 50
class <- rep(1:2, each = N)   # two classes, N observations each
s1 <- diag(c(1.5, 1))
s2 <- diag(c(1, 2))
mu1 <- c(-1,-1)
mu2 <- c(1, 2)
#
X <- rbind( mvrnorm(N, mu1, s1), 
            mvrnorm(N, mu2, s2) ) 
train <- data.frame(y = factor(class), X1 = X[,1], X2 = X[,2])


# plot training data colored by class
cols <- c("deepskyblue4", "darkorange3")
pch <- c(19, 17)
plot(train[,2:3], col = adjustcolor(cols[train$y], 0.8), 
     pch = pch[train$y])


# generate test data from the same process
set.seed(567)
N <- 50
class <- rep(1:2, each = N)   # two classes, N observations each
X <- rbind( mvrnorm(N, mu1, s1), 
            mvrnorm(N, mu2, s2) ) 
test <- data.frame(y = factor(class), X1 = X[,1], X2 = X[,2])

# plot test data -- pretend we don't know classification
plot(test[,2:3], pch = 19)


# perform boosting
fit <- boosting(y ~ ., data = train, boos = FALSE, coeflearn = "Freund")


# assess performance on training data
predTrain <- predict(fit, newdata = train)
table(train$y, predTrain$class)

# assess performance on test data
predTest <- predict(fit, newdata = test)
table(test$y, predTest$class)


# generate a heatmap to demonstrate how the classifier would classify new observations
# in the space of input variables for the training data
# - the redder the region, the more likely it is a triangle
# - the bluer the region, the more likely it is a circle
#
library(RColorBrewer)
L <- 200
X1seq <- seq(min(train$X1), max(train$X1), length = L)
X2seq <- seq(min(train$X2), max(train$X2), length = L)
datgrid <- expand.grid(X1seq, X2seq)
datgrid <- data.frame(y = NA, X1 = datgrid[,1], X2 = datgrid[,2])

predgrid <- predict(fit, newdata = datgrid)$prob[,1]
predgrid <- as.numeric(predgrid)

pal <- brewer.pal(11, "RdBu")
image( X1seq, X2seq, matrix(predgrid, L, L),
       col = adjustcolor(pal, 0.7))
points(train[,2:3], pch = pch[train$y])	

