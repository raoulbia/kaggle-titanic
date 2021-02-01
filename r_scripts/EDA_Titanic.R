library(tidyverse)
library(rio)
library(randomForest)
library(party)
library(rstudioapi)
library("ggpubr") # ggscatter
library(psych) # describeBy

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
rm(list = ls())
options(digits=4, scipen=999)


#############
# load data #
#############

# data  <- import('../local-data/output/titanic_train_clean.csv', setclass = "tibble")
tr  <- import('../local-data/input/train.csv', setclass = "tibble")
te  <- import('../local-data/input/test.csv', setclass = "tibble")
te$Survived <- NA
df <- rbind(tr, te)
head(df)

tab = table(df$Embarked)
barplot(tab, xlab="Genre",ylab="Frequency",col = heat.colors(16))

ptab = prop.table(table(df$Embarked))
barplot(ptab, xlab="Genre",ylab="Proportion",col = heat.colors(16))

summary(df$Age)

hist(df$Age, 
     freq=FALSE, 
     xlab="x", 
     breaks="FD",
     main="Histogram and density estimate")
lines(density(df$Age, na.rm = T), lwd=2, col="blue")

boxplot(df$Age)


# ANALYZING TWO CATEGORICAL VARIABLES

tab2 = table(df$Sex, df$Embarked) # Joint Frequency
mosaicplot(tab2)


# ANALYZING TWO NUMERICAL VARIABLES

cor(df$Age,df$Fare, 
    use = "pairwise.complete.obs" # deal with missing values
    )

cor.test(df$Age,df$Fare, 
         method = "pearson",
         use = "pairwise.complete.obs" # deal with missing values
         )


ggscatter(df, 
          x = "Age", 
          y = "Fare",
          add = "reg.line", 
          conf.int = TRUE,
          cor.coef = TRUE, 
          cor.method = "pearson",
          xlab = "Age", 
          ylab = "Fare")


# ANALYZING A NUMERIC VARIABLE GROUPED BY A CATEGORICAL VARIABLE

boxplot(df$Age~df$Pclass)

describeBy(df$Age, df$Pclass)

summary(df)
