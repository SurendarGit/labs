#Data Preprocessing

#Importing Dataset
dataset = read.csv("Data.csv")


#Taking care of missing data

#If na=true, then calculate average else keep the same x value
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(X) mean(X, na.rm = TRUE))
                     ,dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(X) mean(X, na.rm = TRUE))
                        ,dataset$Salary)


#Categorical data
dataset$Country = factor(dataset$Country,levels = c("France","Spain","Germany"),
                         labels = c("1","2","3"))
dataset$Purchased = factor(dataset$Purchased,levels = c("No","Yes"),
                           labels = c("0","1"))




# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set=subset(dataset,split == TRUE)
test_set=subset(dataset,split == FALSE)

#Scaling data
training_set[,2:3]=scale(training_set[,2:3])
test_set[,2:3]=scale(test_set[,2:3])