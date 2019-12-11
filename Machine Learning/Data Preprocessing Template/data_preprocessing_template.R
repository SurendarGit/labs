#Data Preprocessing

#Importing Dataset
dataset = read.csv("Data.csv")
#dataset = dataset[,2:3] # taking a subset inside a datset


# Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set=subset(dataset,split == TRUE)
test_set=subset(dataset,split == FALSE)

#Scaling data - MultiLine comments Ctrl+Enter+Shift+C
# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])