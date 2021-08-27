############################ SVM Number Recogniser #################################
# 1. Business Understanding
# 2. Data Understanding
# 3. Data Preparation
# 4. Model Building 
#  4.1 Linear kernel
#  4.2 RBF Kernel
# 5 Hyperparameter tuning and cross validation

#####################################################################################

# 1. Business Understanding: 

#The objective is to identify each of a large number of black-and-white
#rectangular pixel displays as one of the 10 digits in the English charecters

#####################################################################################

# 2. Data Understanding: 
# http://yann.lecun.com/exdb/mnist/
# Number of Instances: 60,000
# Number of Attributes: 785 

#3. Data Preparation: 


#Loading Neccessary libraries

library(dplyr)
library(readr)
library(ggplot2)
library("caret")
library(kernlab)
library(gridExtra)


#Loading Data

Data_train <- read_csv("mnist_train.csv",  col_names = F)
Data_test <- read_csv("mnist_test.csv", col_names = F)

colnames(Data_train) <- c("Digit",seq(1, 784))
colnames(Data_test) <- c("Digit",seq(1, 784))

#Understanding Dimensions

dim(Data_train)
dim(Data_test)

#Structure of the dataset

str(Data_train)
str(Data_test)

#printing first few rows

head(Data_train)
head(Data_test)

#Exploring the data

summary(Data_train)
summary(Data_test)

#checking missing value

sum(is.na(Data_train))
sum(duplicated(Data_train))

# No missing values or duplicate values in dataset

# The dataset very large to provide the result in required time and computation limit
# Lets take sample of around 15% of available training data for the Model Building

set.seed(1)
train.indices = sample(1:nrow(Data_train), 0.7*nrow(Data_train))
train_digit = Data_train[train.indices, ]

#Making our target class to factor

train_digit$Digit<-factor(train_digit$Digit)
Data_test$Digit<-factor(Data_test$Digit)

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(Digit~ ., data = train_digit, scale = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, Data_test)

library(e1071)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,Data_test$Digit)
# Model gives 91.7% Accurace and Individual Digit recognition accuracy comes around 85-98%

#Using RBF Kernel
Model_RBF <- ksvm(Digit~ ., data = train_digit, scale = F, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, Data_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,Data_test$Digit)

Model_linear
Model_RBF
# Model gives 95% Accurace and Individual Digit recognition accuracy comes around 94-98%

# RBF Model gives higher accuracy but compared to computational complexity, additional 4% 
# accuracy can be discarded
# So select Linear model for Cross-Validation



############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl_linear <- trainControl(method="cv", number=5)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.C=c(0.75, 1, 1.25) )


#train function takes Target ~ Prediction, Dataset, Method = AlgorithmName
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(Digit~., data=train_digit, method="svmLinear", metric=metric, 
                 tuneGrid=grid, trControl=trainControl_linear)

print(fit.svm)

plot(fit.svm)

# From the plot, it can be confirmed that with Cost (C) at 1, model gives 91% accuracy.

