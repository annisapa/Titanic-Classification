library(caTools)
library(readr) #read data .csv
library(dplyr) #data manipulation
library(caret) #confusion matrix
library(e1071) #naive bayes
library(ROCR)
library(partykit) #classification
library(caret) #classification

data<-read.csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
head(data)
str(data)

Titanic <- data %>%
  mutate(Pclass = factor(Pclass,level=c(1,2,3),labels=c('Upper','Middle','Lower')),
         Sex = factor(Sex), 
         Survived = factor(Survived,level=c(0,1),labels=c('No','Yes'))) %>%
  filter(Age != is.na(Age)) %>%
  select(-c("Ticket", "Cabin", "Name"))

set.seed(123) # for reproducible results
sample.size <- floor(0.75 * nrow(Titanic))
train.index <- sample(seq_len(nrow(Titanic)), size = sample.size)
train <- Titanic[train.index, ]
test <- Titanic[- train.index, ]
  
  model_naive <- naiveBayes(formula = Survived ~., data = train,laplace = 1)
  pred_label_naive <- predict(model_naive, test, type = "class")
  head(data.frame(actual = test$Survived, prediction = pred_label_naive))


# get the probability prediction
prob_survive <- predict(model_naive, test, type = "raw")

# prepare dataframe for  ROC
data_roc <- data.frame(prob = prob_survive[,2], # probability of positive class(survived)
                       labels = as.numeric(test$Survived == "No")) #get the label as the test data who survived
head(data_roc)

naive_roc <- ROCR::prediction(data_roc$prob, data_roc$labels) 

# ROC curve
plot(performance(naive_roc, "tpr", "fpr"), #tpr = true positive rate, fpr = false positive rate
     main = "ROC")
abline(a = 0, b = 1)

mat1 <- confusionMatrix(pred_label_naive, test$Survived, positive = "No")
mat1
#RandomForest
set.seed(283)
ctrl <- trainControl(method = "cv", number = 5,repeats = 3)
model_rf <- train(form = Survived ~., data = train, trControl= ctrl)
model_rf
pred_test_rf <- predict(model_rf, newdata = test, type = "raw")

mat2 <- confusionMatrix(data = pred_test_rf, reference = test$Survived, positive = "No")
mat2

prob_survive_rf <- predict(model_rf,test, type = "prob")

data_roc2 <- data.frame(prob = prob_survive_rf[,2], # probability of positive class(survived)
                        labels = as.numeric(test$Survived == "No")) #get the label as the test data who survived
rf_roc <- ROCR::prediction(data_roc2$prob, data_roc2$labels) 

# ROC curve
plot(performance(rf_roc, "tpr", "fpr"), #tpr = true positive rate, fpr = false positive rate
     main = "ROC")
abline(a = 0, b = 1)