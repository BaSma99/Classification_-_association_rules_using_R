################## Import important Libraries ################## 
library(ROSE) 
library(mice)
library(rpart.plot)  #for plotting decision trees
library(workflows)   #modeling workflows
library(ISLR)        #import desired dataset
library(finalfit)    #to create plots for regression results
library(ggplot2)     #for visualization and plotting
library(caTools)     #for ROC, and AUC
library(corrplot)    #to plot correlation matrix using Heatmap
library(rpart)       #for decision tree application
library(caret)       #for classification and regression training
library(dplyr)       #for data manipulation
library(xgboost)     #for xGboost model
library(magrittr)    #A Forward-Pipe Operator for R
library(car)         #for regression
library(stringr)     #for working with strings easy
library(reshape2)    #for reshaping data
library(party)       #for recursive partitioning
library(partykit)    #for recursive partitioning
library(recipes)     #for data modeling
library(themis)      #for dealing with unbalanced data
library(caret)       #for classification and regression training
library(mltools)     #Machine learning tools
library(Matrix)      #dense matrix classes and methods
library(e1071)       #for statistics and probability
library(dplyr)       #grammer for data manipulation
library(cowplot)     #for plotting
library(magrittr)    #forward pipe operator
require(xgboost)     
require(Matrix)
require(data.table)

#################### load the DataSet #################### 
#Read the dataSet
Churn_DataSet <- read.csv("C:/Users/hp/Downloads/Assignment 2 (1)/Assignment 2/Churn_Dataset.csv")
head(Churn_DataSet)

#print the summary of the dataset
summary(Churn_DataSet)

------------------------------------------------------- #Step 1
#################### Scatter plot matrix ###################

#the pairs function can't operate with non-numeric arguments, so we must choose the numeric columns.
data = subset(Churn_DataSet, select = c("tenure","MonthlyCharges","TotalCharges"))
pairs(data, pch = 19)

scatterplotMatrix(~tenure + MonthlyCharges + TotalCharges, data = Churn_DataSet,
                  diagonal = FALSE,             # Remove kernel density estimates
                  regLine = list(col = "green", # Linear regression line color
                                 lwd = 3),      # Linear regression line width
                  smooth = list(col.smooth = "red",   # Non-parametric mean color
                                col.spread = "blue"))

#################### Correlation matrix ###################
correlationMatrix <- cor(data)
print(correlationMatrix)

corrplot(cor(data),        
         method = "circle",                
         type = "full",                   
         diag = TRUE,                     
         tl.col = "black",                
         bg = "white",                    
         title = "",                      
         col = NULL,                      
         tl.cex =0.7,
         cl.ratio =0.2)   


####################### Heat MAp #######################
cormat <- round(x = cor(data), digits = 2)
head(cormat)

melted_cormat <- melt(cormat)
head(melted_cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill = value)) +
  geom_tile()

heatdf <- as.matrix(data)

library(Hmisc)
ccs=as.matrix(data)
correlation=rcorr(ccs, type="pearson") # You can also use "spearman"
correlation_matrix=data.matrix(correlation$r)
correlation_matrix
heatmap(correlation_matrix)

------------------------------------------------------- #Step 2
#################### Check for missing values #################### 
anyNA(Churn_DataSet)

#print the number of missing values
sum(is.na(Churn_DataSet))

#find the columns with missing values(NA)
NA_list <- colnames(Churn_DataSet)[apply(Churn_DataSet, 2, anyNA) ]
NA_list

#remove the missing values
Churn_DataSet_Clean <- Churn_DataSet %>%
  na.omit()


#check again if any missing values exits
anyNA(Churn_DataSet_Clean)
sum(is.na(Churn_DataSet_Clean))

########## drop the customerID columns from the dataset ##########
Churn_DataSet_Clean <- Churn_DataSet_Clean %>% select(-customerID)

########## remove the duplicated values from the dataset ##########
duplicated(Churn_DataSet_Clean)
Churn_DataSet_Clean = subset(Churn_DataSet_Clean, 
                             !duplicated(Churn_DataSet_Clean))

########## convert categorical variables to numerical ########## 
md.pattern(Churn_DataSet_Clean, plot = FALSE)
------------------------------------------------------- #Step 3
#################### Split the DataSet #######################
set.seed(123)
df <- sample.split(Y = Churn_DataSet_Clean$Churn, SplitRatio = 0.8)
trainingSet <- subset(x = Churn_DataSet_Clean, df == TRUE)
testingSet <- subset(x = Churn_DataSet_Clean, df == FALSE)
dim(trainingSet)
dim(testingSet)

#################### Build Decision Tree #################### 
DecisionTreeModel <- rpart(Churn ~ ., 
                           data = trainingSet, 
                           method = "class")

##################### plot the model #######################
rpart.plot(DecisionTreeModel)
plotcp(DecisionTreeModel)

##################### model predection #####################
y_pred <- predict(DecisionTreeModel, 
                  newdata = testingSet ,
                  type = "class")
  
################ plot confusion matrix plot #################
cm1 <- confusionMatrix(as.factor(testingSet$Churn), factor(y_pred), mode = "prec_recall", dnn = c("Actual", "Prediction"))
cm1 #0.7839

plot_confusionMatrix <- function(cm) {
  plt <- as.data.frame(cm$table)
  plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
  ggplot(plt, aes(Actual , Prediction, fill= Freq)) +
    geom_tile() + geom_text(aes(label=Freq)) +
    scale_fill_gradient(low="white", high="#009194") +
    labs(x = "Prediction",y = "Actual") +
    scale_x_discrete(labels=c("Class_1","Class_2")) +
    scale_y_discrete(labels=c("Class_2","Class_1"))
}
plot_confusionMatrix(cm1)

###################  plot ROC curve ################### 
ROSE::roc.curve(testingSet$Churn, y_pred) #0.683

------------------------------------------------------- #Step 4
                  ################ using gini splitting method #################
set.seed(123) 
ct <- trainControl(method = "cv", number = 10)

#fit a decision tree model and use k-fold CV to evaluate performance
decisionTreeGini <- train(Churn ~ . , data = trainingSet,
                        method = "rpart",
                        parms = list(split = "gini"), 
                        trControl = ct,
                        tuneLength = 100)

################## Check accuracy #####################
y_pred_gini <- predict(decisionTreeGini, newdata = testingSet) #0.7868          

############### print the confusion matrix #############

confusionMatrix()

giniCM <- confusionMatrix(as.factor(testingSet$Churn), factor(y_pred_gini), mode = "prec_recall", dnn = c("Actual", "Prediction"))
giniCM 

confusionMatrix(data = y_pred_gini, reference =  as.factor(testingSet$Churn))
plot_confusionMatrix(giniCM)
################### plot ROC curve ##################### 
ROSE::roc.curve(testingSet$Churn, y_pred_gini) #0.691



              ################### split tree with information ##################
decisionTreeInformation <- rpart(Churn ~ ., 
                                 data = trainingSet, 
                                 method = "class", 
                                 parms = list(split = "information")) 
decisionTreeInformation
plotcp(decisionTreeInformation)

################## Check accuracy #####################
y_i_pred <- predict(decisionTreeInformation, newdata = testingSet , type = "class")

############### print the confusion matrix #############
infoCM <- confusionMatrix(as.factor(testingSet$Churn), factor(y_i_pred), mode = "prec_recall", dnn = c("Actual", "Prediction"))
infoCM #Accuracy : 0.7825

plot_confusionMatrix(infoCM)

################### plot ROC curve  ################### 
ROSE::roc.curve(testingSet$Churn, y_i_pred) #0.677


                      #################### Prune the tree #####################
decitionTreePrune <- rpart(Churn ~ ., 
                           data = trainingSet, 
                           method = "class", 
                          control = rpart.control(cp = 0.0082, 
                                                  maxdepth = 3,
                                                  minsplit = 2))
rpart.plot(decitionTreePrune)
plotcp(decitionTreePrune)

############## accuracy of the pruned tree #############
prune_pred <- predict(decitionTreePrune, newdata = testingSet , type = "class")

############### print the confusion matrix #############
pruneCM <- confusionMatrix(as.factor(testingSet$Churn), factor(prune_pred), mode = "prec_recall", dnn = c("Actual", "Prediction"))
pruneCM  #Accuracy : 0.7903

plot_confusionMatrix(pruneCM)
################### plot ROC curve  ################### 
ROSE::roc.curve(testingSet$Churn, prune_pred) #0.663

----------------------------------------------------- #Step 5
Churn_DataSet_Clean$Churn = factor(Churn_DataSet_Clean$Churn, level = c("Yes", "No"), 
                                     labels = c(0,1))


set.seed(42)
parts <- sample.split(Y = Churn_DataSet_Clean$Churn, SplitRatio = 0.8)
train <- subset(x = Churn_DataSet_Clean, parts == TRUE)
test <- subset(x = Churn_DataSet_Clean, parts == FALSE)

X_train = data.matrix(train[,-20])                  
y_train = train[,20]                               

X_test = data.matrix(test[,-20])                    
y_test = test[,20]                                   

# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

################ train XGBOOST model ################ 
XGBmodel <- xgboost(data = xgboost_train,# the data   
                 max.depth=3, , # max depth 
                 nrounds=70) # max number of boosting iterations

summary(XGBmodel)


############ prediction of XGBOOST model #############
pred_test_xgb = predict(XGBmodel, newdata= X_test)
xgbpred = as.factor((levels(y_test))[round(pred_test_xgb)])
xgbpred

accuracy_xgb <- mean(xgbpred == test$Churn)
print(paste('Accuracy for test is found to be', accuracy_xgb))
precision_xgb <- posPredValue(xgbpred, test$Churn, positive="1")
print(paste('precision for test is found to be',precision_xgb))
recall_xgb <- sensitivity(xgbpred, test$Churn, positive="1")
print(paste('Recall for test is found to be',recall_xgb))
F1_xgb <- (2 * precision_xgb * recall_xgb) / (precision_xgb + recall_xgb)
print(paste('F1-score for test is found to be',F1_xgb))


############### print the confusion matrix #############
xgbcm = confusionMatrix(y_test, xgbpred)
print(xgbcm)  #Accuracy : 0.7996


cmxgb <- confusionMatrix(factor(xgbpred), factor(y_test), dnn = c("Prediction", "Reference"))
print(cmxgb)
plt <- as.data.frame(cmxgb$table)
plt$Prediction <- factor(plt$Prediction, levels=rev(levels(plt$Prediction)))
ggplot(plt, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("Class_1","Class_2")) +
  scale_y_discrete(labels=c("Class_1","Class_2"))


################### plot ROC curve  ################### 
ROSE::roc.curve(y_test, pred_test_xgb) #0.839

--------------------------------------------------------------- #step 6
install.packages('devtools')
devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/keras")
install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
install_tensorflow(method = "auto")
install.packages("keras")
install_tensorflow(envname = "tf")
install_keras(Tensorflow = "1.13.1",
              restart_session = FALSE
)
library(tensorflow)
library(keras)
library(magrittr)
library(reticulate)
library(caTools)

set.seed(123)
train_keras_x <- array(X_train, dim = c(dim(X_train)[1], prod(dim(X_train)[-1]))) 
test_keras_x <- array(X_test, dim = c(dim(X_test)[1], prod(dim(X_test)[-1]))) 

#converting the target variable to once hot encoded vectors using keras inbuilt function
train_keras_y<-to_categorical(y_train,2)
test_keras_y<-to_categorical(y_test,2)


model <- keras_model_sequential() 
model %>%
  layer_dense(units = 128, input_shape = 19) %>%
  layer_dropout(rate=0.1)%>%
  layer_activation(activation = 'tanh') %>%
  layer_dense(units = 64)%>%
  layer_activation(activation = 'tanh')%>%
  layer_dropout(rate=0.1)%>%
  layer_dense(units = 2) %>%
  layer_activation(activation = 'sigmoid')
  
#compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#fitting the model on the training dataset
model %>% fit(train_keras_x, train_keras_y, epochs = 50, batch_size = 128)

nn_pred1 <- model %>% predict(test_keras_x) 
nn_pred1
table(test_keras_y, round(nn_pred1))


nn_pred1 = predict(model,data.matrix(test_keras_x), type = "response")
nn_pred1 <-as.factor(as.numeric(nn_pred1>0.5))

# Confusion Matrix
cm_nn1 <- confusionMatrix(as.factor(test_keras_y), factor(nn_pred1),  mode = "prec_recall", dnn = c("Actual", "Prediction"))
cm_nn1                                                                        
plot_confusionMatrix(cm_nn1)

ROSE::roc.curve(test_keras_y, nn_pred1)  
       

## change the activation function and rate

model2 <- keras_model_sequential() 
model2 %>%
  layer_dense(units = 128, input_shape = 19) %>%
  layer_dropout(rate=0.4)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 64)%>%
  layer_activation(activation = 'relu')%>%
  layer_dropout(rate=0.4)%>%
  layer_dense(units = 2) %>%
  layer_activation(activation = 'sigmoid')

#compiling the defined model with metric = accuracy and optimiser as adam.
model2 %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#fitting the model on the training dataset
model2 %>% fit(train_keras_x, train_keras_y, epochs = 50, batch_size = 128)

nn_pred2 <- model2 %>% predict(test_keras_x) 
nn_pred2
table(test_keras_y, round(nn_pred2))


nn_pred2 = predict(model2,data.matrix(test_keras_x), type = "response")
nn_pred2 <-as.factor(as.numeric(nn_pred2>0.5))

# Confusion Matrix
cm_nn2 <- confusionMatrix(as.factor(test_keras_y), factor(nn_pred2),  mode = "prec_recall", dnn = c("Actual", "Prediction"))
cm_nn2                                 
plot_confusionMatrix(cm_nn2)

ROSE::roc.curve(test_keras_y, nn_pred2) 
#############################################################################
###############################Bart B#######################################
library("arules")
library("arulesViz")
library('reticulate')
library("readr")
library("RColorBrewer")

install.packages("arules")
install.packages("arulesViz")
install.packages('reticulate')
install.packages('keras')
install.packages("readr")
install.packages("RColorBrewer")
#Load the transaction dataset
data=read.transactions("C:/Users/hp/Downloads/Assignment 2 (1)/Assignment 2/transactions.csv",
                       format='basket',header =TRUE ,sep=',')
summary(data)

########################### step1 ########################### 

# plot the frequency of the items
itemFrequency(data[,1:3])
itemFrequencyPlot(data, support = 0.1)

#PLot for Top 10 Transactions
itemFrequencyPlot(data, topN = 10)

#Association Rule
apriori(df)

############################### step2  ############################### 

# Generate association rules using minimum support of 0.002, minimum confidence of
# 0.20, and maximum length of 3. 

association_rule_1 <- apriori(data, parameter = list(support = 0.002,
                                                     confidence =0.20,
                                                     maxlen = 3))

association_rule_1

# Display the rules, sorted by descending lift value
association_rule_lift_sort <- sort(association_rule_1, by = "lift")
# Display the rules, sorted by descending support value
inspect(association_rule_lift_sort)

############################### step3. ############################### 

# Select the rule from QII-b with the greatest lift. Compare this rule with the highest lift
# rule for maximum length of 2.


association_rule_2 <- apriori(data, parameter = list(support = 0.002,
                                                     confidence =0.20,
                                                     maxlen = 2))

# Display the rules, sorted by descending lift value
association_rule_lift_sort2 <- sort(association_rule_2, by = "lift")

# Display the rules, sorted by descending support value
inspect(association_rule_lift_sort2)

# The highest lift rule with maxlen = 3
inspect(association_rule_lift_sort[1])

#he highest lift rule with maxlen = 2
inspect(association_rule_lift_sort2[1])


