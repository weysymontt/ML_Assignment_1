# install.packages("mlr")
# install.packages("mlbench")
# install.packages("randomForest")
# install.packages("e1071")
# install.packages("ggplot2")
# install.packages("GGally")
# install.packages("FNN")
# install.packages("RWeka")
# install.packages("mda")
# install.packages("modeltools")
# install.packages('class')
# install.packages('caret')
# install.packages("lattice")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("RoughSets")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("klaR")
# install.packages("fossil")
# install.packages("ClusterR")
# install.packages("mclust")
# install.packages("corrplot")
# install.packages("rlang")
# install.packages("neuralnet")
library(neuralnet)
library(corrplot)
library(fossil)
library(factoextra)
library(klaR)
library(MASS)
library(mlr)
library(ggplot2)
library(GGally)
library(FNN)
library(RWeka)
library(mda)
library(modeltools)
library(class)
library(caret)
library(dplyr)
library(RoughSets)
library(rpart)
library(rpart.plot)
library(e1071)
library(cluster)
library(ClusterR)
library(mclust, quietly=TRUE)
library(mlbench)



setwd("C:/Master/Machine Learning/Assignment_1")
students <- read.table("Dataset/student-mat.txt", header = TRUE, stringsAsFactors=FALSE)

students$Gmean <- (students$G1+students$G2+students$G3)/3
students$AcadPerformance <- cut(students$Gmean, breaks = c(0,10,20), labels = c("0","1"))

students$G1 <- NULL
students$G2 <- NULL
students$G3 <- NULL
students$Gmean <- NULL

set.seed(5)
students <- students[sample(nrow(students)),]
set.seed(5)

students$school <- ifelse(students$school=="GP",1,2)
students$sex <- ifelse(students$sex=="F",1,2)
students$address <- ifelse(students$address=="U",1,2)
students$famsize <- ifelse(students$famsize=="LE3",1,2)
students$Pstatus <- ifelse(students$Pstatus=="T",1,2)
students$Mjob <- ifelse(students$Mjob=="at_home",1,
                        ifelse(students$Mjob=="health",2,
                               ifelse(students$Mjob=="services",3,
                                      ifelse(students$Mjob=="teacher",4,5))))
students$Fjob <- ifelse(students$Fjob=="at_home",1,
                        ifelse(students$Fjob=="health",2,
                               ifelse(students$Fjob=="services",3,
                                      ifelse(students$Fjob=="teacher",4,5))))
students$reason <- ifelse(students$reason=="course",1,
                          ifelse(students$reason=="home",2,
                                 ifelse(students$reason=="reputation",3,4)))
students$guardian <- ifelse(students$guardian=="father",1,
                            ifelse(students$guardian=="mother",2,3))
students$schoolsup <- ifelse(students$schoolsup=="yes",1,2)
students$famsup <- ifelse(students$famsup=="yes",1,2)
students$paid <- ifelse(students$paid=="yes",1,2)
students$activities <- ifelse(students$activities=="yes",1,2)
students$nursery <- ifelse(students$nursery=="yes",1,2)
students$higher <- ifelse(students$higher=="yes",1,2)
students$internet <- ifelse(students$internet=="yes",1,2)
students$romantic <- ifelse(students$romantic=="yes",1,2)


studentsNum <- students
trainingNum <- studentsNum[1:300,]
testingNum <- studentsNum[301:395,]


varlist <- names(students)[1:30]
students$school <- factor(students$school)
students$sex <- factor(students$sex)
#students$age <- factor(students$age)
students$address <- factor(students$address)
students$famsize <- factor(students$famsize)
students$Pstatus <- factor(students$Pstatus)
students$Medu <- factor(students$Medu)
students$Fedu <- factor(students$Fedu)
students$Mjob <- factor(students$Mjob)
students$Fjob <- factor(students$Fjob)
students$reason <- factor(students$reason)
students$guardian <- factor(students$guardian)
#students$traveltime <- factor(students$traveltime)
#students$studytime <- factor(students$studytime)
#students$failures <- factor(students$failures)
students$schoolsup <- factor(students$schoolsup)
students$famsup <- factor(students$famsup)
students$paid <- factor(students$paid)
students$activities <- factor(students$activities)
students$nursery <- factor(students$nursery)
students$higher <- factor(students$higher)
students$internet <- factor(students$internet)
students$romantic <- factor(students$romantic)
students$famrel <- factor(students$famrel)
#students$freetime <- factor(students$freetime)
#students$goout <- factor(students$goout)
#students$Dalc <- factor(students$Dalc)
#students$Walc <- factor(students$Walc)
#students$health <- factor(students$health)
#students$absences <- factor(students$absences)

training <- students[1:300,]
testing <- students[301:395,]


studentsFactors <- students

studentsFactors$age <- factor(students$age)
studentsFactors$traveltime <- factor(students$traveltime)
studentsFactors$studytime <- factor(students$studytime)
studentsFactors$failures <- factor(students$failures)
studentsFactors$freetime <- factor(students$freetime)
studentsFactors$goout <- factor(students$goout)
studentsFactors$Dalc <- factor(students$Dalc)
studentsFactors$Walc <- factor(students$Walc)
studentsFactors$health <- factor(students$health)
studentsFactors$absences <- factor(students$absences)

trainingFactors <- studentsFactors[1:300,]
testingFactors <- studentsFactors[301:395,]



# Feature subset selection

# First step: drop highly correlated variables
studentsCorr <- studentsNum
studentsCorr$AcadPerformance <- as.numeric(studentsCorr$AcadPerformance)
correlationMatrix <- cor(studentsCorr)
print(correlationMatrix)

source("http://www.sthda.com/upload/rquery_cormat.r")
rquery.cormat(studentsCorr)
studentsFS <- students
studentsFS$Walc <- NULL
studentsFS$Fedu <- NULL
studentsFS$address <- NULL
studentsFS$traveltime <- NULL
studentsFS$school <- NULL


# Second step

fit_glm = glm(AcadPerformance~.,studentsFS,family = "binomial")
summary(fit_glm)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
# We stay with: sex, failures, schoolsup, famsup, internet, health

oneFeatureSubset <- c("failures", "AcadPerformance")
studentsNum <- studentsNum[oneFeatureSubset]
trainingNum <- trainingNum[oneFeatureSubset]
testingNum <- testingNum[oneFeatureSubset]
students <- students[oneFeatureSubset]
training <- training[oneFeatureSubset]
testing <- testing[oneFeatureSubset]
studentsFactors <- studentsFactors[oneFeatureSubset]
trainingFactors <- trainingFactors[oneFeatureSubset]
testingFactors <- testingFactors[oneFeatureSubset]



########################################################################################################################

# NOTE that apriori probability of correct classification based on the frequency of the most common value is 54,68%.

# SUPERVISED CLASSIFICATION:

# Non-probabilistic
# k-nearest neighbors ########################################

folds <- cut(seq(1,nrow(students)),breaks=10,labels=FALSE)
accKNN <- c()
for(x in 1:20){
  accKNNforK <- c()
  for(i in 1:10){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- students[testIndexes,]
    testData$AcadPerformance <- as.numeric(levels(testData$AcadPerformance))[testData$AcadPerformance]
    trainData <- students[-testIndexes,]
    trainData$AcadPerformance <- as.numeric(levels(trainData$AcadPerformance))[trainData$AcadPerformance]
    knn <- knn(train=training, test=testing, cl=training$AcadPerformance, k=x)
    fold.acc.knn <- 100 * sum(testing$AcadPerformance == knn)/NROW(testing$AcadPerformance)
    accKNNforK <- c(accKNNforK, fold.acc.knn)
  }
  meanAccForK <- mean(accKNNforK)
  accKNN <- c(accKNN, meanAccForK)
}

K <- c(1:20)
knnResults <- data.frame(K, accKNN)
knnResults %>%
  ggplot( aes(x=K, y=accKNN)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(0, 20, by = 1)) +
  scale_y_continuous(breaks = seq(0, 100, by = 1)) +
  labs(x= "K nearest neighbour", y = "Accuracy in %")

knnResult <- max(accKNN)/100

##############################################################


# Non-probabilistic
# Rule Induction #############################################

# Not enough explanatory variables.

##############################################################


# Non-probabilistic
# Classification Trees #######################################

accuracy_tune <- function(fit) {
  predict_unseen <- predict(fit, trainingFactors, type = 'class')
  accClassTree <- mean(predict_unseen == testingFactors$AcadPerformance)
}

control <- rpart.control(minsplit = 5,
                         minbucket = 5,
                         maxdepth = 3,
                         cp = 0.1)
tune_fit <- rpart(AcadPerformance~., data=trainingFactors, method='class', control = control)
accClassTree <- accuracy_tune(tune_fit)

classTree <- rpart(AcadPerformance~., data=trainingFactors, method='class')
rpart.plot(classTree, extra = 'auto')
predict_unseen <-predict(classTree, testingFactors, type = 'class')
accClassTree <- mean(predict_unseen == testingFactors$AcadPerformance)

##############################################################


# Non-probabilistic
# Support Vector Machines ####################################

svmfit = svm(AcadPerformance ~ ., data = training, kernel = "linear", cost = 0.15, scale = FALSE, type = "C-classification")
accSVM <- mean(predict(svmfit, newdata = testing) == testing$AcadPerformance)

##############################################################


# Non-probabilistic
# Artificial Neural Network ##################################

require(neuralnet)

nn=neuralnet(AcadPerformance~.,data=trainingNum, hidden=3,act.fct = "logistic",
             linear.output = FALSE)

plot(nn)

PredictNN <- compute(nn,testingNum)
prob <- PredictNN$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred <- pred[,2]
accANN <- mean(pred == testing$AcadPerformance)

##############################################################


# Probabilistic
# Logistic Regression ########################################

trainingLR <- training
testingLR <- testing
trainingLR$AcadPerformance <- trainingLR$AcadPerformance
testingLR$AcadPerformance <- testing$AcadPerformance
LRmodel <- glm(AcadPerformance ~.,binomial(link = "logit"),data=trainingLR)
summary(LRmodel)

fitted.results.LR <- predict(LRmodel,newdata=testingLR,type='response')
fitted.results.LR <- ifelse(fitted.results.LR > 0.5,1,0)

accLR <- mean(fitted.results.LR == testingLR$AcadPerformance)

##############################################################


# Probabilistic
# Bayesian Classifiers #######################################

Naive_Bayes_Model=naiveBayes(AcadPerformance ~., data=training)
NB_Predictions=predict(Naive_Bayes_Model,testing)

accNB <- mean(NB_Predictions == testing$AcadPerformance)

##############################################################


# Probabilistic
# Discriminant analysis ######################################

model.linear <- lda(AcadPerformance~., data = training)
predictions.linear <- model.linear %>% predict(testing)
accCB <- mean(predictions.linear$class==testing$AcadPerformance)

model.mixture <- mda(AcadPerformance~., data = training)
predictions.mixture <- model.mixture %>% predict(testing)
mean(predictions.mixture==testing$AcadPerformance)

model.flexible <- fda(AcadPerformance~., data = training)
predictions.flexible <- model.flexible %>% predict(testing)
mean(predictions.flexible==testing$AcadPerformance)

model.regularized <- rda(AcadPerformance~., data = training)
predictions.regularized <- model.regularized %>% predict(testing)
mean(predictions.regularized$class==testing$AcadPerformance)

##############################################################



# UNSUPERVISED CLASSIFICATION:

# Hierarchical Clustering ####################################

clusters <- hclust(dist(students[, 1]), method = 'centroid')
plot(clusters)
numberClusters <- 2
clusterCut <- cutree(clusters, numberClusters)
clusterTable <- as.data.frame(table(clusterCut, students$AcadPerformance))
class1 <- c(clusterTable$Freq[1:numberClusters])
class2 <- c(clusterTable$Freq[(numberClusters+1):(numberClusters*2)])

correct <- 0
incorrect <- 0

contador = 0
for (freq in class1){
  contador = contador + 1
  maxValue <- ifelse(freq>=class2[contador],freq,class2[contador])
  minValue <- ifelse(freq>=class2[contador],class2[contador],freq)
  correct <- correct + maxValue
  incorrect <- incorrect + minValue
}

accHC <- correct/(correct+incorrect)


clusters <- hclust(dist(students[, 1]))
plot(clusters)
numberClusters <- 2
clusterCut <- cutree(clusters, numberClusters)
clusterTable <- as.data.frame(table(clusterCut, students$AcadPerformance))

class1 <- c(clusterTable$Freq[1:numberClusters])
class2 <- c(clusterTable$Freq[(numberClusters+1):(numberClusters*2)])

correct <- 0
incorrect <- 0

contador = 0
for (freq in class1){
  contador = contador + 1
  maxValue <- ifelse(freq>=class2[contador],freq,class2[contador])
  minValue <- ifelse(freq>=class2[contador],class2[contador],freq)
  correct <- correct + maxValue
  incorrect <- incorrect + minValue
}

accHC <- correct/(correct+incorrect)

##############################################################


# Partitional clustering #####################################

# Not enought explanatory variables. More cluster centers than distinct data points.

##############################################################


# Probabilistic clustering ###################################

X = studentsNum[, 1:30]
Y = studentsNum[, 31]

modelPC <- Mclust(data.frame (X,Y, G=4))
modelPC$classification # All predictions are 1, the algorithm is useless.

##############################################################



# Metaclassifiers: simple and weighted majority votes for supervised classification

knnFit <- knn(train=training, test=testing, cl=training$AcadPerformance, k=7)
#ruleInductionFit <- pred.vals$predictions
classTreeFit <- predict_unseen
svmFit <- predict(svmfit, newdata = testing)
logRegFit <- fitted.results.LR
descreteBayesFit <- NB_Predictions
continuousBayesFit <- predictions.linear$class

globalPredictions <- rowMeans(cbind(knnFit,classTreeFit,svmFit,logRegFit,descreteBayesFit,continuousBayesFit))
globalPredictions <- ifelse(globalPredictions<0.5,0,1)
globalAccuracy <- mean(globalPredictions == testing$AcadPerformance)


knnFitProb <- ifelse(knnFit==0,1-knnResult,0+knnResult)
#ruleInductionFitProb <- ifelse(ruleInductionFit==0,1-ruleInductionResult,0+ruleInductionResult)
classTreeFitProb <- ifelse(classTreeFit==0,1-accClassTree,0+accClassTree)
svmFitProb <- ifelse(svmFit==0,1-accSVM,0+accSVM)
logRegFitProb <- ifelse(logRegFit==0,1-accLR,0+accLR)
descreteBayesFitProb <- ifelse(descreteBayesFit==0,1-accNB,0+accNB)
continuousBayesFitProb <- ifelse(continuousBayesFit==0,1-accCB,0+accCB)

globalWeightedPredictions <- rowMeans(cbind(classTreeFitProb,svmFitProb,logRegFitProb,descreteBayesFitProb,continuousBayesFitProb))
globalWeightedPredictions <- ifelse(globalWeightedPredictions<0.5,0,1)
globalWeightedAccuracy <- mean(globalWeightedPredictions == testing$AcadPerformance)






