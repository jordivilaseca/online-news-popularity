library(randomForest)

calcule.metrics <- function(answ, pred) 
{
  valid.error <- sum(pred != answ)/length(pred)
  conf_matrix <- table(answ, pred)
  accur <- sum(diag(conf_matrix))/sum(conf_matrix)
  
  metrics <- matrix(nrow=2, ncol=3)
  colnames(metrics) = c("precision", "recall", "F score")
  rownames(metrics) = c("C0","C1")
  
  metrics["C0", "precision"] <- conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[2,1])
  metrics["C1", "precision"] <- conf_matrix[2,2] / (conf_matrix[1,2] + conf_matrix[2,2])
  metrics["C0", "recall"] <- conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,2])
  metrics["C1", "recall"] <- conf_matrix[2,2] / (conf_matrix[2,1] + conf_matrix[2,2])
  metrics["C0", "F score"] <- 2*(metrics["C0", "precision"]*metrics["C0", "recall"])/(metrics["C0", "precision"]+metrics["C0", "recall"])
  metrics["C1", "F score"] <- 2*(metrics["C1", "precision"]*metrics["C1", "recall"])/(metrics["C1", "precision"]+metrics["C1", "recall"])
  
  return(list(accur=accur,conf_matrix=conf_matrix,pred=pred, metrics=metrics))
}

train.rf <- function (data, ntrees, folds,  k)
{
  valid.error <- rep(0,k)
  conf_matrix <- vector(mode = "list", length = k)
  pred.complete <- rep(-1, length(folds))
  
  for (i in 1:k) 
  {  
    train <- data[folds!=i,] # for building the model (training)
    valid <- data[folds==i,] # for prediction (validation)
    
    model <- randomForest(shares ~ ., data=train, ntree=ntrees, proximity=FALSE)
    
    x_valid <- valid[,-60]
    t_valid <- valid[,60]
    
    pred.partial <- predict(model, x_valid, type="class")
    pred.array <- as.integer(array(pred.partial))
    pred.complete[folds==i] <- pred.array
    
    err <- sum(pred.array != t_valid)/length(t_valid)
    print(paste("ntrees =",as.character(ntrees), ", i =",as.character(i), ", error =", as.character(err)))
    
  }
  answ = as.integer(array(data$shares))
  
  results <- calcule.metrics(answ, pred.complete)
  
  return(results)
}

# Read data from csv
data <- read.csv(file="./OnlineNewsPopularity.csv", header=TRUE, sep=",")
data$url = NULL

summary(data)

# Threshold of shares used to classify an article as popular or unpopular. Using 1400
# gives a balanced model
THRESHOLD <- 1400

# Transform the shares variable from a continuous to a discrete one using the threshold
# and mark it as a factor.
data$shares[data$shares < THRESHOLD] = 0
data$shares[data$shares >= THRESHOLD] = 1
data$shares <- factor(data$shares)

# Number of instances
N <- nrow(data)

# Indexes from the rows that are part of the learning set
learn <- sample(1:N, round(2*N/3))

nlearn <- length(learn)
ntest <- N - nlearn

## We define now a convenience function (the harmonic mean), to compute the F1 accuracy:
harm <- function (a,b) { 2/(1/a+1/b) }

# K fold cross validation
k <- 10
folds <- sample(rep(1:k, length=nlearn), nlearn, replace=FALSE)

# Test different tree sizes
ntrees <- c(10,20,50,100,150,200,250,300,350,400,450,500)
results = vector(mode="list",length=length(ntrees))
for (i in seq(1,length(ntrees))) {
  results[[i]] <- train.rf(data[learn,], ntrees[i], folds, k)
  print(paste("Accuracity for ntrees equals", as.character(ntrees[i]), "is", as.character(results[[i]]$accur)))
  save.image("RF.RData")
}

# See which values of ntrees are more promising
for (i in seq(1,length(results))) {
  print(as.character(ntrees[i])) 
  print(results[[i]]$metrics)
}

# Best values are between 250 and 300 trees. We can make a finer search
ntrees2 <- c(250,260,270,280,290,300)
results2 = vector(mode="list",length=length(ntrees2))
for (i in seq(1,length(ntrees2))) {
  results2[[i]] <- train.rf(data[learn,], ntrees2[i], folds, k)
  print(paste("Accuracity for ntrees equals", as.character(ntrees2[i]), "is", as.character(results2[[i]]$accur)))
  save.image("RF.RData")
}

# See which values of ntrees2 are more promising
for (i in seq(1,length(ntrees2))) {
  print(as.character(ntrees2[i])) 
  print(results2[[i]]$metrics)
}
