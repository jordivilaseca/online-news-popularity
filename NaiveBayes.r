####################################################################
# Example 3: The Naïve Bayes classifier
####################################################################

library (e1071)

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

train.naiveBayes.kCV <- function (data, folds,  k)
{
  valid.error <- rep(0,k)
  conf_matrix <- vector(mode = "list", length = k)
  pred.complete <- rep(-1, length(folds))
  
  for (i in 1:k) 
  {  
    train <- data[folds!=i,] # for building the model (training)
    valid <- data[folds==i,] # for prediction (validation)
    
    model <- naiveBayes(shares ~ ., data=train)
    
    x_valid <- valid[,-60]
    t_valid <- valid[,60]

    pred.partial <- predict(model, x_valid, type="class")
    pred.array <- as.integer(array(pred.partial))
    pred.complete[folds==i] <- pred.array

    err <- sum(pred.array != t_valid)/length(t_valid)
    print(paste("i =",as.character(i), ", error =", as.character(err)))
    
  }
  answ = as.integer(array(data$shares))

  results <- calcule.metrics(answ, pred.complete)
  
  return(results)
}

# Read data from csv
data <- read.csv(file="./OnlineNewsPopularity.csv", header=TRUE, sep=",")

# Remove url from dataset, it does not give the model any information.
data$url <- NULL

summary(data)

# Threshold of shares used to classify an article as popular or unpopular. Using 1400
# gives a balanced model
THRESHOLD <- 1400

# Transform the shares variable from a continuous to a discrete one using the threshold
# and mark it as a factor.
data$shares[data$shares < THRESHOLD] = 0
data$shares[data$shares >= THRESHOLD] = 1
data$shares <- factor(data$shares)

summary(data$shares)

# Number of instances
N <- nrow(data)

# Indexes from the rows that are part of the learning set
learn <- sample(1:N, round(2*N/3))

nlearn <- length(learn)
ntest <- N - nlearn

k <- 10
folds <- sample(rep(1:k, length=nlearn), nlearn, replace=FALSE)

results <- train.naiveBayes.kCV(data[learn,], folds, k)
print(paste("Accuracity is", as.character(results$accur)))
save.image("NB.RData")

