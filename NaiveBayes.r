####################################################################
# Example 3: The Naïve Bayes classifier
####################################################################

library (e1071)

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
data$shares <- factor(data$shares, labels = c("unpopular","popular"))

summary(data$shares)

# Number of instances
N <- nrow(data)

# Indexes from the rows that are part of the learning set
learn <- sample(1:N, round(2*N/3))

nlearn <- length(learn)
ntest <- N - nlearn

# Train the model
model <- naiveBayes(shares ~ ., data = data[learn,])

# compute now the apparent error for the learn set
pred <- predict(model, data[learn,-60])

# form and display confusion matrix & overall error
tab <- table(pred, data[learn,]$shares) 
tab
1 - sum(tab[row(tab)==col(tab)])/sum(tab)

# compute the test (prediction) error
pred <- predict(model, newdata=data[-learn,-60])

# form and display confusion matrix & overall error
tab <- table(pred, data[-learn,]$shares) 
tab
1 - sum(tab[row(tab)==col(tab)])/sum(tab)

for (i in seq(0,3,0.25)) {
  model <- naiveBayes(shares ~ ., data = data[learn,], laplace = i)
   
  # compute the test (prediction) error
  pred <- predict(model, newdata=data[-learn,-60])
  
  print(paste("Laplace parameter equals",toString(i)))
  # form and display confusion matrix & overall error
  tab <- table(pred, data[-learn,]$shares) 
  print(tab)
  print(1 - sum(tab[row(tab)==col(tab)])/sum(tab)) 
}
