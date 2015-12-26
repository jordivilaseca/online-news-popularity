####################################################################
# Example 3: The Naïve Bayes classifier
####################################################################

library (e1071)

# Read data from csv
data <- read.csv(file="./OnlineNewsPopularity/OnlineNewsPopularity.csv", header=TRUE, sep=",")

summary(data)

# Threshold of shares used to classify an article as popular or unpopular. Using 1400
# gives a balanced model
THRESHOLD <- 1400

# Transform the shares variable from a continuous to a discrete one using the threshold
# and mark it as a factor.
data$shares[data$shares <= THRESHOLD] = 0
data$shares[data$shares > THRESHOLD] = 1
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

# Predict, get the veredict for the first 20 rows
predict(model, data[1:20,-61]) 

# Predict, get the probabilities for the first 20 rows
predict(model, data[1:20,-61], type = "raw")

# compute now the apparent error for the learn set
pred <- predict(model, data[learn,-61])

# form and display confusion matrix & overall error
tab <- table(pred, data[learn,]$shares) 
tab
1 - sum(tab[row(tab)==col(tab)])/sum(tab)

# compute the test (prediction) error
pred <- predict(model, newdata=data[-learn,-61])

# form and display confusion matrix & overall error
tab <- table(pred, data[-learn,]$shares) 
tab
1 - sum(tab[row(tab)==col(tab)])/sum(tab)

model <- naiveBayes(shares ~ ., data = data[learn,], laplace = 1)

# compute the test (prediction) error
pred <- predict(model, newdata=data[-learn,-61])

# form and display confusion matrix & overall error
tab <- table(pred, data[-learn,]$shares) 
tab
1 - sum(tab[row(tab)==col(tab)])/sum(tab)