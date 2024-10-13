#Select columns for general g and the 11 tasks
#Retain subjects that meet inclusion criteria for resting fMRI
set.seed(5)
library(dplyr)
cognition_data <- `ABCD_rest_share.(4)`
cognition_data[cognition_data == 'NaN'] <- NA
cognition_data <- filter(cognition_data,Include_rest==TRUE)[,c(1,13,14,18,seq(110,120))]
cognition_data <- na.omit(cognition_data)
cognition_data <- cognition_data[, -c(2,3)]

#Randomly sample 80 percent of subjects for training, 10 percent for validation, 
# and 10 percent for testing
train <- sample(cognition_data[,1],0.8*length(cognition_data[,1]))
validate <- sample(setdiff(cognition_data[,1],train),0.1*length(cognition_data[,1]))
test<- setdiff(setdiff(cognition_data[,1],train),validate)

#Center fMRI data before PCA
rownames(data_matrix) <- cognition_data$subjectkey
training_set <- data_matrix[train,]
means <- apply(training_set,2,mean)
validation_set <- data_matrix[validate,]
validation_set <- scale(validation_set,center=means,scale=FALSE)
testing_set <- data_matrix[test,]
testing_set <- scale(testing_set,center=means,scale=FALSE) 

#Perform PCA
#Retain first 500 PCs
PC <- prcomp(training_set,rank=500)

# Following simulations, scale features by sigma * sqrt(n) for use in MTL
# Scaling ensures all features are penalized equally. The penalty is 
# inversely-proportional to the sum of coefficients, so low variance features
# might otherwise be penalized less than high variance ones
PC_scalings <- apply(PC$x,2,sd)
training_data <- scale(PC$x)
validation_data <- scale(validation_set%*%PC$rotation,center=rep(0,dim(PC$x)[2]),scale=PC_scalings)
testing_data <- scale(testing_set%*%PC$rotation,center=rep(0,dim(PC$x)[2]),scale=PC_scalings)

training_data <- training_data/sqrt(dim(training_data)[1])
validation_data <- validation_data/sqrt(dim(training_data)[1])
testing_data <- testing_data/sqrt(dim(training_data)[1])

#Pre-process cognition data and combine with PC data
subjects_train <- t(sapply(rownames(training_data),function(x) filter(cognition_data,subjectkey==x)[,2:13]))
subjects_validate <- t(sapply(rownames(validation_data),function(x) filter(cognition_data,subjectkey==x)[,2:13]))
subjects_test <- t(sapply(rownames(testing_data),function(x) filter(cognition_data,subjectkey==x)[,2:13]))
subjects_train <- apply(subjects_train,2,unlist)
subjects_validate <- apply(subjects_validate,2,unlist)
subjects_test <- apply(subjects_test,2,unlist)
means <- apply(subjects_train,2,mean)
subjects_train <- scale(subjects_train,center=TRUE,scale=FALSE)
subjects_validate <- scale(subjects_validate,center=means,scale=FALSE)
subjects_test <- scale(subjects_test,center=means,scale=FALSE)

train <- cbind(training_data,subjects_train)
validate <- cbind(validation_data,subjects_validate)
test <- cbind(testing_data,subjects_test)

write.csv(train,"train.csv",row.names=F)
write.csv(validate,"validate.csv",row.names=F)
write.csv(test,"test.csv",row.names=F)
write.csv(PC$rotation,"V.csv",row.names=F)
write.csv(PC_scalings,"lambda.csv",row.names=F)
