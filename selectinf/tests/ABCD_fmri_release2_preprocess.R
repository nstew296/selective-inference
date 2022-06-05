#Select columns for general g and the 11 tasks
#Retain subjects that meet inclusion criteria for resting fMRI
set.seed(5)
library(dplyr)
cognition_data <- `ABCD_rest_share.(4)`
subjects <- filter(cognition_data,Include_rest==TRUE)[,c(1,18,seq(110,120))]
subjects <- na.omit(subjects)

#Select columns for site ID and psychopathology factor 
#Exclude one subject not in the intersection
psychopathology_data <- ABCD_lavaan_pfactor_loso
joint_subjects <- filter(psychopathology_data,subjectkey%in%subjects$subjectkey)[,c(1,2,69)]
data_matrix_joint <- data_matrix[-which(subjects$subjectkey==setdiff(subjects$subjectkey,joint_subjects$subjectkey)),]

#Randomly sample 80 percent of subjects for training, 10 percent for validation, and 10 percent for testing
train <- sample(joint_subjects[,1],0.8*length(joint_subjects[,1]))
validate <- sample(setdiff(joint_subjects[,1],train),0.1*length(joint_subjects[,1]))
test<- setdiff(setdiff(joint_subjects[,1],train),validate)

#Center fMRI data before PCA
rownames(data_matrix_joint) <- joint_subjects$subjectkey
training_set <- data_matrix_joint[train,]
means <- apply(training_set,2,mean)
validation_set <- data_matrix_joint[validate,]
validation_set <- scale(validation_set,center=means,scale=FALSE)
testing_set <- data_matrix_joint[test,]
testing_set <- scale(testing_set,center=means,scale=FALSE) 

#Perform PCA and standardize PCs
PC <- prcomp(training_set,rank=1000)
PC_scalings <- apply(PC$x,2,sd)
training_data <- scale(PC$x)
validation_data <- scale(validation_set%*%PC$rotation,center=rep(0,1000),scale=PC_scalings)
testing_data <- scale(testing_set%*%PC$rotation,center=rep(0,1000),scale=PC_scalings)

training_data <- training_data/sqrt(dim(training_data)[1])
validation_data <- validation_data/sqrt(dim(training_data)[1])
testing_data <- testing_data/sqrt(dim(training_data)[1])

#Pre-process cognition data and combine with PC data
cognition_train <- t(sapply(rownames(training_data),function(x) filter(subjects,subjectkey==x)[,2]))
cognition_validate <- t(sapply(rownames(validation_data),function(x) filter(subjects,subjectkey==x)[,2]))
cognition_test <- t(sapply(rownames(testing_data),function(x) filter(subjects,subjectkey==x)[,2]))
cognition_train <- apply(cognition_train,2,unlist)
cognition_validate <- apply(cognition_validate,2,unlist)
cognition_test <- apply(cognition_test,2,unlist)
mean_train <- mean(cognition_train)
sd_train <- sd(cognition_train)
cognition_train <- scale(cognition_train,center=TRUE,scale=TRUE)
conition_validate <- scale(cognition_validate,center=mean_train,scale=sd_train)
cognition_test <- scale(cognition_test,center=mean_train,scale=sd_train)

write.csv(cognition_train,"train.csv",row.names=F)
write.csv(cognition_validate,"validate.csv",row.names=F)
write.csv(cognition_test,"test.csv",row.names=F)
write.csv(PC$rotation,"V.csv",row.names=F)
write.csv(PC_scalings,"lambda.csv",row.names=F)

#psychopathology_train <- sapply(rownames(training_data),function(x) filter(joint_subjects,subjectkey==x)[,3])
#psychopathology_validation <- sapply(rownames(validation_data),function(x) filter(joint_subjects,subjectkey==x)[,3])
#psychopathology_test <- sapply(rownames(testing_data),function(x) filter(joint_subjects,subjectkey==x)[,3])
#mean_train <- mean(psychopathology_train)
#sd_train <- sd(psychopathology_train)
#psychopathology_train <- scale(psychopathology_train,center=TRUE,scale=TRUE)
#psychopathology_validation <- scale(psychopathology_validation,center=mean_train,scale=sd_train)
#psychopathology_test <- scale(psychopathology_test,center=mean_train,scale=sd_train)

#train <- cbind(training_data,cognition_train,psychopathology_train)
#validate <- cbind(validation_data,cognition_validate,psychopathology_validation)
#test <- cbind(testing_data,cognition_test,psychopathology_test)

#Output training, validation, and testing data as well as PC loadings and sd's
#write.csv(train,"train.csv",row.names=F)
#write.csv(validate,"validate.csv",row.names=F)
#write.csv(test,"test.csv",row.names=F)
#write.csv(PC$rotation,"V.csv",row.names=F)
#write.csv(PC_scalings,"lambda.csv",row.names=F)


#Pre-process cognition data and combine with PC data
#subjects_train <- t(sapply(rownames(training_data),function(x) filter(subjects,subjectkey==x)[,2:13]))
#subjects_validate <- t(sapply(rownames(validation_data),function(x) filter(subjects,subjectkey==x)[,2:13]))
#subjects_test <- t(sapply(rownames(testing_data),function(x) filter(subjects,subjectkey==x)[,2:13]))
#subjects_train <- apply(subjects_train,2,unlist)
#subjects_validate <- apply(subjects_validate,2,unlist)
#subjects_test <- apply(subjects_test,2,unlist)
#means <- apply(subjects_train,2,mean)
#subjects_train <- scale(subjects_train,center=TRUE,scale=FALSE)
#subjects_validate <- scale(subjects_validate,center=means,scale=FALSE)
#subjects_test <- scale(subjects_test,center=means,scale=FALSE)

#train <- cbind(training_data,subjects_train)
#validate <- cbind(validation_data,subjects_validate)
#test <- cbind(testing_data,subjects_test)

#Pre-process psychopathology data and combine with fMRI data 
#psychopathology_train <- sapply(rownames(training_data),function(x) filter(joint_subjects,subjectkey==x)[,3])
#psychopathology_validation <- sapply(rownames(validation_data),function(x) filter(joint_subjects,subjectkey==x)[,3])
#psychopathology_test <- sapply(rownames(testing_data),function(x) filter(joint_subjects,subjectkey==x)[,3])
#means <- mean(psychopathology_train)
#psychopathology_train <- psychopathology_train - means
#psychopathology_validation <- psychopathology_validation - means
#psychopathology_test <- psychopathology_test - means

#train <- cbind(training_data,psychopathology_train)
#validate <- cbind(validation_data,psychopathology_validation)
#test <- cbind(testing_data,psychopathology_test)

#Output training, validation, and testing data as well as PC loadings and sd's
#write.csv(train,"train.csv",row.names=F)
#write.csv(validate,"validate.csv",row.names=F)
#write.csv(test,"test.csv",row.names=F)
#write.csv(PC$rotation,"V.csv",row.names=F)
#write.csv(PC_scalings,"lambda.csv",row.names=F)
