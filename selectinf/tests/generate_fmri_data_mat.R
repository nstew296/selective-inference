subject_data <- `ABCD_rest_share.(4)`
subjects <- filter(subject_data,Include_rest==TRUE)
subjects <- filter(subjects,Include_rest==TRUE)[,c(1,seq(110,120))]
subjects <- na.omit(subjects)
subjects$subjectkey <- gsub("_","",subjects$subjectkey)
data_matrix <- matrix(NA,0,87153)
for (id in subjects$subjectkey){
data <- read.table(paste(id,".txt",sep=""))
mean_correlations <- data[upper.tri(data, diag = FALSE)]
data_matrix <- rbind(data_matrix,mean_correlations)
}