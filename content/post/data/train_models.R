data<-unzip("data.zip")

#train models

traindata_smote<-data.table::fread(data[1])
testing_data<-data.table::fread(data[3])
validation_data<-data.table::fread(data[4])


library(mlrHyperopt)
library(mlr)
library(parallelMap)
task <- makeClassifTask(id = "telemarketing", data = as.data.frame(traindata_smote), target="y")

parallelStartMulticore(parallel::detectCores())

random_forest_opt <- hyperopt(task, learner = "classif.ranger")

random_forest_opt<-train(makeLearner(cl = "classif.ranger", id="randomf", num.trees = 128, mtry=22, min.node.size=10, num.threads=4), task)

test<-ranger::ranger(dependent.variable.name = "y", traindata_smote, num.trees = 128, mtry=22, min.node.size=10, num.threads=4)


svm_opt <- hyperopt(task, learner = "classif.svm")

nnet_opt <- hyperopt(task, learner = "classif.nnet")

searchspace_knn <- downloadParConfigs(learner.name = "kknn")
knn_opt <- hyperopt(task, learner = "classif.kknn", par.config = searchspace_knn[[1]])


nBayes <- mlr::train(task, learner = "classif.naiveBayes")


#xgboost training needs binary target

traindata_smote$y2<-as.numeric(as.factor(traindata_smote$y))
traindata_smote[traindata_smote$y2<2,"y2"]<-F
traindata_smote[traindata_smote$y2==2,"y2"]<-T


traindata_smote$y2<-as.numeric(traindata_smote$y2)
traindata_smote$y<-as.numeric(traindata_smote$y2)
traindata_smote<-dplyr::select(traindata_smote, -y2)

xgb_task <- makeClassifTask(id = "telemarketing", data = as.data.frame(traindata_smote), target="y")

autoxgboost_opt<-autoxgboost::autoxgboost(task = task)


#sampled data for svm task

sample_df<-dplyr::sample_n(traindata_smote,10000)
svm_task <- makeClassifTask(id = "telemarketing", data = as.data.frame(sample_df), target="y")

svm_opt <- hyperopt(svm_task, learner = "classif.svm")



parallelStop()

parallelStartMulticore(parallel::detectCores())
res = mlr::benchmark(learners = c(nBayes, random_forest_opt, svm_opt, searchspace_xgb, nnet_opt, searchspace_knn), tasks = task, resamplings = cv10)
parallelStop()

plotBMRBoxplots(res) 
