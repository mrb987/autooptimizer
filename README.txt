AutoOptimizer provides tools to automatically optimize machine learning model for a dataset with very little user intervention.

It refers to techniques that allow semi-sophisticated machine learning practitioners and non-experts 
to discover a good predictive model pipeline for their machine learning algorithm task quickly,
with very little intervention other than providing a dataset.


#Prerequisites:

jupyterlab(contains all sub packages except mlxtend) or: {sklearn,matplotlib,mlxtend,numpy}	


#Usage:


*Optimize scikit learn supervised and unsupervised learning models using python.


{DBSCAN, KMeans, MeanShift,  LogisticRegression, KNeighborsClassifier, SupportVectorClassifier, DecisionTree}


*Metrics for Your Regression Model


*Clear data by removing outliers



>#Running auto optimizer:


>>from autooptimizer.cluster import dbscan, meanshift, kmeans

 
>>from autooptimizer.neighbors import kneighborsclassifier


>>from autooptimizer.tree import decisiontreeclassifier


>>from autooptimizer.svm import svc


>>from autooptimizer.linear_model import logisticregression


>>dbscan(x)


>>kmeans(x)


>>meanshift(x)


>>logisticregression(x,y)


>>kneighborsclassifier(x,y)


>>svc(x,y)


>>decisiontreeclassifier(x,y)


'x' should be your independent variable or feature's values and 'y' is target variable or dependent variable.
The output of the program is the maximum possible accuracy with the appropriate parameters to use in model.

>#Evaluation Metrics for Your Regression Model

{root_mean_squared_error, root_mean_squared_log_error, root_mean_squared_precentage_error,
symmetric_mean_absolute_precentage_error, mean_bias_error, relative_squared_error, root_relative_squared_error
relative_absolute_error, median_absolute_percentage_error, mean_absolute_percentage_error}

>#Running for example


>>from autooptimizer.metrics import root_mean_squared_error


>>root_mean_squared_error(true, predicted)


>#Running outlier remover


>>from autooptimizer.process import outlier_remover


>>outlier_remover(data)


>>plot_outlier_removal(data) #with plot charts for more details



#Contact and Contributing:
Please share your good ideas with us. 
Simply letting us know how we can improve the programm to serve you better.
Thanks for contributing with the programm.

>>https://github.com/mrb987/autooptimizer

>>https://www.linkedin.com/in/mohammad-reza-barghi-6337a061/
>>info@genesiscube.ir
