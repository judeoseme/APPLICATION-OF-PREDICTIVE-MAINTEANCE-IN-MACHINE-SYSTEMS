# APPLICATION-OF-PREDICTIVE-MAINTEANCE-IN-MACHINE-SYSTEMS
APPLICATION OF PREDICTIVE MAINTENANCE IN MACHINE SYSTEMS 
 Introduction
Predictive maintenance is a theory which is executed to effectively manage the maintenance of machine system by predicting the failure with a well-inform data driven techniques. Maintenance in the context of field generally mean repair and replacement of equipment parts to maintain its operating condition. Maintenance is a key factor with a high implication on costs. It plays a huge impact on the organizations price quality as well as performance. Over the years, maintenance of machines has been an issue of concern to industry.  For instance, in a production company with a large volume of production as well as competition from other industries, efficient maintenance should be a priority. When a machine is not properly managed, it breaks down and this results to unplanned downtime. In this period of downtime, no production is expected to take place until a thorough maintenance is carried out before it starts functioning again In today’s world of Industrialization, it is important to consider effective maintenance management. Predictive maintenances are used in a wide range of modern industrial systems, including electronics aeronautical, automotive, and industrial machineries and more applications are being developed on daily basis. For example, Batzel and Swanson (2009) developed a framework for predicting electrical failures in aircraft power generators which can prevent unexpected failures and reduce maintenance costs. These techniques involve monitoring the condition of equipment and predicting when maintenance is required before problem occurs. Basically, there are two types of maintenance management namely: run-to-failure and preventive maintenance. To understand a predictive maintenance management program, traditional management techniques should first be considered. Industrial and process plants typically employ two types of maintenance management: run-to-failure or preventive maintenance. In run-to-failure maintenance management, you don’t fix a machine except there is a breakdown. In this method of maintenance, no money has been spent on the machine until there is a breakdown. However, it is important to note that among the various method of maintenance management this one is the most expensive due to the high cost of spare, high overtime labour cost, unavailability of products due to long period of downtime. This is a not a good maintenance approach as you are made to fix every part of the machine at once before operation can start. In preventive maintenance method, the approach is usually based on mean-time-to=failure (MTTF). In this method, organization assume that a machine will breakdown within a particular time frame. So, they ensure they carry out repairs to avoid a total breakdown of the machine. For instance, in generating plant, the maximum number of days or month an engine oil can be used in a machine is 1 month before it can be changed. Although the machine can continue running for another two months without breakdown. However, you are gradually killing the machine if you continue without changing it base on time. In this method, changing the oil regularly will not only save cost but will prolong the life of the machine system. This study is set out to propose a system that will help to predict machine failure based on parameters like temperature, machine speed etc and to add to existing knowledge of how machine failure could be prevented. It will also be effective to know when downtime will be experienced by the machine based on the historical data

Datasets
The dataset used in this study was obtained from the website below https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
For this study, a dataset of 10000 data points was selected. Each data point was identified by a unique ID ranging from 1 to 10000 and was stored as a row with 14 features in columns. The features used in the dataset are:
1.	Unique ID
2.	Product ID
3.	Type
4.	Air temperature 
5.	Process temperature
6.	Rational speed:
7.	Torque: 
8.	Tool wear
9.	Machine failure
10.	Tool wear failure (TWF)
11.	Heat dissipation failure
12.	Power failure (PWF)
13.	Overstrain failure (OSF)
14.	Random failure

Explanation and Preparation of Datasets
Preparation of data is very critical in analysis. It is the first step taken to ensure the data can be used for further processing and analysis. The dataset used in this study are made up of 14 characteristics in columns. They include the following
1.	Unique ID
2.	Product ID 
3.	Type
4.	Air temperature
5.	Process temperature
6.	Rational speed
7.	Torque
8.	Tool wear
9.	Machine failure
10.	Tool wear failure (TWF)
11.	Heat dissipation failure
12.	Power failure (PWF)
13.	Overstrain failure (OSF)
14.	Random failure
 
In this dataset, no null-values were encountered. 

Brief Description of The Algorithm 
Over the years, technology advancement has contributed so much especially to the field of machine learning. For this reason, machine learning techniques are considered to develop maintenance prediction model. In this study, decision tree and neural networks were used. 

1.Decision Tree
Decision trees are a popular method for creating classifiers which are algorithms that predict the class or value of a target variable based on input data. It is a graphical representation of data that resembles an upside-down tree. It is built in a top-down recursive manner, starting with the root node that represents the training examples. The tree consists of internal nodes which are nodes with outgoing edges and leaves, which are the other nodes. Decision trees are used for both classification and regression tasks. Decision trees algorithm are fast and easy to understand and interpret. They can handle both continuous and categorical data.    The goal of the decision tree is to create a training model that can be used to predict the class or value of the target variable. The idea of decision tree was executed using different algorithm. Which are
•	Hunt algorithm which is one of the earliest
•	CART (Classification and Regression Trees) (Breiman et al., 1984)
•	ID3 (Quinlan 1986)
•	C4.5 (Quinlan 1993)
•	Sliq
•	Sprint
Decision tree are like the KNN algorithm. It is easier to change our model from KNN algorithm to decision tree. All parts are almost the same except for some few changes. All you need to do is to make a copy of the KNN algorithm and change it to decision tree classifier. From the kernel menu, restart and run all and see the result of the decision tree. Let us determine the class feature and input feature 
Determining the class feature and input feature
In this study, machine failure is the dependent variable, while all other columns are the independent variable except product ID and type. Our major aim is to predict the dependent variable of sample based on its feature. So, the first step we will take is to slice our data into input and output. The product ID and type will not determine whether there will be a machine failure or not, so it won’t be included.
 
1.	Neural Network
They belong to the family of machine learning models which have proven huge success in solving a wide range of regression and classification problems. The human brain is being represented by a large neural network about 100 billion neurons. These neurons basically communicate through electrical signal by receiving signal from other neurons and transform the signal base on the input to other neurons. A neuron is a basic unit of the nervous system that is composed of three main parts. They are dendrite, nucleus, and axon. These three parts are interconnected through an electrochemical junctions called synapses. The dendrites are a terminal that receives electrical signal and transfer it to the nucleus. The nucleus then sends a positive output if the input satisfies it or not to the axon. The axon then receives the output which is connected to several other neurons.

The Application of Data-Mining Techniques to Selected Dataset
Data mining technique is a method applied on large volume of data to extract valuable information. It is applied on various field like education, healthcare, marketing, etc. However, in this study, one of the application of data-mining technique is that I was able to explore some important trends in the dataset. For example, using the code below, we can find out the type of distribution that has resulted to machine failure.
 
 
Explanation of the Experimental Procedure
I shall be discussing the experimental procedure used in this study on the following headings. 

1.	Determination of the class feature and input feature
We shall determine whether the target variable is a class feature or input feature. In classification, the goal of the decision tree is to create a training model that can be used to predict the class label (output or y) or value of the target variable based on its feature (input or x). In this study, the machine failure is the dependent variable while the rest columns are the independent variable except product type. Therefore, it is necessary to share our data into input (x) and output (y). the code below will be used to determine the class feature and input feature. 
x = dataset. iloc[:, [3, 4, 5, 6, 7, 9, 10, 11, 12,13 ]].values
y = dataset. iloc[:, 8].values  
 
2.	Splitting the dataset into the training set and the test set
In this article, we shall split our dataset into train and test set. The test size can be float. If it is float, the value should be between 0.0 and 1.0 to constitute the portion of the dataset included in the test split. Also, the test size can be float or integer or simply equal to none. If it is integer, it will constitute the absolute number of test samples. If otherwise, the value is set to the compliment of the train size. If the random_state equal none, we will be having a totally different test set and train test in all executions. But if a fixed value like 0 is assigned to it (random_state = 0), the test and train set will be same in different executions.  A dataset consists of features of varieties of dimension and scales. These scales can determine the modelling of a dataset. 

3.	Scaling features
A dataset is made up of varieties of different dimension and scales. Scaling different data features can affect the building up of a dataset. This is the reason why it is necessary to ensure that all features being scaled are all the same. features are used to make decision about which paths to follow through the tree. There are several methods to use feature. One common way is the use of standardization. You Standardizes each scale input variable by subtracting the mean of the feature from each value and dividing it by the standard deviation. This will transform the distribution to have a mean value of zero and a standard deviation value of one.  To standardize the train and test dataset, it is important to use the fit_transform. This method is used to enable us to scale the training data and learn the scaling parameters. method Another option is the use of normalization which involves transforming the feature value so that they are between 0 and 1. This can be done by dividing each value by the maximum value of the feature. The code below standardizes the train and test dataset. 
 



Visualization of the result
To visualize the result, we use seaborn. Using seaborn will offer us the opportunity to use various plot types use for statistical data exploration. Such as line plots, scatter plots, bar plots, etc.  Seaborn library will show statistical information about different columns of dataset. Executing the code below, we give you the type distribution 

 
From the above visualization, Type L has got the highest count of 5800 responsible for machine failure. Followed by Type M with 3000 count and lastly type H with the least number of counts. 

Results analysis and discussion

Evaluating the Model                                                                                                                 
To train the model, all parts of the decision tree algorithm is like the KNN algorithm. All you need to do is to replace the code of KNN (KNeighborsClassifier) to the Decision tree (Decision tree classifier)
 
Decisiontreeclassifier
From the kernel menu, select ‘Restart and Run All’ and see the result of the decision tree. 
To evaluate the model, we ensure that the model is trained. Once that is done. We make use of the predict function on our model to make predictions on our test data.
 
Y_pred shows the predicted label for test dataset. Now, compare the predicted value with the real value of the labels in test dataset.
 
To evaluate the performance of the model, execute the following code
 
To generates the confusion matrix, we use the generated predictions (y_pred) and the true class labels for the test data (y_test). This classification report will show different metrics such as precision, recall, fi-score and support. 
 
The confusion matrix can be visualize using the seaborn heatmap.
 
Conclusions
In summary, machine failure can have a huge consequence in a production and manufacturing industries. These include downtime in productions, financial losses, loss of reliability in customer relationship. Having a high effective and monitoring systems in place to checkmate and minimize the risk of machine failure during production is a necessity for the growth and development of the industry.  In a large volume production, maintenance is a necessary factor to be considered. The performance of a machine system is not dependent on the design or layout but on the effective maintenance of the machine during their operational lifetime.

References 
Batzel, Todd D and David C Swanson (2009). “Prognostic health management of aircraft power generators”. In: IEEE Transactions on Aerospace and Electronic Systems 45.2, pp. 473–482.
Breiman, L., Friedman, J.H., Olshen, R.A., & Stone, C.J. (1984). Classification and Regression Trees.
Quinlan, J.R. (1986) Induction of Decision Trees. Machine Learning, 1, 81-106.

Quinlan, R. (1993) C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers, San Mateo.
