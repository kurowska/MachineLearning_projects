Summary
-------

With the use of devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity. These devices are part of the quantified self movement – people who take measurements about themselves regularly to improve their health and/or to find patterns in their behavior. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict how well they do the excercises. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information about this data is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (section on the Weight Lifting Exercise Dataset). The data for this project also come from this site.

Data Preprocessing
------------------

``` r
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "pml-training.csv"
testFile  <- "pml-testing.csv"

if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}

training <- read.csv(file = trainFile, header = TRUE, sep = ",")
testing  <- read.csv(file = testFile, header = TRUE, sep = ",")

dim(training); dim(testing)
```

    ## [1] 19622   160

    ## [1]  20 160

The training file contains 19622 observations of 160 variables, while the test file has 20 observations of 160 variables. Closer look at training data reveals that there are a lot of NA values in a number of columns. We check now how many rows in each column are affected, to see if it's better to remove observations, or the variables.

``` r
training_na <- training[, colSums(is.na(training)) != 0]
na_count <-sapply(training_na, function(na) sum(is.na(na)))
na_count <- data.frame(na_count)
```

The 67 columns that have NA values contain almost only NA values (this check is not shown here). In each case 19216 observations (98% of the training sample) are NA, therefore these variables will be deleted. Additionally, first 7 colums of data are not useful for predictions, they will be removed as well. Out of remaining variables there are still a few which are empty, they will also be removed. This leaves the training sample with 53 predictors.

``` r
trainingData <- training[, colSums(is.na(training)) == 0]#removing variables with NA
trainingData <- trainingData[ ,-c(1:7)]#removing irrelevant/empty variables
classe <- trainingData$classe
trainingData <- trainingData[, sapply(trainingData, is.numeric)]
trainingData$classe <- classe

testingData <- testing[, colSums(is.na(testing)) == 0]
testingData <- testingData[ ,-c(1:7)]
testingData <- testingData[, sapply(testingData, is.numeric)]

dim(trainingData); dim(testingData)
```

    ## [1] 19622    53

    ## [1] 20 53

The training set will be divided now into train and validation samples, to help model selection.

``` r
inTrain <- createDataPartition(y = trainingData$classe, p = 0.7, list = FALSE)
trainSub <- trainingData[inTrain,]
validSub <- trainingData[-inTrain,]
```

Prediction model evaluation
---------------------------

We consider here classification trees (`rpart`), random forests (`rf`), and boosted regression (`gbm`) models. The cross validation is done by `cv` method in the trainControl, for which 5-fold method was chosen.

#### Classification trees (rpart)

``` r
set.seed(123)
trCtrl <- trainControl(method = 'cv', number = 5, summaryFunction = defaultSummary)
grid_rpart <- expand.grid(cp = seq(0, 0.05, 0.005))#complexity parameter
fit_rpart <- train(classe~., data = trainSub, method = 'rpart', trControl = trCtrl,tuneGrid = grid_rpart)
```

    ## Loading required package: rpart

``` r
print(fit_rpart)
```

    ## CART 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10990, 10989, 10990, 10989, 10990 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp     Accuracy   Kappa    
    ##   0.000  0.9221094  0.9014704
    ##   0.005  0.7974106  0.7436579
    ##   0.010  0.7321101  0.6598938
    ##   0.015  0.6774411  0.5910279
    ##   0.020  0.6447558  0.5505470
    ##   0.025  0.5701381  0.4491893
    ##   0.030  0.5344706  0.3994950
    ##   0.035  0.5116144  0.3701184
    ##   0.040  0.5021495  0.3546366
    ##   0.045  0.4641444  0.2952941
    ##   0.050  0.4394630  0.2482774
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.

``` r
plot(fit_rpart)
```

![](MachineLearningProject_files/figure-markdown_github/rpart-1.png)

``` r
###
folds = createFolds(trainSub$classe, k = 5)
error = rep(0, 5)
for (k in 1:5) {
    test = trainSub[unlist(folds[k]), ]
    pred = predict(fit_rpart, test)
    accuracy = sum(as.numeric(pred) == as.numeric(test$classe)) / length(pred)
    error_k = 1 - accuracy
    error[k] = error_k
}
error_rpart = round(mean(error)*100, digits = 2)
```

The out of sample error estimations for classification tree is **3.87%**. This is the weakest model of all considered.

#### Boosted regression (gbm)

``` r
set.seed(456)
trCtrl <- trainControl(method = 'cv', number = 5, summaryFunction = defaultSummary)
grid_gbm <- expand.grid( n.trees = seq(50, 200, 5), interaction.depth = c(10), shrinkage = c(0.1), n.minobsinnode = 20)
fit_gbm <- train(classe~., data = trainSub, method = 'gbm', trControl = trCtrl, tuneGrid = grid_gbm, verbose = FALSE)
print(fit_gbm)
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10988, 10991, 10990, 10989, 10990 
    ## Resampling results across tuning parameters:
    ## 
    ##   n.trees  Accuracy   Kappa    
    ##    50      0.9724099  0.9650937
    ##    55      0.9744484  0.9676734
    ##    60      0.9772874  0.9712656
    ##    65      0.9793986  0.9739370
    ##    70      0.9813641  0.9764245
    ##    75      0.9826744  0.9780828
    ##    80      0.9842034  0.9800177
    ##    85      0.9851496  0.9812154
    ##    90      0.9859505  0.9822281
    ##    95      0.9866056  0.9830571
    ##   100      0.9871149  0.9837016
    ##   105      0.9874062  0.9840701
    ##   110      0.9877701  0.9845304
    ##   115      0.9879158  0.9847147
    ##   120      0.9886435  0.9856356
    ##   125      0.9885709  0.9855438
    ##   130      0.9890803  0.9861880
    ##   135      0.9893715  0.9865562
    ##   140      0.9894442  0.9866484
    ##   145      0.9900995  0.9874773
    ##   150      0.9902448  0.9876612
    ##   155      0.9910455  0.9886739
    ##   160      0.9909728  0.9885821
    ##   165      0.9913369  0.9890426
    ##   170      0.9911912  0.9888582
    ##   175      0.9914824  0.9892267
    ##   180      0.9917008  0.9895030
    ##   185      0.9918465  0.9896874
    ##   190      0.9917736  0.9895948
    ##   195      0.9919920  0.9898710
    ##   200      0.9921376  0.9900552
    ## 
    ## Tuning parameter 'interaction.depth' was held constant at a value of
    ##  10
    ## Tuning parameter 'shrinkage' was held constant at a value of
    ##  0.1
    ## Tuning parameter 'n.minobsinnode' was held constant at a value of 20
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final values used for the model were n.trees = 200,
    ##  interaction.depth = 10, shrinkage = 0.1 and n.minobsinnode = 20.

``` r
plot(fit_gbm)
```

![](MachineLearningProject_files/figure-markdown_github/gbm-1.png)

``` r
###
folds = createFolds(trainSub$classe, k = 5)
error = rep(0, 5)
for (k in 1:5) {
    test = trainSub[unlist(folds[k]), ]
    pred = predict(fit_gbm, test)
    accuracy = sum(as.numeric(pred) == as.numeric(test$classe)) / length(pred)
    error_k = 1 - accuracy
    error[k] = error_k
}
error_gbm = round(mean(error)*100, digits = 7)
```

The out of sample error estimations for boosted regression is **0%**. This model has much higher accuracy with a very small (near zero) error .

#### Random forests (rf)

``` r
set.seed(789)
trCtl <- trainControl(method = 'cv', number = 5, summaryFunction = defaultSummary)
grid_rf <- expand.grid( mtry = seq(2, 40, 5))
fit_rf <- train(classe~., data = trainSub, method = 'rf', trControl = trCtl, tuneGrid = grid_rf, verbose = FALSE)
print(fit_rf)
```

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10989, 10991, 10989, 10989, 10990 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9906094  0.9881209
    ##    7    0.9922836  0.9902392
    ##   12    0.9921382  0.9900557
    ##   17    0.9915558  0.9893185
    ##   22    0.9912646  0.9889500
    ##   27    0.9903911  0.9878452
    ##   32    0.9897362  0.9870159
    ##   37    0.9888625  0.9859113
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 7.

``` r
plot(fit_rf)
```

![](MachineLearningProject_files/figure-markdown_github/rf-1.png)

``` r
###
folds = createFolds(trainSub$classe, k = 5)
error = rep(0, 5)
for (k in 1:5) {
    test = trainSub[unlist(folds[k]), ]
    pred = predict(fit_rf, test)
    accuracy = sum(as.numeric(pred) == as.numeric(test$classe)) / length(pred)
    error_k = 1 - accuracy
    error[k] = error_k
}
error_rf = round(mean(error)*100, digits = 7)
```

The out of sample error estimations for random forests is **0%**. This model also has high accuracy and low (near zero) error. Let's compare boosted regression and random forests models, as they both perform better than classification tree (rpart).

``` r
results <- resamples(list(GBM = fit_gbm, RandomForest = fit_rf))

# summarize the distributions
summary(results)
```

    ## 
    ## Call:
    ## summary.resamples(object = results)
    ## 
    ## Models: GBM, RandomForest 
    ## Number of resamples: 5 
    ## 
    ## Accuracy 
    ##                Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
    ## GBM          0.9902  0.9905 0.9913 0.9921  0.9938 0.9949    0
    ## RandomForest 0.9902  0.9916 0.9927 0.9923  0.9934 0.9934    0
    ## 
    ## Kappa 
    ##                Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
    ## GBM          0.9876  0.9880 0.9889 0.9901  0.9922 0.9936    0
    ## RandomForest 0.9876  0.9894 0.9908 0.9902  0.9917 0.9917    0

``` r
bwplot(results)
```

![](MachineLearningProject_files/figure-markdown_github/plots-1.png)

Model choice and prediction
---------------------------

Model comparison suggests that among checked models, boosted regression perform as well as random forests, with very slight differences. Boosted regression is a bit faster (on the machine this report is done with parameters described above) than random forests which is reason good enough to choose it as the final model.

One can check the model accuracy on full (not folded) trainSub sample and check out of sample error on the validSub sample:

``` r
set.seed(333)
grid <- expand.grid(fit_gbm$bestTune)
model <- train(classe~., data = trainSub, method = 'gbm', tuneGrid = grid, verbose = FALSE)

prediction <- predict(model, validSub)
error = 1- sum(as.numeric(prediction) == as.numeric(validSub$classe)) / length(prediction)
```

The training model accuracy is **99.49%** and the out of sample error on validSub data set for this model is **0.51%**

Test sample prediction
----------------------

The final test sample prediction gives the following results:

``` r
set.seed(3234)
final_prediction <- predict(model, testingData, n.trees=200)
final_prediction
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

The results will be written to files:

``` r
predictions <- as.character(final_prediction)

write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        file_name <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=file_name, quote=F, row.names=F, col.names=F)
    }
}

# create prediction files to submit
write_files(predictions)
```
