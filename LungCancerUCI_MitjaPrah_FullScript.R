##########################
#DATA COLLECTION AND PROCESSING
##########################

#Install the required R packages, if applicable
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(farff)) install.packages("farff", repos = "http://cran.us.r-project.org")
if(!require(mltools)) install.packages("mltools", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(DMwR)) install.packages("DMwR", repos = "http://cran.us.r-project.org")
if(!require(broom)) install.packages("broom", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(MLeval)) install.packages("MLeval", repos = "http://cran.us.r-project.org")
if(!require(ggpubr)) install.packages("ggpubr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(kableExtra)
library(farff) #needed to convert the original .arff file into a dataframe
library(mltools) #used for one-hot encoding
library(data.table)
library(funModeling) #needed for some functions, e.g. plot_num(); NOTE: masks dplyr::summarize!
library(corrplot) #for creating correlation plot
library(DMwR) #for data balancing (SMOTE method)
library(broom)
library(randomForest)
library(MLeval) #for ROC and other cuves and metrics
library(ggpubr) #for combining plots into one

#Download the dataset from the internet and save into temporary file:
dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00277/ThoraricSurgery.arff",
              dl)
#Read the downloaded .arff file
lc <- readARFF(dl) %>% data.frame()

#Remove the temporary file
file.remove(dl)

#Observe the dataset
str(lc)

#Determine if the outcome and other binary variables are balanced
z <- sapply(lc, summary)
as.data.frame(z[c("Risk1Yr", "PRE7", "PRE8", "PRE9", "PRE10", "PRE11",
                  "PRE17", "PRE19", "PRE25", "PRE30", "PRE32")]) %>%
  kable() %>%
  kable_styling(latex_options = "scale_down",
                bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Rename the columns with more intuitive names
colnames(lc) <- c("Dg", "FVC", "FEV1", "Zubrod", "Pain", "Hemoptysis",
                "Dyspnea", "Cough", "Weakness", "Tumor_size", "T2DM", "MI", "PAD",
                "Smoking", "Asthma", "Age", "Death")

#Save the original dataset for further exploration
lc_orig <- lc

#Convert Boolean variables into integers; F into 0, T into 1
lc <- lc %>% mutate(Pain = as.factor(str_replace_all(Pain, "F", "0"))) %>%
  mutate(Pain = as.factor(str_replace_all(Pain, "T", "1"))) %>%
  mutate(Hemoptysis = as.factor(str_replace_all(Hemoptysis, "F", "0"))) %>%
  mutate(Hemoptysis = as.factor(str_replace_all(Hemoptysis, "T", "1"))) %>%
  mutate(Dyspnea = as.factor(str_replace_all(Dyspnea, "F", "0"))) %>%
  mutate(Dyspnea = as.factor(str_replace_all(Dyspnea, "T", "1"))) %>%
  mutate(Cough = as.factor(str_replace_all(Cough, "F", "0"))) %>%
  mutate(Cough = as.factor(str_replace_all(Cough, "T", "1"))) %>%
  mutate(Weakness = as.factor(str_replace_all(Weakness, "F", "0"))) %>%
  mutate(Weakness = as.factor(str_replace_all(Weakness, "T", "1"))) %>%
  mutate(T2DM = as.factor(str_replace_all(T2DM, "F", "0"))) %>%
  mutate(T2DM = as.factor(str_replace_all(T2DM, "T", "1"))) %>%
  mutate(MI = as.factor(str_replace_all(MI, "F", "0"))) %>%
  mutate(MI = as.factor(str_replace_all(MI, "T", "1"))) %>%
  mutate(PAD = as.factor(str_replace_all(PAD, "F", "0"))) %>%
  mutate(PAD = as.factor(str_replace_all(PAD, "T", "1"))) %>%
  mutate(Smoking = as.factor(str_replace_all(Smoking, "F", "0"))) %>%
  mutate(Smoking = as.factor(str_replace_all(Smoking, "T", "1"))) %>%
  mutate(Asthma = as.factor(str_replace_all(Asthma, "F", "0"))) %>%
  mutate(Asthma = as.factor(str_replace_all(Asthma, "T", "1"))) %>%
  mutate(Death = as.factor(str_replace_all(Death, "F", "0"))) %>%
  mutate(Death = as.factor(str_replace_all(Death, "T", "1")))
lc <- lc %>% mutate(Pain = as.numeric(levels(Pain))[Pain],
                      Hemoptysis = as.numeric(levels(Hemoptysis))[Hemoptysis],
                      Dyspnea = as.numeric(levels(Dyspnea))[Dyspnea],
                      Cough = as.numeric(levels(Cough))[Cough],
                      Weakness = as.numeric(levels(Weakness))[Weakness],
                      T2DM = as.numeric(levels(T2DM))[T2DM],
                      MI = as.numeric(levels(MI))[MI],
                      PAD = as.numeric(levels(PAD))[PAD],
                      Smoking = as.numeric(levels(Smoking))[Smoking],
                      Asthma = as.numeric(levels(Asthma))[Asthma],
                    Death = as.numeric(levels(Death))[Death])
lc <- lc %>% mutate(Pain = as.integer(Pain),
                        Hemoptysis = as.integer(Hemoptysis),
                        Dyspnea = as.integer(Dyspnea),
                        Cough = as.integer(Cough),
                        Weakness = as.integer(Weakness),
                        T2DM = as.integer(T2DM),
                        MI = as.integer(MI),
                        PAD = as.integer(PAD),
                        Smoking = as.integer(Smoking),
                        Asthma = as.integer(Asthma),
                    Death = as.integer(Death))

#OPTIONAL: Double-check if the sums of True values remained unchanged
#lc %>% select(-FVC, -FEV1, -Age) %>% summarize_if(is.numeric, sum, na.rm=TRUE)

#Convert categorical ordinal variables into integers, by removing the repeating strings
lc <- lc %>% mutate(Zubrod = str_replace(Zubrod, "PRZ", "")) %>%
  mutate(Tumor_size = str_replace(Tumor_size, "OC1", ""))
lc <- lc %>% mutate(Zubrod = as.integer(Zubrod),
              Tumor_size = as.integer(Tumor_size))

#Convert nominal variable with multiple categories (factors)
#into multiple variables (integers with walues 0 or 1) by one-hot encoding
data <- data.table(
  ID = seq(1,nrow(lc),by=1),
  Variable = lc$Dg)
newdata <- one_hot(data)
colnames(newdata) <- c("ID", "DGN3", "DGN2", "DGN4", "DGN6",
                       "DGN5", "DGN8", "DGN1")

#Join the dataframe with one-hot encoding variables and lc
lc <- lc %>% mutate(ID = row_number()) %>%
  left_join(newdata, by="ID") %>%
  mutate(Dg3 = DGN3, Dg2 = DGN2, Dg4 = DGN4, Dg6 = DGN6,
         Dg5 = DGN5, Dg8 = DGN8, Dg1 = DGN1) %>%
  select(-Dg, -ID, -DGN3, -DGN2, -DGN4, -DGN6, -DGN5, -DGN8, -DGN1)

#Show the first 6 rows of the pre-processed dataset
head(lc) %>%
  kable(caption = "lc dataset after pre-processing (first 6 rows)") %>%
  kable_styling(latex_options = "scale_down",
                bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Check the number of missing values
sum(is.na(lc))

#Plot the histograms of FVC, FEV1 and Age to check for outliers
lc_numerics <- lc %>% select(FVC, FEV1, Age)
plot_outliers <- plot_num(lc_numerics, bins=15)
#Format the plot (Code separated due to markdown issues)
plot_outliers +
  labs(title = "Histograms of metric variables",
       x = "Value",
       y = "Count")+
  theme_bw()

#Count the numbers of outliers for FEV1 with the Tukey's test
sum(is.na(prep_outliers(
  lc$FEV1,
  type = "set_na",
  method = "tukey")))

#Confirm there are no outliers for FVC with the Tukey's test
sum(is.na(prep_outliers(
  lc$FVC,
  type = "set_na",
  method = "tukey")))

#Count the numbers of outliers for Age with the Tukey's test
sum(is.na(prep_outliers(lc$Age,
                        type = "set_na",
                        method = "tukey")))

#Extract the rows with the outliers for FEV1
FEV1_outliers <- which(is.na(prep_outliers(lc$FEV1,
                          type = "set_na",
                          method = "tukey")))
lc[FEV1_outliers,] %>%
  select(FVC, FEV1, Age, Death) %>%
  kable(caption = "FEV1 Outliers") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Extract the row with the outlier for Age
Age_outlier <- which(is.na(prep_outliers(lc$Age,
                        type = "set_na",
                        method = "tukey")))
lc[Age_outlier,] %>%
  select(FVC, FEV1, Age, Death) %>%
  kable(caption = "Age Outlier") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Remove outliers from the lc dataset
lc <- lc[-c(FEV1_outliers, Age_outlier),]
#Check the structure of the cleaned lc dataset
str(lc)

#Join the cleaned lc dataset with "Dg" from the original lc dataset,
#then remove all added columns/variables
lc_orig_cleaned <- lc_orig %>% mutate(Dg = as.factor(str_replace_all(Dg, "DGN1", "1"))) %>%
  mutate(Dg = as.factor(str_replace_all(Dg, "DGN2", "2"))) %>%
  mutate(Dg = as.factor(str_replace_all(Dg, "DGN3", "3"))) %>%
  mutate(Dg = as.factor(str_replace_all(Dg, "DGN4", "4"))) %>%
  mutate(Dg = as.factor(str_replace_all(Dg, "DGN5", "5"))) %>%
  mutate(Dg = as.factor(str_replace_all(Dg, "DGN6", "6"))) %>%
  mutate(Dg = as.factor(str_replace_all(Dg, "DGN8", "8"))) %>%
  mutate(Dg = as.numeric(levels(Dg))[Dg]) %>%
  mutate(Dg = as.integer(Dg)) %>%
  mutate(ID = row_number())
lc_orig_to_join <- lc_orig_cleaned %>%
  filter(!((row_number() %in% FEV1_outliers) | (row_number() %in% Age_outlier))) %>%
  mutate(ID = row_number()) %>%
  select(ID, Dg)
lc_wID <- lc %>%
  mutate(ID = row_number())
lc_orig_cleaned_final <- left_join(lc_wID, lc_orig_to_join, by="ID") %>%
  select(FVC, FEV1, Age, Dg, Zubrod, Pain, Hemoptysis,
         Dyspnea, Cough, Weakness, Tumor_size,
         T2DM, MI, PAD, Smoking, Asthma, Death)

#Plot the distribution of the metric variables after removing the outliers
plot_lc_metric_cleaned <-
  plot_num(lc_orig_cleaned_final[,c("FVC","FEV1","Age")], bins = 15)

#Format the plot (Code separated due to markdown issues)
plot_lc_metric_cleaned +
  labs(title = "Distribution of metric variables without outliers",
       x = "Value",
       y = "Count")+
  theme_bw()

#Plot the distribution of the original categorical variables after removing the outliers
plot_lc_categorical_cleaned <-
  plot_num(lc_orig_cleaned_final[,c("Dg","Zubrod","Pain",
                                    "Hemoptysis", "Dyspnea", "Cough", "Weakness",
                                    "Tumor_size", "T2DM", "MI", "PAD", "Smoking",
                                    "Asthma", "Death")], bins = 15)

#Format the plot (Code separated due to markdown issues)
plot_lc_categorical_cleaned +
  labs(title = "Distribution of original categorical variables without outliers",
       x = "Factor Levels (from the lowest to highest)",
       y = "Count")+
  theme_bw()+
  theme(axis.text.x=element_blank())

#Create correlation matrix
correlationMatrix <- cor(lc)

#Create correlation plot
res1 <- cor.mtest(lc, conf.level = .95) #used for p.mat argument to determine stat. significance
corrplot(correlationMatrix, addrect = 6, tl.col = "black",
         p.mat = res1$p, insig = "blank") #squares with insignif. coefficients are left blank

#Calculate strong correlations
cm <- as.data.frame(as.table(correlationMatrix))
cm %>% filter(abs(Freq) > 0.5 & abs(Freq) < 1)

#Print the correlation coefficients with absolute values between 0.5 and 1.0
round(cor(lc$FVC, lc$FEV1), digits = 2)
round(cor(lc$Cough, lc$Zubrod), digits = 2)
round(cor(lc$Dg2, lc$Dg3), digits = 2)
round(cor(lc$Dg4, lc$Dg3), digits = 2)

#Determine which strongly correlated variables to be removed
strong_cor <- findCorrelation(correlationMatrix, cutoff=0.5)
colnames(lc[,strong_cor])

#Plot the FVC vs FEV1 and mapped Death
lc %>% group_by(Death) %>%
  ggplot(aes(FEV1, FVC, color=factor(Death), fill=factor(Death)))+
  geom_point(size=3, shape=21, alpha=0.5, position=position_jitter(h=0.1, w=0.1))+
  scale_fill_manual(values=c("turquoise", "plum2"))+ #change default fill color scheme
  scale_color_manual(values=c("turquoise", "plum2"))+
  theme_bw()+
    labs(fill = "Death", color = "Death",
         title = "FVC vs FEV1 with mapped Death",
         x = "FEV1 (Litres)",
         y = "FVC (Litres)")+
  annotate(geom="text", x=4.75, y=1.8,
           label=paste("Paerson's r =",round(cor(lc$FEV1, lc$FVC),2)), color="black")

#Plot the FVC vs Age and mapped Death
lc %>% group_by(Death) %>%
  ggplot(aes(Age, FVC, color=factor(Death), fill=factor(Death)))+
  geom_point(size=3, shape=21, alpha=0.6, position=position_jitter(h=0.1, w=0.1))+
  scale_fill_manual(values=c("turquoise", "plum2"))+ #change default fill color scheme
  scale_color_manual(values=c("turquoise", "plum2"))+
  theme_bw()+
  labs(fill = "Death", color = "Death",
       title = "FVC vs Age with mapped Death",
       x = "Age (Years)",
       y = "FVC (Litres)")+
  annotate(geom="text", x=78, y=6.2,
           label=paste("Paerson's r =",round(cor(lc$Age, lc$FVC),2)), color="black")

#Plot the Zubrod vs Tumor Size and mapped Death
lc %>% ggplot(aes(factor(Tumor_size), factor(Zubrod),
                  color=factor(Death), fill=factor(Death)))+
  geom_jitter(size=2, shape=21, alpha=0.6)+
  scale_fill_manual(values=c("turquoise", "plum2"))+ #change default fill color scheme
  scale_color_manual(values=c("turquoise", "plum2"))+
  theme_bw()+
  labs(fill = "Death", color = "Death",
       title = "Zubrod score vs Tumor size with mapped Death",
       x="Tumor Size",
       y="Zubrod Score")

#Calculate Paerson's r for Tumor_size and Zubrod
paste("Paerson's r =", round(cor(lc$Tumor_size, lc$Zubrod),2))


#Plot the FEV1/FVC ratio vs Age and mapped Death
lc_wCOPD <- lc %>% mutate(Tiff = FEV1/FVC, COPD=as.numeric(FEV1/FVC <=0.7))
lc_wCOPD %>%
  group_by(Death) %>%
  ggplot(aes(Age, Tiff, color=factor(Death), fill=factor(Death)))+
  geom_point(size=3, shape=21, alpha=0.6, position=position_jitter(h=0.1, w=0.1))+
  scale_fill_manual(values=c("turquoise", "plum2"))+ #change default fill color scheme
  scale_color_manual(values=c("turquoise", "plum2"))+
  theme_bw()+
  labs(fill = "Death", color = "Death",
       title = "FEV1/FVC Ratio vs Age with mapped Death",
       x="Age (Years)",
       y="FEV1/FVC Ratio")+
  geom_hline(yintercept=0.7, col = "black", lty=2)+
  annotate(geom="text", x=80, y=1.2,
           label=paste("Paerson's r =",round(cor(lc_wCOPD$Tiff, lc_wCOPD$Age),2)),
           color="black")

#Calculate Paerson's r for FEV1/FVC Ratio and Death
paste("Paerson's r =", round(cor(lc_wCOPD$Tiff, lc$Death),2))


#Plot the COPD vs Smoking and mapped Death
lc_wCOPD %>%
  ggplot(aes(factor(Smoking), factor(COPD),
             color=factor(Death), fill=factor(Death)))+
  geom_jitter(size=2, shape=21, alpha=0.6)+
  scale_fill_manual(values=c("turquoise", "plum"))+ #change default fill color scheme
  scale_color_manual(values=c("turquoise", "plum"))+
  theme_bw()+
  labs(fill = "Death", color = "Death",
       title = "COPD vs Smoking with mapped Death",
       x="Smoking",
       y="COPD")

#Calculate Paerson's r for COPD and Smoking
paste("Paerson's r =", round(cor(lc_wCOPD$COPD, lc$Smoking),2))

##########################
#DATA ANALYSIS
##########################

# Split the dataset to train and test set; test set will have 20% of the lc data
set.seed(16)
test_index <- createDataPartition(y = lc$Death, times = 1, p = 0.2, list = FALSE)
lc_train <- lc[-test_index,]
lc_test <- lc[test_index,]

#Observe the structure of both subsets
str(lc_train)
str(lc_test)

#prepare the training scheme with 10-fold cross validation on the whole train dataset
control <- trainControl(method="cv",
                        number=10,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        savePredictions = TRUE)

#Extract outcomes y and features x from the lc_train and lc_test
x_train <- lc_train %>% dplyr::select(-Death)
y_train <- factor(lc_train$Death, labels = c("Alive", "Dead"))
x_test <- lc_test %>% dplyr::select(-Death)
y_test <- factor(lc_test$Death, labels = c("Alive", "Dead"))

#Build a Naive Bayes model
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
library(klaR) #for Naive Bayes model; open here to avoid masking some dplyr functions (select)
set.seed(16)
model_nbayes <- train(x_train, y_train,
                      method = "nb",
                      metric="ROC",
                      tuneGrid = data.frame(fL = 0, usekernel = TRUE, adjust = 1),
                      trControl = control)
detach("package:klaR", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_nbayes <- predict(model_nbayes, x_test)

#Create the model's confusion matrix
cm_nbayes <- confusionMatrix(prediction_nbayes, y_test)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_nbayes <- tidy(cm_nbayes) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "Naive Bayes"=estimate)


#Build a Random Forest model
set.seed(16)
model_rf <- train(x_train, y_train,
                  method = "rf",
                  metric="ROC",
                  trControl = control)
#use the model to predict the observations from the lc_test features
prediction_rf <- predict(model_rf, x_test)

#Create the model's confusion matrix
cm_rf <- confusionMatrix(prediction_rf, y_test)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_rf <- tidy(cm_rf) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "Random Forest"=estimate)

#Build a XGBoost model
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb <- train(x_train, y_train,
                  method = "xgbTree",
                  metric = "ROC",
                  trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb <- predict(model_xgb, x_test)

#Create the model's confusion matrix
cm_xgb <- confusionMatrix(prediction_xgb, y_test)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb <- tidy(cm_xgb) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, XGBoost=estimate)

#Basic metrics of the models built on the original dataset
metrics_orig <- left_join(tidy_cm_nbayes, tidy_cm_rf, by="Metric") %>%
  left_join(.,tidy_cm_xgb, by="Metric")
metrics_orig %>% mutate(Metric = str_to_title(Metric)) %>%
  kable(caption = "Basic Metric Scores per Model - Cleaned Original Dataset") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Calculate and compare the model AUROCs
models_list_orig <- list(NaiveBayes = model_nbayes, RandomForest = model_rf, XGBoost = model_xgb)
models_results_orig <- resamples(models_list_orig)
AUROCs_orig <- summary(models_results_orig)
AUROCs_orig_df <- data.frame(AUROCs_orig$statistics$ROC[,4])
AUROCs_orig_df <- cbind(rownames(AUROCs_orig_df), AUROCs_orig_df) 
rownames(AUROCs_orig_df) <- NULL
colnames(AUROCs_orig_df) <- c("Model", "Mean AUROC")
AUROCs_orig_df %>%
  kable(caption = "Mean AUROC per Model - Cleaned Original Dataset") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)



#Plot the ROCs with AUROCs
auroc_orig <- evalm(list(model_nbayes,model_rf,model_xgb),
             gnames=c('NaiveBayes','RandomForest','XGBoost'),
             cols=c("turquoise", "plum", "plum1"),
             fsize=10,
             plots=c("r"),
             rlinethick = 0.5,
             dlinethick = 0.5,
             title="Original dataset")
#Plot the PRCs with AUPRCs
auprc_orig <- evalm(list(model_nbayes,model_rf,model_xgb),
                    gnames=c('NaiveBayes','RandomForest','XGBoost'),
                    cols=c("turquoise", "plum", "plum1"),
                    fsize=10,
                    plots=c("pr"),
                    rlinethick = 0.5,
                    dlinethick = 0.5,
                    title="Original dataset")

#Combine ROCs and PRCs into one plot
ggarrange(auroc_orig$roc, auprc_orig$proc,
          ncol = 1, nrow = 2)


###DATA BALANCING
#Apply the SMOTE method to the lc_train dataset
lc_train_fact <- lc_train %>% mutate(Death = factor(Death, labels = c("Alive", "Dead")))
set.seed(16)
lc_train_smote <- SMOTE(Death ~ ., data = lc_train_fact, k=5)
#Check the balance of the target variable
table(lc_train_smote$Death)

#Plot the variable distributions after the SMOTE
lc_train_smote_num <- lc_train_smote %>%
  mutate(Death = as.factor(str_replace_all(Death, "Alive", "0"))) %>%
  mutate(Death = as.factor(str_replace_all(Death, "Dead", "1"))) %>%
  mutate(Death = as.numeric(levels(Death))[Death])
plot_lc_train_smote_num <- plot_num(lc_train_smote_num)
plot_lc_train_smote_num +
  labs(title = "Distribution of variables after SMOTE data balancing",
       x = "Value",
       y = "Count")+
  theme_bw()

#Build a Random Forest model on the balanced SMOTE dataset
set.seed(16)
model_rf_smote <- train(Death ~ ., data = lc_train_smote,
                       method = "rf",
                       metric="ROC",
                       trControl = control)
#use the model to predict the observations from the lc_test features
prediction_rf_smote <- predict(model_rf_smote, x_test)
#Create the model's confusion matrix
cm_rf_smote <- confusionMatrix(prediction_rf_smote, y_test)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_rf_smote <- tidy(cm_rf_smote) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "Random Forest - Balanced"=estimate)


#Build a XGBoost model on the SMOTE balanced dataset
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb_smote <- train(Death ~ ., data = lc_train_smote,
                        method = "xgbTree",
                        metric = "ROC",
                        trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb_smote <- predict(model_xgb_smote, x_test)
#Create the model's confusion matrix
cm_xgb_smote <- confusionMatrix(prediction_xgb_smote, y_test)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb_smote <- tidy(cm_xgb_smote) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "XGBoost - Balanced"=estimate)

#Basic metrics of the models after data balancing with SMOTE
metrics_smote <- left_join(tidy_cm_rf_smote, tidy_cm_xgb_smote, by="Metric")
metrics_smote  %>%
  mutate(Metric = str_to_title(Metric)) %>%
  kable(caption = "Basic Metric Scores per Model - After SMOTE Data Balancing") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Calculate and compare the model AUROCs - after SMOTE
models_list_smote <- list(RandomForest_balanced = model_rf_smote, XGBoost_balanced = model_xgb_smote)
models_results_smote <- resamples(models_list_smote)
AUROCs_smote <- summary(models_results_smote)
AUROCs_smote_df <- data.frame(AUROCs_smote$statistics$ROC[,4])
AUROCs_smote_df <- cbind(rownames(AUROCs_smote_df), AUROCs_smote_df) 
rownames(AUROCs_smote_df) <- NULL
colnames(AUROCs_smote_df) <- c("Model", "Mean AUROC")
AUROCs_smote_df %>%
  kable(caption = "Mean AUROC per Model - Smote Balanced Dataset") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Plot the ROCs with AUROCs - after SMOTE
auroc_smote <- evalm(list(model_rf_smote,model_xgb_smote),
                    gnames=c('RandomForest','XGBoost'),
                    cols=c("turquoise", "plum"),
                    fsize=10,
                    plots=c("r"),
                    rlinethick = 0.5,
                    dlinethick = 0.5,
                    title="SMOTE Balanced dataset")
#Plot the PRCs with AUPRCs  - after SMOTE
auprc_smote <- evalm(list(model_rf_smote,model_xgb_smote),
                    gnames=c('RandomForest','XGBoost'),
                    cols=c("turquoise", "plum"),
                    fsize=10,
                    plots=c("pr"),
                    rlinethick = 0.5,
                    dlinethick = 0.5,
                    title="SMOTE Balanced dataset")
#Combine ROCs and PRCs into one plot - after SMOTE
ggarrange(auroc_smote$roc, auprc_smote$proc,
          ncol = 1, nrow = 2)

#FEATURE IMPORTANCE
#Plot the RF model's 10 most important variables
plot_fi_rf <- plot(varImp(model_rf), top=10, main="The most important variables - Random Forest")
#Plot the XGB model's 10 most important variables
plot_fi_xgb <- plot(varImp(model_xgb), top=10, main="The most important variables - XGBoost")
ggarrange(plot_fi_rf, plot_fi_xgb,
          ncol = 1, nrow = 2)


#Add new computated features to lc_train and lc_test
lc_train_eng <- lc_train %>% mutate(Death = factor(Death, labels = c("Alive", "Dead")),
                                    "FEV1/FVC" = FEV1/FVC,
                                    "FVC*FEV1" = FVC*FEV1,
                                    "FVC*FEV1^2" = FVC*(FEV1)^2,
                                    "FVC^2*FEV1" = (FVC^2)*FEV1,
                                    "FVC^2" = FVC^2,
                                    "Age*FVC" = Age*FVC,
                                    "Age*FEV1" = Age*FEV1,
                                    "Age/Tumor_size" = Age/Tumor_size,
                                    "FVC*Tumor_size" = FVC*Tumor_size)
lc_test_eng <- lc_test %>% mutate(Death = factor(Death, labels = c("Alive", "Dead")),
                                  "FEV1/FVC" = FEV1/FVC,
                                  "FVC*FEV1" = FVC*FEV1,
                                  "FVC*FEV1^2" = FVC*(FEV1)^2,
                                  "FVC^2*FEV1" = (FVC^2)*FEV1,
                                  "FVC^2" = FVC^2,
                                  "Age*FVC" = Age*FVC,
                                  "Age*FEV1" = Age*FEV1,
                                  "Age/Tumor_size" = Age/Tumor_size,
                                  "FVC*Tumor_size" = FVC*Tumor_size)
x_lc_test_eng <- lc_test_eng %>% dplyr::select(-Death)

#Evaluate the importance with the new features on XGB model
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb_eng <- train(Death ~ ., data = lc_train_eng,
                         method = "xgbTree",
                         metric = "ROC",
                         trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb_eng <- predict(model_xgb_eng, x_lc_test_eng)
#Create the model's confusion matrix
cm_xgb_eng <- confusionMatrix(prediction_xgb_eng, lc_test_eng$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb_eng <- tidy(cm_xgb_eng) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "XGBoost - New Features"=estimate)
#Plot the XGB model's 10 most important variables After Engineering
plot_fi_xgb_eng <- plot(varImp(model_xgb_eng), top=10,
                       main="The most important variables, \n XGBoost (After Engineering)")
plot_fi_xgb_eng

#FIND STRONG CORRELATIONS after the feature engineering
lc_train_eng_numeric <- lc_train %>% mutate("FEV1/FVC" = FEV1/FVC,
                                            "FVC*FEV1" = FVC*FEV1,
                                            "FVC*FEV1^2" = FVC*(FEV1)^2,
                                            "FVC^2*FEV1" = (FVC^2)*FEV1,
                                            "FVC^2" = FVC^2,
                                            "Age*FVC" = Age*FVC,
                                            "Age*FEV1" = Age*FEV1,
                                            "Age/Tumor_size" = Age/Tumor_size,
                                            "FVC*Tumor_size" = FVC*Tumor_size)
#Create correlation matrix
correlationMatrix_eng <- cor(lc_train_eng_numeric)
#Determine which strongly correlated variables to be removed
strong_cor_eng <- findCorrelation(correlationMatrix_eng, names=TRUE, cutoff=0.5)
strong_cor_eng

############
#CONSTRUCT THE FINAL DATASET (Balanced train subset + Added New Features)
lc_train_final <- lc_train_smote %>% mutate("FEV1/FVC" = FEV1/FVC,
                          "FVC*FEV1" = FVC*FEV1,
                          "FVC*FEV1^2" = FVC*(FEV1)^2,
                          "FVC^2*FEV1" = (FVC^2)*FEV1,
                          "FVC^2" = FVC^2,
                          "Age*FVC" = Age*FVC,
                          "Age*FEV1" = Age*FEV1,
                          "Age/Tumor_size" = Age/Tumor_size,
                          "FVC*Tumor_size" = FVC*Tumor_size)
lc_test_final <- lc_test_eng
x_lc_test_final <- x_lc_test_eng

#Build a XGBoost model on the FINAL dataset (balanced + added features)
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb_final <- train(Death ~ ., data = lc_train_final,
                         method = "xgbTree",
                         metric = "ROC",
                         trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb_final <- predict(model_xgb_final, x_lc_test_final)
#Create the model's confusion matrix
cm_xgb_final <- confusionMatrix(prediction_xgb_final, lc_test_final$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb_final <- tidy(cm_xgb_final) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "XGBoost - Final"=estimate)

#Costruct the Final Dataset With Removed highly correlated variables
lc_train_final_num <- lc_train_final %>%
  mutate(Death = as.factor(str_replace_all(Death, "Alive", "0"))) %>%
  mutate(Death = as.factor(str_replace_all(Death, "Dead", "1"))) %>%
  mutate(Death = as.numeric(levels(Death))[Death]) %>%
  dplyr::select(-Dg1) #removed because it has zero SD, and prevents calculation of correlations
lc_train_final_nocor <- lc_train_final %>% dplyr::select(-strong_cor_eng)
lc_test_final_nocor <- lc_test_final %>% dplyr::select(-strong_cor_eng)
x_lc_test_final_nocor <- lc_test_final_nocor %>% dplyr::select(-Death)

#Build a XGBoost model on the FINAL dataset without highly correlated variables
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb_final_nocor <- train(Death ~ ., data = lc_train_final_nocor,
                         method = "xgbTree",
                         metric = "ROC",
                         trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb_final_nocor <- predict(model_xgb_final_nocor, x_lc_test_final_nocor)
#Create the model's confusion matrix
cm_xgb_final_nocor <- confusionMatrix(prediction_xgb_final_nocor, lc_test_final_nocor$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb_final_nocor <- tidy(cm_xgb_final_nocor) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "XGBoost - Final (No_Corr)"=estimate)

#Basic metrics of the XGB models
metrics_xgbs <- left_join(tidy_cm_xgb, tidy_cm_xgb_smote, by="Metric") %>%
  left_join(.,tidy_cm_xgb_final, by="Metric") %>%
  left_join(.,tidy_cm_xgb_final_nocor, by="Metric")
metrics_xgbs %>% mutate(Metric = str_to_title(Metric)) %>%
  kable(format = "pandoc",
        caption = "Basic Metric Scores - XGB, Original and Modified Datasets") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Calculate and compare the AUROCs - XGBs
models_list_xgbs <- list(XGB_orig = model_xgb, XGB_balanced = model_xgb_smote,
                         XGB_final = model_xgb_final,
                         XGB_final_No_Cor = model_xgb_final_nocor)
models_results_xgbs<- resamples(models_list_xgbs)
AUROCs_xgbs <- summary(models_results_xgbs)
AUROCs_xgbs_df <- data.frame(AUROCs_xgbs$statistics$ROC[,4])
AUROCs_xgbs_df <- cbind(rownames(AUROCs_xgbs_df), AUROCs_xgbs_df) 
rownames(AUROCs_xgbs_df) <- NULL
colnames(AUROCs_xgbs_df) <- c("Model", "Mean AUROC")
AUROCs_xgbs_df %>%
  kable(caption = "Mean AUROC for XGBoost per each dataset") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Plot the ROCs with AUROCs - all XGBs
auroc_xgbs <- evalm(list(model_xgb, model_xgb_smote, model_xgb_final, model_xgb_final_nocor),
                    cols = c("red", "blue", "plum1", "turquoise"),
                    fsize=10,
                    plots=c("r"),
                    rlinethick = 0.5,
                    dlinethick = 0.5,
                    title="Original And Modified Datasets")
auprc_xgbs <- evalm(list(model_xgb, model_xgb_smote, model_xgb_final, model_xgb_final_nocor),
                    cols = c("red", "blue", "plum1", "turquoise"),
                    fsize=10,
                    plots=c("pr"),
                    rlinethick = 0.5,
                    dlinethick = 0.5,
                    title="Original And Modified Datasets")
#Combine ROCs and PRCs into one plot - all XGBs
ggarrange(auroc_xgbs$roc, auprc_xgbs$proc,
          ncol = 1, nrow = 2)

#Show the confusion matrix of the xgbs
cm_xgb$table
cm_xgb_smote$table
cm_xgb_final_nocor$table


#Build a RF model on the FINAL dataset (balanced + added features)
set.seed(16)
model_rf_final <- train(Death ~ ., data = lc_train_final,
                         method = "rf",
                         metric = "ROC",
                         trControl = control)
#use the model to predict the observations from the lc_test_final features
prediction_rf_final <- predict(model_rf_final, x_lc_test_final)
#Create the model's confusion matrix
cm_rf_final <- confusionMatrix(prediction_rf_final, lc_test_final$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_rf_final <- tidy(cm_rf_final) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "RF - Final"=estimate)

#Compare the performance with RF original and balanced
#Basic metrics of the RF models
metrics_rfs <- left_join(tidy_cm_rf, tidy_cm_rf_smote, by="Metric") %>%
  left_join(.,tidy_cm_rf_final, by="Metric")
metrics_rfs %>% mutate(Metric = str_to_title(Metric)) %>%
  kable(format = "pandoc",
        caption = "Basic Metric Scores - RF, Original and Modified Datasets") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Show the CM
cm_rf$table
cm_rf_smote$table
cm_rf_final$table

#Build a SVM model on the FINAL dataset (balanced + added features)
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
library(kernlab)
set.seed(16)
model_svm_final <- train(Death ~ ., data = lc_train_final,
                            method = "svmRadial",
                            metric="ROC",
                            trControl = control)
detach("package:kernlab", unload = TRUE)
#use the model to predict the observations from the lc_test_final features
prediction_svm_final <- predict(model_svm_final, x_lc_test_final)
#Create the model's confusion matrix
cm_svm_final <- confusionMatrix(prediction_svm_final, lc_test_final$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_svm_final <- tidy(cm_svm_final) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "SVM - Final"=estimate)
#Show the CM and basic metrics
cm_svm_final$table
tidy_cm_svm_final

#Build a Model Averaged Neural Network model on the FINAL dataset (balanced + added features)
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
library(nnet)
set.seed(16)
model_avnnet_final <- train(Death ~ ., data = lc_train_final,
                            method = "avNNet",
                            metric="ROC",
                            trControl = control)
detach("package:nnet", unload = TRUE)
#use the model to predict the observations from the lc_test_final features
prediction_avnnet_final <- predict(model_avnnet_final, x_lc_test_final)
#Create the model's confusion matrix
cm_avnnet_final <- confusionMatrix(prediction_avnnet_final, lc_test_final$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_avnnet_final <- tidy(cm_avnnet_final) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "AvNNet - Final"=estimate)

#Build a Model Averaged NN model on the FINAL dataset without highly correlated variables
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
library(nnet)
set.seed(16)
model_avnnet_final_nocor <- train(Death ~ ., data = lc_train_final_nocor,
                            method = "avNNet",
                            metric="ROC",
                            trControl = control)
detach("package:nnet", unload = TRUE)
#use the model to predict the observations from the lc_test_final features
prediction_avnnet_final_nocor <- predict(model_avnnet_final_nocor, x_lc_test_final_nocor)
#Create the model's confusion matrix
cm_avnnet_final_nocor <- confusionMatrix(prediction_avnnet_final_nocor, lc_test_final_nocor$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_avnnet_final_nocor <- tidy(cm_avnnet_final_nocor) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "AvNNet - Final_NoCor"=estimate)

#Basic metrics of the NNet models
metrics_nn <- left_join(tidy_cm_avnnet_final, tidy_cm_avnnet_final_nocor, by="Metric")
metrics_nn %>% mutate(Metric = str_to_title(Metric)) %>%
  kable(format = "pandoc", caption = "Basic Metric Scores - NeuralNetwork") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Show the CMs of the NNet models
cm_avnnet_final$table
cm_avnnet_final_nocor$table


#Plot the ROCs with AUROCs - both NNs
auroc_nn <- evalm(list(model_avnnet_final, model_avnnet_final_nocor),
                    cols = c("plum1", "turquoise"),
                  gnames = c("NNet_final", "NNet_final_NoCor"),
                    fsize=10,
                    plots=c("r"),
                    rlinethick = 0.5,
                    dlinethick = 0.5,
                    title="Final Dataset")
auprc_nn <- evalm(list(model_avnnet_final, model_avnnet_final_nocor),
                  cols = c("plum1", "turquoise"),
                  gnames = c("NNet_final", "NNet_final_NoCor"),
                  fsize=10,
                  plots=c("pr"),
                  rlinethick = 0.5,
                  dlinethick = 0.5,
                  title="Final Dataset")
#Combine ROCs and PRCs into one plot - all XGBs
ggarrange(auroc_nn$roc, auprc_nn$proc,
          ncol = 1, nrow = 2)


####EXPLORE THE IMPACT OF ONE HOT ENCODING

#Cleaned original dataset without One Hot Encoding (=DGN dataset)
lc_orig_cleaned_final

#Plot the correlation plot of this dataset
correlationMatrix_dgn <- cor(lc_orig_cleaned_final)
res_dgn <- cor.mtest(lc_orig_cleaned_final, conf.level = .95) #used for p.mat argument to determine stat. significance
corrplot(correlationMatrix_dgn, addrect = 6, tl.col = "black",
         p.mat = res_dgn$p, insig = "blank")

#Determine which strongly correlated variables to be removed
strong_cor_dgn <- findCorrelation(correlationMatrix_dgn, cutoff=0.5)
colnames(lc_orig_cleaned_final[,strong_cor_dgn])

#Split the DGN dataset to train and test set; test set will have 20% of the lc data
set.seed(16)
test_index_dgn <- createDataPartition(y = lc_orig_cleaned_final$Death, times = 1, p = 0.2, list = FALSE)
lc_train_dgn <- lc_orig_cleaned_final[-test_index_dgn,] %>%
  mutate(Death = factor(Death, labels = c("Alive", "Dead")))
lc_test_dgn <- lc_orig_cleaned_final[test_index_dgn,] %>%
  mutate(Death = factor(Death, labels = c("Alive", "Dead")))
x_lc_test_dgn <- lc_test_dgn %>% dplyr::select(-Death)


#Build a Random Forest model - DGN
set.seed(16)
model_rf_dgn <- train(Death ~ ., data = lc_train_dgn,
                      method = "rf",
                      metric="ROC",
                      trControl = control)
#use the model to predict the observations from the lc_test features
prediction_rf_dgn <- predict(model_rf_dgn, x_lc_test_dgn)
#Create the model's confusion matrix
cm_rf_dgn <- confusionMatrix(prediction_rf_dgn, lc_test_dgn$Death)


#Build a XGB model - DGN
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb_dgn <- train(Death ~ ., data = lc_train_dgn,
                       method = "xgbTree",
                       metric="ROC",
                       trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb_dgn <- predict(model_xgb_dgn, x_lc_test_dgn)
#Create the model's confusion matrix
cm_xgb_dgn <- confusionMatrix(prediction_xgb_dgn, lc_test_dgn$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb_dgn <- tidy(cm_xgb_dgn) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "XGBoost - DGN original"=estimate)


###DATA BALANCING
#Apply the SMOTE method to the lc_train_dgn dataset
set.seed(16)
lc_train_dgn_smote <- SMOTE(Death ~ ., data = lc_train_dgn, k=5)
#Check the balance of the target variable
table(lc_train_dgn_smote$Death)

###Add new engineered variables
lc_train_dgn_final <- lc_train_dgn_smote %>%
  mutate("FEV1/FVC" = FEV1/FVC,
         "FVC*FEV1" = FVC*FEV1,
         "FVC*FEV1^2" = FVC*(FEV1)^2,
         "FVC^2*FEV1" = (FVC^2)*FEV1,
         "FVC^2" = FVC^2,
         "Age*FVC" = Age*FVC,
         "Age*FEV1" = Age*FEV1,
         "Age/Tumor_size" = Age/Tumor_size,
         "FVC*Tumor_size" = FVC*Tumor_size)
lc_test_dgn_final <- lc_test_dgn %>%
  mutate("FEV1/FVC" = FEV1/FVC,
         "FVC*FEV1" = FVC*FEV1,
         "FVC*FEV1^2" = FVC*(FEV1)^2,
         "FVC^2*FEV1" = (FVC^2)*FEV1,
         "FVC^2" = FVC^2,
         "Age*FVC" = Age*FVC,
         "Age*FEV1" = Age*FEV1,
         "Age/Tumor_size" = Age/Tumor_size,
         "FVC*Tumor_size" = FVC*Tumor_size)
x_lc_test_dgn_final <- lc_test_dgn_final %>% dplyr::select(-Death)


#Build a XGB model - DGN final 
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
library(xgboost) #open here to avoid masking some dplyr functions (slice)
set.seed(16)
model_xgb_dgn_final <- train(Death ~ ., data = lc_train_dgn_final,
                             method = "xgbTree",
                             metric="ROC",
                             trControl = control)
detach("package:xgboost", unload = TRUE)
#use the model to predict the observations from the lc_test features
prediction_xgb_dgn_final <- predict(model_xgb_dgn_final, x_lc_test_dgn_final)
#Create the model's confusion matrix
cm_xgb_dgn_final <- confusionMatrix(prediction_xgb_dgn_final, lc_test_dgn_final$Death)
#Transform the CM into the tidy format and extract key evaluation metrics
tidy_cm_xgb_dgn_final <- tidy(cm_xgb_dgn_final) %>% slice(1,4,5,8,9,10,11,14) %>%
  dplyr::select(term, estimate) %>% rename(Metric=term, "XGBoost - DGN Final"=estimate)

#Basic metrics of the XGB models with and without one hot encoding
metrics_xgbs_dgn <- left_join(tidy_cm_xgb_dgn, tidy_cm_xgb, by="Metric") %>%
  left_join(.,tidy_cm_xgb_dgn_final, by="Metric") %>%
  left_join(.,tidy_cm_xgb_final, by="Metric")
metrics_xgbs_dgn %>% mutate(Metric = str_to_title(Metric)) %>%
  kable(format = "pandoc",
        caption = "Basic Metric Scores - Impact of One-Hot Encoding") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#AUROC with caret
models_list_dgn <- list(XGB_dgn_orig = model_xgb_dgn, XGB_orig = model_xgb, 
                        XGB_dgn_final = model_xgb_dgn_final, XGB_final = model_xgb_final)
models_results_dgn <- resamples(models_list_dgn)
AUROCs_dgn <- summary(models_results_dgn)
AUROCs_dgn_df <- data.frame(AUROCs_dgn$statistics$ROC[,4])
AUROCs_dgn_df <- cbind(rownames(AUROCs_dgn_df), AUROCs_dgn_df) 
rownames(AUROCs_dgn_df) <- NULL
colnames(AUROCs_dgn_df) <- c("Model", "Mean AUROC")
AUROCs_dgn_df %>%
  kable(format = "pandoc", caption = "Mean AUROC - Impact of One-Hot Encoding") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

######################
#RESULTS COMPARISON
######################

#Basic metrics of the best individual models
metrics_all_top <- left_join(tidy_cm_xgb_final, tidy_cm_xgb_final_nocor, by="Metric") %>%
  left_join(.,tidy_cm_rf_final, by="Metric") %>%
  left_join(.,tidy_cm_svm_final, by="Metric") %>%
  left_join(.,tidy_cm_avnnet_final, by="Metric") %>%
  left_join(.,tidy_cm_avnnet_final_nocor, by="Metric")
metrics_all_top %>% mutate(Metric = str_to_title(Metric)) %>%
  kable(format = "pandoc",
        caption = "Basic Metric Scores - All Top Models") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Mean AUROC all top performing models (CARET)
models_list_all_top <- list(XGBoost_final = model_xgb_final,
                            XGBoost_final_NoCor = model_xgb_final_nocor,
                            RandomForest_final = model_rf_final,
                            SVM_final = model_svm_final,
                            NNet_final = model_avnnet_final,
                            NNet_final_NoCor = model_avnnet_final_nocor)
models_results_all_top <- resamples(models_list_all_top)
AUROCs_all_top <- summary(models_results_all_top)
AUROCs_all_top_df <- data.frame(AUROCs_all_top$statistics$ROC[,4])
AUROCs_all_top_df <- cbind(rownames(AUROCs_all_top_df), AUROCs_all_top_df) 
rownames(AUROCs_all_top_df) <- NULL
colnames(AUROCs_all_top_df) <- c("Model", "Mean AUROC")
AUROCs_all_top_df %>%
  kable(format = "pandoc", caption = "Mean AUROC - All Top Models") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"),
                position = "center",
                font_size = 10,
                full_width = FALSE)

#Plot the AUROCs and their variability for the top models
bwplot(models_results_all_top, metric="ROC")

#AUPRC all top performing models
auprc_all_top <- evalm(list(model_xgb_final, model_xgb_final_nocor,
                            model_avnnet_final, model_avnnet_final_nocor),
                       gnames = c("XGBoost_final", "XGBoost_final_NoCor",
                            "NNet_final", "NNet_final_NoCor"),
                       cols = c("turquoise", "turquoise1", "plum", "plum1"),
                   fsize=10,
                   plots=c("pr"),
                   rlinethick = 0.5,
                   dlinethick = 0.5,
                   title="PRC - Top Models")
auprc_all_top$proc


#R version details
version

