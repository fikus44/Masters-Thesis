##################################################
#                                                #
#  Master's Thesis                               #
#  Machine Learning in Asset Pricing             #
#                                                #
#  Thomas Theodor Kjølbye                        #
#                                                #
#  The following script produces all output      #
#  used in the paper. On my computer, the        #
#  entire script takes approx. 1  minute to run. #
#                                                #
##################################################

########################## Settings ###############################

# Clear 
rm(list = ls())

# Set working directory to source file location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

# Set language to english
Sys.setenv(LANG = "EN")

# Disable scientific notation (e) 
options(scipen = 999)

# Load packages
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, lubridate, ggplot2, egg, gridExtra, knitr, kableExtra, glmnet, zeallot, neuralnet, comprehenr, data.table)

# Import own table customization compatible with LaTeX
source("table_theme.R", local = FALSE)

# Initialize warning log container
log <- c()

# Timer start
start_time <- Sys.time()


### 1. Tables -----------------------------------------------------

# Set WD
setwd("C:/Users/thoma/OneDrive - KØbenhavns Universitet/Documents/Økonomi - Kandidat/6. Semester/Speciale")

# Load data
table1_data <- read.csv(file = "table1_data.csv")
table2_data <- read.csv(file = "table2_data.csv")
table4_data <- read.csv(file = "table4_data.csv")

# Load Linear model regressions
CAPM_params <- read.csv(file = "CAPM_params.csv")
CAPM_tvalues <- read.csv(file = "CAPM_tvalues.csv")
FF3_params <- read.csv(file = "FF3_params.csv")
FF3_tvalues <- read.csv(file = "FF3_tvalues.csv")
FF5_params <- read.csv(file = "FF5_params.csv")
FF5_tvalues <- read.csv(file = "FF5_tvalues.csv")


# Create tables
names1 <- c("Metric", "Linear Regression", "Lasso", "Neural Network")
table1 <- table_theme(table1_data, colnames = names1, caption = "Squared Prediction Error and Explained Variation") 

names2 <- c("Metric", "Linear Regression", "Lasso", "Neural Network")
table2 <-  table_theme(table2_data, colnames = names1, caption = "Squared Prediction Error and Explained Variation") 

names4 <- c("X", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
table4 <- table_theme(table4_data, colnames = names4, caption = "Cumulative return of all machine learning portfolios")


# Create Linear model tables
names_FF3 <- c("Model" ,"Constant", "Mkt-RF", "SMB", "HML")
FF3_params_table <- table_theme(FF3_params, colnames = names_FF3, caption = "FF3 Params")
FF3_tvalues_table <- table_theme(FF3_tvalues, colnames = names_FF3, caption = "FF3 t")

names_CAPM <- c("Model" ,"Constant", "Mkt-RF")
CAPM_params_table <- table_theme(CAPM_params, colnames = names_CAPM, caption = "CAPM Params")
CAPM_tvalues_table <- table_theme(CAPM_tvalues, colnames = names_CAPM, caption = "CAPM t")

names_FF5 <- c("Model" ,"Constant", "Mkt-RF", "SMB", "HML", "RMW", "CMA")
FF5_params_table <- table_theme(FF5_params, colnames = names_FF5, caption = "FF5 Params")
FF5_tvalues_table <- table_theme(FF5_tvalues, colnames = names_FF5, caption = "FF5 t")


# Timer finished
end_time <- Sys.time()

print(paste("Total time:", end_time - start_time))
