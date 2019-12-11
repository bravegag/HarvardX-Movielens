##########################################################################################
## Capstone Project MovieLens.
##
## Author: Giovanni Azua Garcia <giovanni.azua@outlook.com>
##########################################################################################

# clean the environment
rm(list = ls())
# trigger garbage collection and free some memory if possible
gc(TRUE, TRUE, TRUE)

##########################################################################################
## Install and load required library dependencies
##########################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(tictoc)) install.packages("tictoc", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(doMC)) install.packages("doMC", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(tictoc)
library(lubridate)
library(stringr)
library(doMC)

##########################################################################################
## Define important reusable functions e.g. the RMSE function
##########################################################################################

# Loss function: the root mean squares estimate
RMSE <- function(x, y) {
  sqrt(mean((x - y)^2))
}

# portable (across R versions) set.seed function implementation
portable.set.seed <- function(seed) {
  if (R.version$minor < "6") {
    set.seed(seed)
  } else {
    set.seed(seed, sample.kind="Rounding")
  }
}

##########################################################################################
## Create edx set, validation set
##########################################################################################

# Note: this process could take a couple of minutes

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

portable.set.seed(1)

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################################################
## Standarize the dataset names
##########################################################################################

# training set is edx
train_set <- edx

# validation set
validation_set <- validation

##########################################################################################
## Computing and adding the minimum timestamp `min_ts` feature needed later. The minimum 
## timestamp is computed per movie and it's a proxy for the release date of a movie. Here I 
## assume that the release date of a movie corresponds to the first available rating entry 
## for that movie. This is needed in order to create a new feature, the number of weeks 
## since the movie was launched.
##########################################################################################

## VALIDATION SET ACCESS ALERT! accessing the validation set to add a feature globally.
tic("adding the minimum timestamp feature to the full dataset")
ts_mins <- train_set %>% 
  bind_rows(validation_set) %>%
  group_by(movieId) %>%
  summarise(min_ts = min(timestamp))

# add ts_min attribute to the train_set
train_set <- train_set %>%
  left_join(ts_mins, by="movieId")

# add ts_min attribute to the validation_set
validation_set <- validation_set %>%
  left_join(ts_mins, by="movieId")
toc()

# remove variables and data that are no longer needed
rm(ts_mins)

##########################################################################################
## Create experimental dataset as subset of the training dataset with 500k samples
##########################################################################################

# set the seed again
portable.set.seed(1)
# create subset of the training edx set
experimental_set <- train_set %>%
  sample_n(1000000)

# global average
mu <- mean(experimental_set$rating)
# compute predictions
rmse_results <- tibble(method = "Just the average", RMSE = RMSE(experimental_set$rating, mu))

# lambda is the optimal one found using CV later
lambda <- 5
# compute regularized movie effects
movie_avgs <- experimental_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + lambda))
# compute predictions
predicted_ratings <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  mutate(pred=mu + b_i) %>%
  pull(pred)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie Effects Model",
                                 RMSE = RMSE(predicted_ratings, experimental_set$rating)))

# compute regularized user effects
user_avgs <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n() + lambda))
# compute predictions
predicted_ratings <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User Effects Model",
                                 RMSE = RMSE(predicted_ratings, experimental_set$rating)))

# compute regularized genre effects
genre_avgs <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - (mu + b_i + b_u))/(n() + lambda))
# compute predictions
predicted_ratings <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genre Effects Model",
                                 RMSE = RMSE(predicted_ratings, experimental_set$rating)))

# show the progress so far, to see how the RMSE keeps decreasing as we account 
# for more effect types
as.data.frame(rmse_results)

# add day of week feature to the experimental dataset
experimental_set <- experimental_set %>% 
  mutate(dayOfTheWeek = recode_factor(as.factor(wday(as_datetime(timestamp))), `1`='Mon', `2`='Tue', `3`='Wed', `4`='Thu', `5`='Fri', `6`='Sat', `7`='Sun'))
# compute regularized day of the week effects
day_avgs <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(dayOfTheWeek) %>%
  summarise(b_d = sum(rating - (mu + b_i + b_u + b_g))/(n() + lambda))
# plot the day of the week effects
day_avgs %>%
  ggplot(aes(dayOfTheWeek, b_d, group = 1)) + geom_point() + geom_line()
# compute predictions
predicted_ratings <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(day_avgs, by='dayOfTheWeek') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d) %>%
  pull(pred)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genre + Day of the Week Effects Model",
                                 RMSE = RMSE(predicted_ratings, experimental_set$rating)))

# add day of month feature to the experimental dataset
experimental_set <- experimental_set %>% 
  mutate(dayOfTheMonth = as.factor(day(round_date(as_datetime(timestamp), unit = "day"))))
# compute regularized day of the month effects
day_avgs <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  group_by(dayOfTheMonth) %>%
  summarise(b_d = sum(rating - (mu + b_i + b_u + b_g))/(n() + lambda))
# plot the day of the month effects
day_avgs %>%
  ggplot(aes(dayOfTheMonth, b_d, group = 1)) + geom_point() + geom_line()
# compute predictions
predicted_ratings <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(day_avgs, by='dayOfTheMonth') %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d) %>%
  pull(pred)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genre + Day of the Month Effects Model",
                                 RMSE = RMSE(predicted_ratings, experimental_set$rating)))

# show the progress so far, to see how the RMSE keeps decreasing as we account 
# for more effect types
as.data.frame(rmse_results)

# compute the week of each rating since the movie release date and smooth the effect
experimental_set %>%
  mutate(week_block_2 = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(2))) %>%
  group_by(week_block_2) %>%
  summarise(b_w_Effect=mean(rating)) %>%
  ggplot(aes(week_block_2, b_w_Effect)) + geom_point() + 
  geom_smooth(color="red", span=0.3, method.args=list(degree=2))

# fit the elapsed time week model
week_fit <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(day_avgs, by='dayOfTheMonth') %>%
  mutate(week = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(2))) %>%
  group_by(week) %>%
  summarise(rating_residual=mean(rating - (mu + b_i + b_u + b_g + b_d))) %>%
  loess(rating_residual~week, data=., span=0.3, degree=2)
# compute predictions
predicted_ratings <- experimental_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  left_join(day_avgs, by='dayOfTheMonth') %>%
  mutate(week = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(2))) %>%
  mutate(pred = mu + b_i + b_u + b_g + b_d + predict(week_fit, .)) %>%
  pull(pred)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularized Movie + User + Genre + Day of the Month + Week Effects Model",
                                 RMSE = RMSE(predicted_ratings, experimental_set$rating)))

# show the progress so far, to see how the RMSE keeps decreasing as we account 
# for more effect types
as.data.frame(rmse_results)

# remove variables and data that are no longer needed
rm(predicted_ratings, rmse_results, week_fit, experimental_set, day_avgs, genre_avgs, user_avgs, 
   movie_avgs, lambda, mu)

##########################################################################################
## Create the CF BASIC model integrated with the caret package. The model is called cfBasic
## standing for Collaborative Filtering basic method which provides some extensions and 
## improvements over the model described in the lectures and book.
##
## Extensions over the book model:
## - Models the genres effect by computing the regularized rating genres effects.
## - Models the time effects using the day of the week or month for a rating.
## - Models the time effects using the number of week blocks since the movie was released.
##
## This caret model specification follows the implementation details described here:
## https://topepo.github.io/caret/using-your-own-model-in-train.html
##########################################################################################

# trigger garbage collection and free some memory if possible
gc(TRUE, TRUE, TRUE)

# Define the model cFBasic (Collaborative Filtering basic)
cFBasic <- list(type = "Regression",
                library = c("lubridate", "stringr"),
                loop = NULL,
                prob = NULL,
                sort = NULL)

# Five different parameters are supported:
# @param lambda the regularizaton parameter applied to the different effects
# @param span the span parameter applied to loess for smoothing the week elapsed time effects
# @param degree the degree parameter applied to loess for smoothing the week elapsed time effects
# @param weekSpan the number of weeks to bin the data with.
# @param dayType whether "dayOfTheWeek" e.g. 1-7 or "dayOfTheMonth" 1-31
#
cFBasic$parameters <- data.frame(parameter = c("lambda", "span", "degree", "weekSpan", "dayType"),
                                 class = c(rep("numeric", 4), "character"),
                                 label = c("Lambda", "Loess Span", "Loess Degree", "Week Span", "Day Type"))

# Define the required grid function, which is used to create the tuning grid (unless the user 
# gives the exact values of the parameters via tuneGrid)
cFBasic$grid <- function(x, y, len = NULL, search = "grid") {
  lambda <- seq(2, 5, 0.1) # like in the book
  span <- c(0.05, 0.1, 0.3, 0.5, 0.75)
  degree <- c(1, 2)
  weekSpan <- c(1, 2, 3, 4, 5, 6, 7)
  dayType <- c("dayOfTheWeek", "dayOfTheMonth")
  
  # to use grid search
  out <- expand.grid(lambda = lambda,
                     span = span,
                     degree = degree,
                     weekSpan = weekSpan,
                     dayType = dayType)
  
  if(search == "random") {
    # random search simply random samples from the expanded grid
    out <- out %>%
      sample_n(100)
  }
  out
}

# Define the fit function so we can fit our model to the data
cFBasic$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  # check whether we have a correct x
  stopifnot("userId" %in% colnames(x))
  stopifnot("movieId" %in% colnames(x))
  stopifnot("timestamp" %in% colnames(x))
  stopifnot("genres" %in% colnames(x))
  stopifnot("rating" %in% colnames(x))
  stopifnot("min_ts" %in% colnames(x))
  stopifnot(all(x$rating == y))
  
  # compute global mean
  mu <- mean(x$rating)
  
  # compute movie effects b_i
  movie_avgs <- x %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n() + param$lambda))
  
  # compute user effects b_u
  user_avgs <- x %>%
    left_join(movie_avgs, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + param$lambda))
  
  # compute genre effects b_g
  genre_avgs <- x %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - (mu + b_i + b_u))/(n() + param$lambda))
  
  # add the day feature to model temporal effects
  if (param$dayType == "dayOfTheWeek") {
    x <- x %>% 
      mutate(day = as.factor(wday(as_datetime(timestamp))))
  } else {
    stopifnot(param$dayType == "dayOfTheMonth")
    x <- x %>% 
      mutate(day = as.factor(day(round_date(as_datetime(timestamp), unit = "day"))))
  }
  
  # compute the day effects b_d
  stopifnot("day" %in% colnames(x))
  day_avgs <- x %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    group_by(day) %>%
    summarize(b_d = sum(rating - (mu + b_i + b_u + b_g))/(n() + param$lambda))
  
  # add the week feature to model temporal effects
  x <- x %>%
    mutate(week = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(param$weekSpan)))
  
  stopifnot("week" %in% colnames(x))
  week_fit <- x %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    left_join(day_avgs, by='day') %>%
    group_by(week) %>%
    summarise(rating_residual=mean(rating - (mu + b_i + b_u + b_g + b_d))) %>%
    loess(rating_residual~week, data=., span=param$span, degree=param$degree)
  
  # return the model fit as a list
  list(mu=mu,
       movie_avgs=movie_avgs, 
       user_avgs=user_avgs, 
       genre_avgs=genre_avgs, 
       day_avgs=day_avgs, 
       week_fit=week_fit,
       params=param)
}

# Define the predict function that produces a vector of predictions
cFBasic$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
  # add the day feature to model temporal effects
  if (modelFit$params$dayType == "dayOfTheWeek") {
    newdata <- newdata %>% 
      mutate(day = as.factor(wday(as_datetime(timestamp))))
  } else {
    stopifnot(modelFit$params$dayType == "dayOfTheMonth")
    newdata <- newdata %>%
      mutate(day = as.factor(day(round_date(as_datetime(timestamp), unit = "day"))))
  }
  
  # add the week feature to model temporal effects
  newdata <- newdata %>%
    mutate(week = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(modelFit$params$weekSpan)))
  
  stopifnot("day" %in% colnames(newdata))
  stopifnot("week" %in% colnames(newdata))
  predicted <- newdata %>%
    left_join(modelFit$movie_avgs, by='movieId') %>%
    left_join(modelFit$user_avgs, by='userId') %>%
    left_join(modelFit$genre_avgs, by='genres') %>%
    left_join(modelFit$day_avgs, by='day') %>%
    mutate(pred = modelFit$mu + b_i + b_u + b_g + b_d + predict(modelFit$week_fit, .)) %>%
    pull(pred)
  
  predicted
}

##########################################################################################
## Build the calibration (cross validation) data set subset of the training set.
##########################################################################################

tic("collecting the calibration set of 2k samples, making sure it's a bit dense")
# set the seed again
portable.set.seed(1)
calibration_set <- train_set %>%
  group_by(movieId) %>%
  filter(n() > 3600) %>%
  ungroup() %>%
  group_by(userId) %>%
  filter(n() > 400) %>%
  ungroup() %>%
  sample_n(2000)
toc()

# how many distinct movies and users in the calibration set?
cat(sprintf("The calibration set contains %d unique movies and %d unique users\n", 
            length(unique(calibration_set$movieId)), length(unique(calibration_set$userId))))

##########################################################################################
## Calibrate the CF BASIC model on the calibration set (subset of the training set). Here
## I look for the best hyper-parameters that fit the basic model on a small subset of the 
## training data.
##########################################################################################

# register number of cores for parallelizing the calibration or cross validation
registerDoMC(6)

cat(sprintf('The BASIC tunning grid contains %d hyper-parameter permutations.', nrow(cFBasic$grid())))

tic('BASIC - calibrating the model')
# set the seed again
portable.set.seed(1)
control <- trainControl(method = "cv",
                        search = "grid",
                        number = 10,
                        p = .9,
                        allowParallel = TRUE,
                        verboseIter = TRUE)
calFitCFBasic <- train(x = calibration_set,
                       y = calibration_set$rating,
                       method = cFBasic,
                       trControl = control)
toc()
## The bestTune model found is:
stopifnot(calFitCFBasic$bestTune$lambda == 5)
stopifnot(calFitCFBasic$bestTune$span == 0.3)
stopifnot(calFitCFBasic$bestTune$degree == 2)
stopifnot(calFitCFBasic$bestTune$weekSpan == 2)
stopifnot(calFitCFBasic$bestTune$dayType == "dayOfTheWeek")

##########################################################################################
## Fit the best model found to the complete training set
##########################################################################################

tic('BASIC - training the model on the full training set')
fitCFBasic <- train(x = train_set,
                    y = train_set$rating,
                    method = cFBasic,
                    trControl = trainControl(method = "none"),
                    tuneGrid = calFitCFBasic$bestTune)
toc()

##########################################################################################
## Finally test the BASIC model on the final test (a.k.a. validation) set
##########################################################################################

## VALIDATION SET ACCESS ALERT! accessing the validation set to compute RMSE.
tic('BASIC - predicting ratings')
predicted_ratings <- predict(fitCFBasic, validation_set)
rmse_val <- RMSE(predicted_ratings, validation_set$rating)
toc()
cat(sprintf("BASIC - RMSE on validation data is %.9f", rmse_val))
# check that we get a reproducible result
stopifnot(abs(rmse_val - 0.864080283) < 1e-9)

##########################################################################################
## Create the CF ADVANCED model integrated with the caret package. The model is called cfAdv
## standing for Collaborative Filtering advanced method which builds on top of the basic 
## model and employs low-rank matrix factorization trained using SGD.
##
## This caret model specification follows the implementation details described here:
## https://topepo.github.io/caret/using-your-own-model-in-train.html
##########################################################################################

# free a bit of memory if possible
gc(TRUE, TRUE, TRUE)

# Define the model cFAdv (Collaborative Filtering advanced)
cFAdv <- list(type = "Regression",
              library = c("lubridate", "stringr"),
              loop = NULL,
              prob = NULL,
              sort = NULL)

# Four different parameters are supported:
# @param K the number of latent dimensions
# @param gamma the learning rate
# @param lambda the regularizaton parameter applied to the different effects
# @param sigma the standard deviation of the initial values
#
cFAdv$parameters <- data.frame(parameter = c("K", "gamma", "lambda", "sigma"),
                               class = c(rep("numeric", 4)),
                               label = c("K-Latent Dim", "Learning rate", "Lambda", "Sigma of Init Values"))

# Define the required grid function, which is used to create the tuning grid (unless the user 
# gives the exact values of the parameters via tuneGrid)
cFAdv$grid <- function(x, y, len = NULL, search = "grid") {
  K <- 2:3
  gamma <- c(0.02, 0.04, 0.06, 0.08, 0.1)
  lambda <- c(10^seq(-2, -1), 5*10^seq(-2, -1))
  sigma <- c(0.05, 0.1)

  # to use grid search
  out <- expand.grid(K = K,
                     gamma = gamma,
                     lambda = lambda,
                     sigma = sigma)
  
  if(search == "random") {
    # random search simply random samples from the expanded grid
    out <- out %>%
      sample_n(100)
  }
  out
}

# Define the fit function so we can fit our model to the data
# NOTE: the fit function requires the CF BASIC model as extended parameter argument
# @param P the initial P matrix.
# @param Q the initial Q matrix.
# @param maxIter maximum number of iterations or random samples.
# @param trackConv whether to track RMSE convergence of the algorithm.
# @param perTrack percent of samples to track for RMSE convergence.
# @param iterBreaks number of steps before checking for convergence.
# @param fitCFBasic the BASIC CF fit model.
cFAdv$fit <- function(x, y, wts, param, lev, last, weights, classProbs, P=NULL, Q=NULL, 
                      maxIter=1500, trackConv=FALSE, perTrack=0.01, iterBreaks=100, fitCFBasic, ...) {
  # check whether we have a correct x
  stopifnot("userId" %in% colnames(x))
  stopifnot("movieId" %in% colnames(x))
  stopifnot("timestamp" %in% colnames(x))
  stopifnot("genres" %in% colnames(x))
  stopifnot("rating" %in% colnames(x))
  stopifnot("min_ts" %in% colnames(x))
  stopifnot(all(x$rating == y))
  
  # read model information from the CF BASIC fit
  mu  <- fitCFBasic$finalModel$mu
  user_avgs  <- fitCFBasic$finalModel$user_avgs
  movie_avgs <- fitCFBasic$finalModel$movie_avgs
  genre_avgs <- fitCFBasic$finalModel$genre_avgs
  week_fit <- fitCFBasic$finalModel$week_fit
  
  K <- param$K          # number of latent dimensions
  N <- nrow(user_avgs)  # number of users
  M <- nrow(movie_avgs) # number of movies
  
  # randomly initialize P and Q
  if (is.null(P)) P <- matrix(rnorm(K*N, sd=param$sigma), K, N)
  if (is.null(Q)) Q <- matrix(rnorm(K*M, sd=param$sigma), K, M)
  
  # ensure that the columns dimension match the number of distinct 
  # users and movies
  stopifnot(ncol(P) == N)
  stopifnot(ncol(Q) == M)
  
  # identify rows by user or movie respectively. Note that this is the 
  # lookup method to match users to P and movies to Q using the userId
  # or movieId as column name key.
  colnames(P) <- user_avgs$userId
  colnames(Q) <- movie_avgs$movieId

  computeRMSE <- function(subset_samples) {
    # compute the user-movie interaction effects contained in P and Q
    pq_effects <- subset_samples %>%
      group_by(userId, movieId) %>%
      mutate(pq=(P[,u]%*%Q[,i])[1]) %>%
      select(userId, movieId, pq)

    # compute the predictions    
    predicted <- subset_samples %>%
      left_join(pq_effects, by=c('userId', 'movieId')) %>%
      mutate(predicted=mu + b_i + b_u + b_g + b_w + pq) %>% 
      pull(predicted)
    
    if(any(is.na(predicted))) {
      stop(sprintf("train - na detected for K=%d, gamma=%.6f, lambda=%.6f, sigma=%.6f", 
                  param$K, param$gamma, param$lamda, param$sigma))
    }
    rmse_val <- RMSE(predicted, subset_samples$rating)
  }
  
  # add the week feature to model temporal effects
  x <- x %>%
    mutate(week = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(fitCFBasic$finalModel$params$weekSpan)))
  
  # select random samples corresponding to the number of iterations parameter
  samples <- x %>%
    left_join(movie_avgs, by='movieId') %>%
    left_join(user_avgs, by='userId') %>%
    left_join(genre_avgs, by='genres') %>%
    mutate(i=as.character(movieId), u=as.character(userId), b_w = predict(week_fit, .), residual=rating - (mu + b_i + b_u + b_g + b_w)) %>%
    sample_n(maxIter)
  
  # use a subset of the samples to test for convergence
  subset_samples <- samples %>% 
    sample_n(nrow(samples)*perTrack)
  
  rmse_val <- computeRMSE(subset_samples)
  
  if (trackConv) {
    rmse_hist <- tibble(k=0, rmse=rmse_val)
  } else {
    rmse_hist <- NULL
  }
  
  for (k in 1:nrow(samples)) {
    i <- as.character(samples[k,]$movieId)
    u <- as.character(samples[k,]$userId)
    
    # compute the residual
    residual <- samples[k,]$residual - (P[,u]%*%Q[,i])[1]
    
    # update the latent vectors
    P[,u] <- (P[,u] + param$gamma*(residual*Q[,i] - param$lambda*P[,u]))
    Q[,i] <- (Q[,i] + param$gamma*(residual*P[,u] - param$lambda*Q[,i]))

    # track convergence every "iterBreaks" steps
    if (trackConv && k %% iterBreaks == 0) {
      # check rmse
      rmse_val <- computeRMSE(subset_samples)
      cat(sprintf('the RMSE at k=%d is %.9f\n', k, rmse_val))
      rmse_hist <- rbind(rmse_hist, tibble(k=k, rmse=rmse_val))
    }
  }

  if(any(is.na(P)) || any(is.na(Q))) {
    stop(sprintf("na detected in P or Q for K=%d, gamma=%.6f, lambda=%.6f, sigma=%.6f", 
                param$K, param$gamma, param$lamda, param$sigma))
  }
  
  # return the model fit as a list
  list(mu=mu,
       user_avgs=user_avgs,
       movie_avgs=movie_avgs,
       week_fit=week_fit,
       genre_avgs=genre_avgs,
       P=P,
       Q=Q,
       rmse_hist=rmse_hist,
       params=c(param, weekSpan=fitCFBasic$finalModel$params$weekSpan))
}

# Define the predict function that produces a vector of predictions
cFAdv$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
  if(any(is.na(modelFit$P)) || any(is.na(modelFit$Q))) {
    stop(sprintf("predict - na detected in P or Q for K=%d, gamma=%.6f, lambda=%.6f, sigma=%.6f", 
                 param$K, param$gamma, param$lamda, param$sigma))
  }
  
  # add the week feature to model temporal effects
  newdata <- newdata %>%
    mutate(week = ceiling(as.duration(as_datetime(min_ts) %--% as_datetime(timestamp)) / dweeks(modelFit$params$weekSpan)))
  
  # compute the user-movie interaction effects contained in P and Q
  pq_effects <- newdata %>%
    group_by(userId, movieId) %>%
    mutate(i=as.character(movieId), u=as.character(userId), pq=(modelFit$P[,u]%*%modelFit$Q[,i])[1]) %>%
    select(userId, movieId, pq)
  
  # compute the predictions    
  predicted <- newdata %>%
    left_join(modelFit$movie_avgs, by='movieId') %>%
    left_join(modelFit$user_avgs, by='userId') %>%
    left_join(modelFit$genre_avgs, by='genres') %>%
    left_join(pq_effects, by=c('userId', 'movieId')) %>%
    mutate(b_w=predict(modelFit$week_fit, .), predicted=modelFit$mu + b_i + b_u + b_g + b_w + pq) %>% 
    pull(predicted)
  
  if(any(is.na(predicted))) {
    stop(sprintf("predict - na detected for K=%d, gamma=%.6f, lambda=%.6f, sigma=%.6f", modelFit$params$K, 
                modelFit$params$gamma, modelFit$params$lamda, modelFit$params$sigma))
  }
  predicted
}

##########################################################################################
## Calibrate the CF ADVANCED model on the calibration set (subset of the training set). 
## Here I look for the best hyper-parameters that fit the advanced model on a small subset  
## of the training data.
##########################################################################################

tic('ADVANCED: calibrating the model')
# set the seed again
portable.set.seed(1)
control <- trainControl(method = "cv",
                        search = "grid",
                        number = 10,
                        p = .9,
                        allowParallel = TRUE,
                        verboseIter = TRUE)
calFitCFAdv <- train(x = calibration_set,
                     y = calibration_set$rating,
                     method = cFAdv,
                     trControl = control,
                     fitCFBasic = calFitCFBasic)
toc()
## The bestTune model found is:
stopifnot(calFitCFAdv$bestTune$K == 2)
stopifnot(calFitCFAdv$bestTune$gamma == 0.06)
stopifnot(calFitCFAdv$bestTune$lambda == 0.1)
stopifnot(calFitCFAdv$bestTune$sigma == 0.1)

##########################################################################################
## Fit the best model found to the complete training set
##########################################################################################

maxIter <- 150000
cat(sprintf("ADVANCED - training on the full edx set using %.2f%% random samples\n", 
            100*maxIter/nrow(train_set)))

tic('ADVANCED: training model on the full trainig data set')
# set the seed again
portable.set.seed(1)
fitCFAdv <- train(x = train_set,
                  y = train_set$rating,
                  method = cFAdv,
                  trControl = trainControl(method = "none"),
                  tuneGrid = calFitCFAdv$bestTune,
                  maxIter = maxIter,
                  trackConv = TRUE,
                  perTrack = 0.003,
                  iterBreaks = 10000,
                  fitCFBasic = fitCFBasic)
toc()

# plot the convergence for the advanced model training on the full edx
fitCFAdv$finalModel$rmse_hist %>%
  ggplot(aes(k, rmse)) + geom_point() + geom_line()

##########################################################################################
## Finally test the ADVANCED model on the final test (a.k.a. validation) set
##########################################################################################

## VALIDATION SET ACCESS ALERT! accessing the validation set to compute RMSE.
tic("ADVANCED: predicting ratings on the full validation set")
predicted_ratings <- predict(fitCFAdv, validation_set)
rmse_val <- RMSE(predicted_ratings, validation_set$rating)
cat(sprintf("ADVANCED - RMSE on validation data is %.9f", rmse_val))
stopifnot(abs(rmse_val - 0.864173163) < 1e-9)
toc()
