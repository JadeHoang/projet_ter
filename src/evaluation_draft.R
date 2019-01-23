# Evaluation using CORA dataset
# dataset available at:
# https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
# Documentation available at:
# https://relational.fit.cvut.cz/dataset/CORA
# https://linqs.soe.ucsc.edu/data

# Here we use modified cora dataset found at https://github.com/thunlp/CANE/tree/master/datasets/cora

# classes:
    # Case_Based
    # Genetic_Algorithms
    # Neural_Networks
    # Probabilistic_Methods
    # Reinforcement_Learning
    # Rule_Learning
    # Theory


#### Chargement des données

data_filename <- '../dataset/cora_modified/data.txt'
graph_filename <- '../dataset/cora_modified/graph.txt'
group_filename <- '../dataset/cora_modified/group.txt'

data.group <- readLines(group_filename)
data.group <- as.matrix(data.group)
data.group <- factor(data.group)
empty <- which(data.group == '')
data.group <- data.group[-empty]

data.text <- readLines(data_filename, encoding = "utf8")
data.text <- data.text[-empty]
data.docsplit <- strsplit(tolower(data.text), ' ')

data.graph <- data.frame(read.table(graph_filename))
colnames(data.graph) <- c('tar', 'src')


#### Extraction des vecteurs de documents

source('baseline.R')

# Calcul des pondérations tfidf et okapi
tfidf_avg <- weight_embedding(corpus = data.text, ponderation = 'tfidf')
okapi_avg <- weight_embedding(corpus = data.text, ponderation = "okapi")


# load word embedding vectors
file_vec <- "../dataset/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec"
# file_vec <- "../dataset/Common_Crawl_Wikipedia/cc.en.300.vec/data"
# file_vec <- "../dataset/GoogleNews-vectors-negative300.bin/data"
word_embedding <- load_word_embedding_vector("../dataset/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec", nrows = 120000)
we <- read.csv("../dataset/GoogleNews-vectors-negative300.bin/data", 
                 header = FALSE, 
                 stringsAsFactors = FALSE, 
                 sep = " ",
                 quote= "", 
                 nrows = 1000)

doc_emb_bar <- sapply( data.docsplit, function(doc) document_embedding(word_embedding, doc, methode = "barycentre") )
doc_emb_tfidf <- sapply( data.docsplit, function(doc) document_embedding(word_embedding, doc, methode = "tfidf", weights = tfidf_avg) )
doc_emb_okapi <- sapply( data.docsplit, function(doc) document_embedding(word_embedding, doc, methode = "okapi", weights = okapi_avg) )






#### Tâche d'évaluation: Classification avec 2 méthodes de régression (Bayésien Naïf / SVM)
# Création des données d'entrainement et de test
generate_train_test <- function(x, y, training_amount){
  nobs <- length(y)
  n_sample_train <- round(training_amount * nobs)
  n_sample_test <- nobs - n_sample_train
  training_ind <- sample(nobs, n_sample_train)
  test_ind <- setdiff(1:nobs, training_ind)
  
  training_x <- x[, training_ind]
  test_x <- x[, test_ind]
  training_y <- y[training_ind]
  test_y <- y[test_ind]
  
  train <- data.frame(x = t(training_x), y = training_y)
  test <- data.frame(x = t(test_x), y = test_y)
  
  return( list(train = train, test = test))
}


prf <- function(truth, pred){
  cont_tab <- table(truth, pred)
  precision <- mean( diag(cont_tab) / colSums(cont_tab), na.rm = TRUE ) 
  recall <- mean( diag(cont_tab) / rowSums(cont_tab), na.rm = TRUE ) 
  f1_score <- 2 * (precision * recall) / (precision + recall) 
  return( list(precision = precision, recall = recall, f1_score = f1_score, tab_err = cont_tab))
}

mean_prf <- function(x){
  precision_ind <- which(names(x) == "precision")
  recall_ind <- which(names(x) == "recall")
  f1_score_ind <- which(names(x) == "f1_score")
  mean_precision <- mean(x[precision_ind]$precision)
  mean_recall <- mean(x[recall_ind]$recall)
  mean_f1_score <- mean(x[f1_score_ind]$f1_score)
  return( list( precision = mean_precision, recall = mean_recall, f1_score = mean_f1_score))
}


### Entrainement et test de modèles avec Bayésien Naïf et SVM
library(MASS)
library(e1071)


prf_nb_bar <- c()
prf_svm_bar <- c()
for(i in 1:10){
  ## Vecteurs barycentriques ##
  dataset <- generate_train_test(doc_emb_bar, data.group, 0.1)
  
  # Bayésien Naïf
  nb_bar <- naiveBayes(y ~ ., data = dataset$train)
  pred_nb_bar <- predict(nb_bar, dataset$test)
  prf_nb_bar <- c(prf_nb_bar, prf(dataset$test$y, pred_nb_bar))
  
  # SVM
  svm_model_bar = svm(y ~ ., data = dataset$train )
  pred_svm_bar <- predict(svm_model_bar, dataset$test )
  prf_svm_bar <- c(prf_svm_bar, prf(dataset$test$y, pred_svm_bar))
}
mean_prf_nb_bar <- mean_prf(prf_nb_bar)
mean_prf_svm_bar <- mean_prf(prf_svm_bar)



prf_nb_tfidf <- c()
prf_svm_tfidf <- c()
for(i in 1:10){
  ## Vecteurs tf-idf ##
  dataset <- generate_train_test(doc_emb_tfidf, data.group, training_amount = 0.1)
  
  # Bayésien Naïf
  nb_tfidf <- naiveBayes(y ~ ., data = dataset$train)
  pred_nb_tfidf <- predict(nb_tfidf, dataset$test )
  prf_nb_tfidf <- c(prf_nb_tfidf, prf(dataset$test$y, pred_nb_tfidf) )
  
  # SVM
  svm_model_tfidf = svm(y ~ ., data = dataset$train )
  pred_svm_tfidf <- predict(svm_model_tfidf, dataset$test )
  prf_svm_tfidf <- c(prf_svm_tfidf, prf(dataset$test$y, pred_svm_tfidf) )
}
mean_prf_nb_tfidf <- mean_prf(prf_nb_tfidf)
mean_prf_svm_tfidf <- mean_prf(prf_svm_tfidf)


prf_nb_okapi <- c()
prf_svm_okapi <- c()
for(i in 1:10){
  ## Vecteurs okapi ##
  dataset <- generate_train_test(doc_emb_okapi, data.group, training_amount = 0.1)
  
  # Bayésien Naïf
  nb_okapi <- naiveBayes(y ~ ., data = dataset$train)
  pred_nb_okapi <- predict(nb_okapi, dataset$test )
  prf_nb_okapi <- c( prf_nb_okapi, prf(dataset$test$y, pred_nb_okapi) )
  
  # SVM
  svm_model_okapi = svm(y ~ ., data = dataset$train )
  pred_svm_okapi <- predict(svm_model_okapi, dataset$test )
  prf_svm_okapi <- c( pred_svm_okapi, prf(dataset$test$y, pred_svm_okapi) )
}
mean_prf_nb_okapi <- mean_prf(prf_nb_okapi)
mean_prf_svm_okapi <- mean_prf(prf_svm_okapi)

