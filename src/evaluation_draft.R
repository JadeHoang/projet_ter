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
data.docsplit <- strsplit(data.text, ' ')

data.graph <- data.frame(read.table(graph_filename))
colnames(data.graph) <- c('tar', 'src')


#### Extraction des vecteurs de documents

source('baseline.R')

# Calcul des pondérations tfidf et okapi
tfidf_avg <- weight_embedding(corpus = data.text, ponderation = 'tfidf')
okapi_avg <- weight_embedding(corpus = data.text, ponderation = "okapi")


# load word embedding vectors
file_vec <- "../dataset/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec"
word_embedding <- load_word_embedding_vector(file_vec)


doc_emb_bar <- sapply( data.docsplit, function(doc) document_embedding(word_embedding, doc, methode = "barycentre") )
doc_emb_tfidf <- sapply( data.docsplit, function(doc) document_embedding(word_embedding, doc, methode = "tfidf", weights = tfidf_avg) )
doc_emb_okapi <- sapply( data.docsplit, function(doc) document_embedding(word_embedding, doc, methode = "okapi", weights = okapi_avg) )






#### Tâche d'évaluation: Classification avec plusieurs méthodes de régression
# Création des données d'entrainement et de test
nobs <- length(data.group)
n_sample_train <- round(0.8 * nobs)
n_sample_test <- nobs - n_sample_train
set.seed(123)
training_ind <- sample(nobs, n_sample_train)
testing_ind <- setdiff(1:nobs, training_ind)

prf <- function(truth, pred){
  cont_tab <- table(truth, pred)
  precision <- mean( diag(cont_tab) / colSums(cont_tab), na.rm = TRUE ) 
  recall <- mean( diag(cont_tab) / rowSums(cont_tab), na.rm = TRUE ) 
  f1_score <- 2 * (precision * recall) / (precision + recall) 
  return( list(precision = precision, recall = recall, f1_score = f1_score, tab_err = cont_tab))
}



### Entrainement et test de modèles avec régression logistique et SVM
library(MASS)
library(e1071)



## Vecteurs barycentriques ##
training_data <- doc_emb_bar[, training_ind]
testing_data <- doc_emb_bar[, testing_ind]
training_labels <- data.group[training_ind]
testing_labels <- data.group[testing_ind]

train <- data.frame(x = t(training_data), y = training_labels)
test <- data.frame(x = t(testing_data), y = testing_labels)

# Régression logistique
logistic_bar <- polr(y ~ ., data = train, method = "logistic")
pred_logistic_bar <- predict(logistic_bar, test )
prf_logistic_bar <- prf(test$y, pred_logistic_bar)

# SVM
svm_model_bar = svm(y ~ ., data = train )
pred_svm_bar <- predict(svm_model_bar, t(testing_data) )
prf_svm_bar <- prf(test$y, pred_svm_bar)




## Vecteurs tf-idf ##
training_data <- doc_emb_tfidf[, training_ind]
testing_data <- doc_emb_tfidf[, testing_ind]

train <- data.frame(x = t(training_data), y = training_labels)
test <- data.frame(x = t(testing_data), y = testing_labels)

# Régression logistique
logistic_tfidf <- polr(y ~ ., data = train, method = "logistic")
pred_logistic_tfidf <- predict(logistic_tfidf, test )
prf_logistic_tfidf <- prf(test$y, pred_logistic_tfidf)

# SVM
svm_model_tfidf = svm(y ~ ., data = train )
pred_svm_tfidf <- predict(svm_model_tfidf, t(testing_data) )
prf_svm_tfidf <- prf(test$y, pred_svm_tfidf)




## Vecteurs okapi ##
training_data <- doc_emb_okapi[, training_ind]
testing_data <- doc_emb_okapi[, testing_ind]

train <- data.frame(x = t(training_data), y = training_labels)
test <- data.frame(x = t(testing_data), y = testing_labels)

# Régression logistique
logistic_okapi <- polr(y ~ ., data = train, method = "logistic")
pred_logistic_okapi <- predict(logistic_okapi, test )
prf_logistic_okapi <- prf(test$y, pred_logistic_okapi)

# SVM
svm_model_okapi = svm(y ~ ., data = train )
pred_svm_okapi <- predict(svm_model_okapi, t(testing_data) )
prf_svm_okapi <- prf(test$y, pred_svm_okapi)



