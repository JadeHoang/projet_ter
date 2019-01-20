# Fonctions pour le document embedding
# Baseline: combinaison linéaire des vecteurs d'embedding des mots du document
#         - barycentre: moyenne des vecteurs
#         - tfidf     : pondération des vecteurs par tfidf
#         - okapi     : pondération des vecteurs par okapi

load_word_embedding_vector <- function(filename, header = FALSE , stringsAsFactors = FALSE, sep = " "  ,quote= "", nrows = 50000, skip = 1){
  word_vec <- read.csv(filename, header = header ,stringsAsFactors = FALSE,
                       sep = sep ,quote= "", nrows = nrows, skip = skip)
  names(word_vec) <- NULL
  return( list( vec = word_vec[, -1], vocabulary = word_vec[, 1]) )
}

weight_embedding <- function(corpus, ponderation, k = 2 , b = 0.75){
  require(udpipe)
  
  # description: calculer les ponderation tf idf et okapi
  # input:
  # - corpus
  # - ponderation: type de ponderation : tf-idf ou okapi
  # - k et b: les paramètres de Okapi
  # output: selon le type de ponderation, les moyennes des tf-idf ou Okapi par termes
  x <- document_term_frequencies(x = corpus, split = " ")
  x <- document_term_frequencies_statistics(x, k, b)
  
  if (ponderation == "tfidf"){
    return(x[, c("term","tf_idf")])
  }else if (ponderation == "okapi"){
    return(x[, c("term","bm25")])
  }else message("Entrez le type de ponderation.")
}

document_embedding <- function(embedding, document, methode = "barycentre", weights = NA){
  # description: calculer le barycentre des vecteurs de word embedding
  # input: liste de mots, vecteur de word embedding, methode = c("barycentre","tfidf","okapi")
  # ouput: la moyenne
  
  word_vec <- embedding$vec
  vocabulary <- embedding$vocabulary

  #tokens
  mots <- unlist(strsplit(document, split = " "))
  
  #trouver l'index des mots entrés dans les vecteurs des mots embedding
  ind_mots <- na.omit(match(mots , vocabulary))
  N <- length(ind_mots) # Tous les mots du document ne figurent pas dans l'embedding
  
  if (methode == "barycentre") {
    barycentre <- mapply(sum, word_vec[ind_mots, ]) / N
    return (barycentre)
    
  }else if (methode == "tfidf") {
    
    # words_in_vec <- as.vector(word_vec[ind_mots,1])
    # pond <- dtm_tfidf[num_doc , match(words_in_vec[,1],colnames(dtm_tfidf))]
    # tfidf <- colSums(pond * as.matrix(word_vec[ind_mots,-1]))/as.numeric(length(pond))
    
    words_in_vec <- as.vector( vocabulary[ind_mots] )
    pond <- as.numeric( as.matrix( weights[match(words_in_vec, weights$term), "tf_idf"] ) )
    Z <- sum(pond)
    tfidf <- colSums(pond * as.matrix( word_vec[ind_mots, ] ) ) / Z
    return (tfidf)
    
  }else if (methode == "okapi"){
    words_in_vec <- as.vector( vocabulary[ind_mots] )
    pond <- as.numeric(as.matrix(weights[match(words_in_vec, weights$term), "bm25"]))
    Z <- sum(pond)
    okapi <- colSums(pond * as.matrix(word_vec[ind_mots,-1])) / Z
    return (okapi)
  }
}

#### Tester ####

# # Chargement des données du corpus de documents
# data.text <- readLines("../dataset/cora_modified/data.txt",encoding = "utf8")
# 
# tfidf_avg <- weight_embedding(corpus = data.text, ponderation = 'tfidf')
# okapi_avg <- weight_embedding(corpus = data.text, ponderation = "okapi")
# 
# 
# # load word embedding vectors
# file_vec <- "../dataset/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec"
# word_embedding <- load_word_embedding_vector(file_vec)
# 
# 
# emb_doc_bar <- document_embedding(word_embedding, doc1, methode = "barycentre")
# emb_doc_tfidf <- document_embedding(word_embedding, doc1, methode = "tfidf", weights = tfidf_avg)
# emb_doc_okapi <- document_embedding(word_embedding, doc1, methode = "okapi", weights = okapi_avg)
# 
# 
# doc_emb_bar <- sapply( data.text, function(doc) document_embedding(word_embedding, doc, methode = "barycentre"))
# doc_emb_tfidf <- sapply( data.text, function(doc) document_embedding(word_embedding, doc, methode = "tfidf"), weights = tfidf_avg)
# doc_emb_okapi <- sapply( data.text, function(doc) document_embedding(word_embedding, doc, methode = "okapi"), weights = okapi_avg)




