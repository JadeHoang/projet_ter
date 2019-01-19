# install.packages("text2vec")
# install.packages("udpipe")

#### Les vecteurs embedding ####

# Chargement dataset CORA
data.text <- readLines("../../data.txt",encoding = "utf8")

# Chargement 50000 vecteurs de mots 

# vec_wiki_news <- read.csv("../../wiki-news-300d-1M.vec", header = FALSE,stringsAsFactors = FALSE,sep = " ",quote= "", nrows = 50000, skip = 1)
# vec_cc <- read.csv("../../cc.en.300.vec", header = FALSE,stringsAsFactors = FALSE,sep = " ",quote= "", nrows = 50000, skip = 1)
# 
# names(vec_wiki_news) <- NULL
# names(vec_cc) <- NULL

#### Fonctions ####

#avec package udpipe
pond_udpipe <- function(corpus,ponderation, k = 2 , b = 0.75){
  library(udpipe)
  
  # description: calculer les ponderation tf idf et okapi
  # input:
  # - corpus
  # - ponderation: type de ponderation : tf-idf ou okapi
  # - k et b: les paramètres de Okapi
  # output: selon le type de ponderation, les moyennes des tf-idf ou Okapi par termes
  x <- document_term_frequencies(x = corpus, split = " ")
  
  x <- document_term_frequencies_statistics(x,k,b)
  
  if (ponderation == "tfidf"){
    
    return(x[,c("term","tf_idf")])
    
  }else if (ponderation == "okapi"){
    
    return(x[,c("term","bm25")])
    
  }else message("Entrez le type de ponderation.")
}

centroid <- function(file_word_vec, header = F,sep = " ", 
                     nrows = 50000, skip = 1,
                     num_doc, methode = "barycentre"){
  
  # description: calculer le barycentre des vecteurs de word embedding
  # input: liste de mots, vecteur de word embedding, methode = c("barycentre","tfidf","okapi")
  # ouput: la moyenne
  
  #charger le fichier de word embedding
  word_vec <- read.csv(file_word_vec, header = header ,stringsAsFactors = FALSE,
                       sep = sep ,quote= "", nrows = nrows, skip = skip)
  
  names(word_vec) <- NULL
  
  #tokens
  mots <- unlist(strsplit(data.text[num_doc], split = " "))
  
  #trouver l'index des mots entrés dans les vecteurs des mots embedding
  ind_mots <- na.omit(match(mots , word_vec[,1]))
  
  if (methode == "barycentre") {
  
    barycentre <- mapply(sum,word_vec[ind_mots,-1])/as.numeric(length(mots))
    
    return (barycentre)
    
  }else if (methode == "tfidf") {
      
    # words_in_vec <- as.vector(word_vec[ind_mots,1])
    # 
    # pond <- dtm_tfidf[num_doc , match(words_in_vec[,1],colnames(dtm_tfidf))]
    # 
    # tfidf <- colSums(pond * as.matrix(word_vec[ind_mots,-1]))/as.numeric(length(pond))
    
    words_in_vec <- as.vector(word_vec[ind_mots,1])

    pond <- as.numeric(as.matrix(tfidf_avg[match(words_in_vec[,1],tfidf_avg$term), "tf_idf"]))

    tfidf <- colSums(pond * as.matrix(word_vec[ind_mots,-1]))/as.numeric(length(pond))
    
    return (tfidf)
    
  }else if (methode == "okapi"){
    
    words_in_vec <- as.vector(word_vec[ind_mots,1])
    
    pond <- as.numeric(as.matrix(okapi_avg[match(words_in_vec[,1],okapi_avg$term), "bm25"]))
    
    okapi <- colSums(pond * as.matrix(word_vec[ind_mots,-1]))/as.numeric(length(pond))
    
    return (okapi)
    
  }
}

#avec package text2vec
# tf.idf_text2vec <- function(corpus){
#   
#   library(text2vec)
#   # description: calculer la ponderation TF IDF
#   # input: corpus
#   # output: ponderation TF IDF pour chaque mot dans input
#   
#   #processing
#   iterator <- itoken(corpus, tokeniser = space_tokenizer)
#   vocab <- create_vocabulary(iterator)
#   
#   vectorizer <- vocab_vectorizer(vocab)
#   dtm <- create_dtm(iterator, vectorizer)
#   
#   tfidf <- TfIdf$new()
#   dtm_tfidf <- tfidf$fit_transform(dtm)
#   
#   return(list(dtm_tfidf=dtm_tfidf,dtm = dtm))
# }



#### Tester ####

#text2vec
# dtm_tfidf <- tf.idf(corpus = data.text)$dtm_tfidf
# dtm <- tf.idf(corpus = data.text)$dtm

#udpipe
tfidf_avg <- pond_udpipe(corpus = data.text, ponderation = 'tfidf')
okapi_avg <- pond_udpipe(corpus = data.text, ponderation = "okapi")


centroid(file_word_vec = "../../wiki-news-300d-1M.vec", num_doc = 1, methode = "barycentre")
centroid(file_word_vec = "../../wiki-news-300d-1M.vec", num_doc = 1, methode = "tfidf")
centroid(file_word_vec = "../../wiki-news-300d-1M.vec", num_doc = 1, methode = "okapi")









