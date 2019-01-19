#### Les vecteurs embedding ####

# Chargement 50000 vecteurs de mots 

vec_wiki_news <- read.csv("../../wiki-news-300d-1M.vec", header = FALSE,stringsAsFactors = FALSE,sep = " ",quote= "", nrows = 50000, skip = 1)
vec_cc <- read.csv("../../cc.en.300.vec", header = FALSE,stringsAsFactors = FALSE,sep = " ",quote= "", nrows = 50000, skip = 1)

names(vec_wiki_news) <- NULL
names(vec_cc) <- NULL

#### Fonctions ####

centroid <- function(mots){
  # description: calculer le barycentre des vecteurs de word embedding
  # input: liste de mots
  # ouput: le barycentre 
  
  #trouver l'index des mots entrés dans les vecteurs des mots embedding
  ind_mots <- match(mots,vec_wiki_news[,1])
  
  #calculer la moyenne
  barycentre <- mapply(sum,vec_wiki_news[ind_mots,-1])/as.numeric(length(mots))
  
  return (barycentre)
    
}

#### Tester ####
# centroid()







