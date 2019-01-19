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


# Chargement des données

data_filename <- './dataset/cora_modified/data.txt'
graph_filename <- './dataset/cora_modified/graph.txt'
group_filename <- './dataset/cora_modified/group.txt'

data.text <- readLines(data_filename, encoding = "utf8")
data.graph <- data.frame(read.table(graph_filename))
colnames(data.graph) <- c('tar', 'src')
data.group <- readLines(group_filename)

# Extraction des vecteurs de Documents Embedding
N <- 50 # size embedding
nobs <- length(data.text)
d <- matrix(runif(nobs * N), ncol = N)

# data.group <- data.frame( round(runif(nobs, min = 0, max = 6)) )
data.group <- as.matrix(data.group)
data.group <- factor(data.group)


library(MASS)
logistic <- polr(data.group ~ d, method = "logistic")



