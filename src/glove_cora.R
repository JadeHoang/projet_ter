library(text2vec)

data_filename <- "dataset/cora_modified/data.txt"
graph_filename <- 'dataset/cora_modified/graph.txt'
group_filename <- 'dataset/cora_modified/group.txt'

data.group <- readLines(group_filename)
data.group <- as.matrix(data.group)
data.group <- factor(data.group)
empty <- which(data.group == '') # On relève les indices des cases vides
data.group <- data.group[-empty] # On enlève les données vides

#charger les documents
data.text <- readLines(data_filename, encoding = "utf8")
# On enlève les données documents correspondant aux groupes non renseignés
data.text <- data.text[-empty]
#minuscule
data.text <- tolower(data.text)

text8 <- "../../Word Embedding/text8"

text8 <- readLines(text8, n = 1, encoding = "utf8", warn = FALSE)

data.text <- c(data.text, text8)

#créer iterateur
tokens <- space_tokenizer(data.text)

#créer vocabulaire
it <- itoken(tokens, progresbar = FALSE)
vocab <- create_vocabulary(it)

#ne garder que des mots qui existent au moins 5 fois
vocab <- prune_vocabulary(vocab, term_count_min = 5L)

#utilise le vocab
vectorizer <- vocab_vectorizer(vocab)
#fenetre de 5 mots
tcm <- create_tcm(it, vectorizer, skip_grams_window = 3L)

#glove
glove <- GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
glove$fit_transform(tcm, n_iter = 20)

word_vectors <- glove$get_word_vectors()

