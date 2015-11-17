# designed to be applied over word-pair vectors using np.apply_along_axis

# preliminaries
import gensim

# define our function
def w2v_vecsim(word_vector2d,target_w2vmodel):
    # word_vector2d: a two-word vector (one word per column)
    # target_w2vmodel: a gensim-created word2vec model
        
    # if they are, compare them with gensim's built-in similarity evalutator
    return target_w2vmodel.similarity(word_vector2d[0],word_vector2d[1])