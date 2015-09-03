# preliminaries
import pandas as pd
import scipy
import gensim

# define our function
def wordsim(word1,word2,target_dictionary,target_lsi_model):
    
    # look up each word in the LSA/LSI model of choice
    vec_bow1 = target_dictionary.doc2bow([word1])
    vec_bow2 = target_dictionary.doc2bow([word2])
    
    # make sure both words are actually in the topic space
    if len(vec_bow1) > 0 and len(vec_bow2) > 0:
        
        # if they are, go ahead and find their values in the "num_topic"-dimensional space created from gensim.models.LsiModel
        vec_lsi1 = pd.DataFrame(target_lsi_model[vec_bow1],columns=['dim','val'])
        vec_lsi2 = pd.DataFrame(target_lsi_model[vec_bow2],columns=['dim','val'])
        return -1*scipy.spatial.distance.cosine(vec_lsi1['val'],vec_lsi2['val'])+1 # snippet from Rick Dale 
    
    # if the word isn't in the topic space, kick back an error
    else:
        raise RuntimeError('Word pair not found in topic space: '+str(word1)+','+str(word2)+'.')