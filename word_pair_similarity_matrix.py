# A series of functions designed to quickly and painlessly create a word-word similarity matrix in a hyperdimensional LSA/LSI space.

# Includes three separate functions that can be used in order:
    # create_lsi_lookup_table: Create an LSI lookup table for later use in calculating word-word similarity scores
    # create_similarity_matrix: Create a matrix of word-word pairs, complete with word, location in LSI dictionary, and hyperdimensional vector
    # calculate_similarity_matrix: Calculate pairwise distance scores; implement with np.apply_along_axis (axis=1)

# Includes an additional function that combines all three of the earlier functions without intervening steps:
    # all_in_one_similiarity_matrix

# Written by: A. Paxton
# Date last modified: 24 November 2015

import pandas as pd
import scipy.spatial.distance as dist
import gensim, itertools
import numpy as np
from ast import literal_eval

####
        
# create an LSI lookup table for later use in calculating word-word similarity scores
def create_lsi_lookup_table(unique_word_vector,target_dictionary,target_lsi_model,nd):
    
    # unique_word_vector: 1D string vector to include in lookup table
    # target_dictionary: gensim-created dictionary that must contain all words in unique_word_vector
    # target_lsi_model: gensim-created LSI model generated with target_dictionary
    # nd: number of dimensions used in LSI creation

    # create lookup table from word vector
    lookup_table = pd.DataFrame(unique_word_vector,columns=['word'])
    
    # create a function to look up the dictionary location and return as tuple-string
    def word_lookup(word_vector,dictionary):
        return str(dictionary.doc2bow([word_vector[0]]))

    # apply new function to find words, then convert from tuple to 
    lookup_table['loc'] = np.apply_along_axis(word_lookup,1,lookup_table,target_dictionary)
    lookup_table['loc'] = lookup_table['loc'].replace('\]|\[','',regex=True)
    lookup_table['loc'] = lookup_table['loc'].apply(literal_eval)

    # look up hyperdimensional vectors for each word and add them to the lookup table
    lookup_table['hdv'] = 0
    lookup_table['hdv'] = lookup_table['hdv'].astype(object)
    for next_word in lookup_table['word']:
        next_location = lookup_table['loc'].loc[lookup_table['word']==next_word].index[0]
        next_hdv = np.array([val for (dim, val) in target_lsi_model[[lookup_table['loc'].loc[next_location]]]], dtype=object)
        if len(next_hdv) == nd:
            lookup_table['hdv'].loc[next_location] = next_hdv
        else:
            lookup_table['hdv'].loc[next_location] = 0
            lookup_table['loc'].loc[next_location] = 0

    # remove any lines that didn't have the full nd dimensions and then reset the index
    lookup_table = lookup_table.loc[-(lookup_table['loc']==0)].reset_index().drop(['index'],1)

    # spit out lookup table
    return lookup_table

####

# create a matrix of word-word pairs, complete with word, location in LSI dictionary, and hyperdimensional vector
def create_similarity_matrix(unique_word_vector, lookup_table, word_varname, loc_varname, hdv_varname, winnow_variable, ascending):

    # unique_word_vector: 1D string vector over which to calculate all possible pairwise similarity scores
    # lookup_table: pandas dataframe with (minimally) all words in word_list and their hyperdimensional value
    # word_varname: name of column in lookup_table dataframe that includes words
    # hdv_varname: name of column in lookup_table dataframe that includes hyperdimensional vectors 
    # loc_varname: name of column in lookup_table dataframe that includes dictionary location
    # winnow_variable: 1D vector of strings that must be present in a pair to be included in final matrix; set to [] if not desired
    # ascending: (optionally) sort ascending in 'word1' and then 'word2'
        
    # make non-repeating word pairs
    word_pairs = pd.DataFrame(list(itertools.combinations(unique_word_vector,2)),columns=['word1','word2'])

    # create duplicates of the lookup table for the 'word1' and 'word2' columns of word_pairs
    lt1 = lookup_table
    lt1 = lt1.rename(columns={word_varname:'word1',loc_varname:'loc1',hdv_varname:'hdv1'})
    lt2 = lookup_table
    lt2 = lt2.rename(columns={word_varname:'word2',loc_varname:'loc2',hdv_varname:'hdv2'})

    # merge both with word_pairs
    first_merge = pd.merge(word_pairs, lt1, how='inner')
    word_pairs = pd.merge(first_merge, lt2,how='inner')

    # reduce word_pairs so that one member of pair must be a member of target word group
    if len(winnow_variable) > 0:
        target_word_matrix1 = word_pairs.loc[word_pairs['word1'].isin(winnow_variable)]
        target_word_matrix2 = word_pairs.loc[word_pairs['word2'].isin(winnow_variable)]
        word_pairs = target_word_matrix1.append(target_word_matrix2)

    # if desired, sort ascending
    if ascending==True:
        word_pairs = word_pairs.sort_values(by=['word1','word2'],ascending=True)

    # reset the index and drop old index column
    word_pairs = word_pairs.reset_index().drop(['index'],1)

    # when we're done, spit out the completed pandas dataframe
    return word_pairs

####

# calculate pairwise distance scores; implement with np.apply_along_axis (axis=1)
def calculate_similarity_matrix(similarity_matrix,first_hdv_column_loc,second_hdv_column_loc):

    # similarity_matrix: minimially, a pandas dataframe with 2 columns of np.arrays with hyperdimensional vectors
    # first_hdv_column_loc: location of column in similarity_matrix of first hyperdimensional vector
    # second_hdv_column_loc: location of column in similarity_matrix of second hyperdimensional vector

    # grab separate words from input vector
    loc1 = similarity_matrix[first_hdv_column_loc]
    loc2 = similarity_matrix[second_hdv_column_loc]
    
    # cycle through each word pair and calculate the similarity
    return -1*dist.cosine(loc1,loc2)+1 # thanks to Rick Dale for this snippet


####

# implement all three of the earlier functions in a single function that will only spit out final similarity matrix 
def all_in_one_similiarity_matrix(unique_word_vector,target_dictionary,target_lsi_model,nd,winnow_variable,ascending):

    # unique_word_vector: 1D string vector to include in lookup table
    # target_dictionary: gensim-created dictionary that must contain all words in unique_word_vector
    # target_lsi_model: gensim-created LSI model generated with target_dictionary
    # nd: number of dimensions used in LSI creation
    # winnow_variable: 1D vector of strings that must be present in a pair to be included in final matrix; set to [] if not desired
    # ascending: (optionally) sort ascending in 'word1' and then 'word2'

    import pandas as pd
    import scipy.spatial.distance as dist
    import gensim, itertools
    import numpy as np

    ###

    # STEP ONE: create LSI lookup table (standalone: create_lsi_lookup_table)

    # create lookup table from word vector
    lookup_table = pd.DataFrame(unique_word_vector,columns=['word'])
    
    # create a function to look up the dictionary location and return as tuple-string
    def word_lookup(word_vector,dictionary):
        return str(dictionary.doc2bow([word_vector[0]]))

    # apply new function to find words, then convert from tuple to 
    lookup_table['loc'] = np.apply_along_axis(word_lookup,1,lookup_table,target_dictionary)
    lookup_table['loc'] = lookup_table['loc'].replace('\]|\[','',regex=True)
    lookup_table['loc'] = lookup_table['loc'].apply(literal_eval)

    # look up hyperdimensional vectors for each word and add them to the lookup table
    lookup_table['hdv'] = 0
    lookup_table['hdv'] = lookup_table['hdv'].astype(object)
    for next_word in lookup_table['word']:
        next_location = lookup_table['loc'].loc[lookup_table['word']==next_word].index[0]
        next_hdv = np.array([val for (dim, val) in target_lsi_model[[lookup_table['loc'].loc[next_location]]]], dtype=object)
        if len(next_hdv) == nd:
            lookup_table['hdv'].loc[next_location] = next_hdv
        else:
            lookup_table['hdv'].loc[next_location] = 0
            lookup_table['loc'].loc[next_location] = 0

    # remove any lines that didn't have the full nd dimensions and then reset the index
    lookup_table = lookup_table.loc[-(lookup_table['loc']==0)].reset_index().drop(['index'],1)

    ###

    # STEP TWO: assemble first part of similarity matrix (standalone: create_similarity_matrix)
    # make non-repeating word pairs
    word_pairs = pd.DataFrame(list(itertools.combinations(unique_word_vector,2)),columns=['word1','word2'])

    # create duplicates of the lookup table for the 'word1' and 'word2' columns of word_pairs
    lt1 = lookup_table
    lt1 = lt1.rename(columns={'word':'word1','loc':'loc1','hdv':'hdv1'})
    lt2 = lookup_table
    lt2 = lt2.rename(columns={'word':'word2','loc':'loc2','hdv':'hdv2'})

    # merge both with word_pairs
    first_merge = pd.merge(word_pairs, lt1, how='inner')
    word_pairs = pd.merge(first_merge, lt2,how='inner')

    # reduce word_pairs so that one member of pair must be a member of target word group
    if len(winnow_variable) > 0:
        target_word_matrix1 = word_pairs.loc[word_pairs['word1'].isin(winnow_variable)]
        target_word_matrix2 = word_pairs.loc[word_pairs['word2'].isin(winnow_variable)]
        word_pairs = target_word_matrix1.append(target_word_matrix2)

    # if desired, sort ascending
    if ascending==True:
        word_pairs = word_pairs.sort_values(by=['word1','word2'],ascending=True)

    # reset the index and drop old index column
    word_pairs = word_pairs.reset_index().drop(['index'],1)

    ###

    # STEP THREE: calculate pairwise distance scores (standalone: calculate_similarity_matrix)

    # see notes in calculate_similarity_matrix for description
    def calculate_similarity_matrix(similarity_matrix,first_hdv_column_loc,second_hdv_column_loc):
        loc1 = similarity_matrix[first_hdv_column_loc]
        loc2 = similarity_matrix[second_hdv_column_loc]
        return -1 * dist.cosine(loc1,loc2) + 1 # thanks to Rick Dale for this snippet
    word_pairs['cosine'] = np.apply_along_axis(calculate_similarity_matrix,1,word_pairs,3,5)

    ###

    # when we're done, spit out the completed dataframe
    return word_pairs

