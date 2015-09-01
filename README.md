# Gensim-LSI-Word-Similarities
Two simple little functions to create word-word similarities from Gensim's latent semantic indexing in Python. Both functions produce an inverted cosine similarity score (0 = low, 1 = high) between two words in a Gensim-generated LSA/LSI space across the total number of dimensions specified in the creation of the model (i.e., <i>num_topics</i> from <i>gensim.models.LsiModel</i>).

<p>Both require Gensim, Pandas, and SciPy.

<p>
<ul>Includes two versions:
<li><b>wordsim</b>: Create cosine-derived similarity score (from 0-1) between individual words. Input: <i>word1</i> (string or string variable), <i>word2</i> (string or string variable), <i>target_dictionary</i> (Gensim-created LSI dictionary), and <i>target_lsi_model</i> (Gensim-created LSI model).
<li><b>wordvectsim</b>: Same as <i>wordvect</i> but created to iterate over a vector of word pairs (e.g., using <i>numpy.apply_along_axis</i>). Create cosine-derived similarity score (from 0-1) between 2D word-pair vector. Input: <i>word_vector2d</i> (2D string vector or 2D string vector variable), <i>target_dictionary</i> (Gensim-created LSI dictionary), and <i>target_lsi_model</i> (Gensim-created LSI model).