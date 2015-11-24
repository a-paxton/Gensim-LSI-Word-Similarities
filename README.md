# Gensim-LSI-Word-Similarities
Two simple little functions to create word-word similarities from Gensim's latent semantic indexing in Python. Both functions produce an inverted cosine similarity score (0 = low, 1 = high) between two words in a Gensim-generated LSA/LSI space across the total number of dimensions specified in the creation of the model (i.e., <i>num_topics</i> from <i>gensim.models.LsiModel</i>).

<p>Both require Gensim, Pandas, and SciPy.

<p>Includes four functions:
<ul>
<li><b>wordsim</b>: Create cosine-derived similarity score (from 0-1) between individual words. Input: </li>
	<ul>
	<li><i>word1</i> (string or string variable)</li>
	<li><i>word2</i> (string or string variable)</li>
	<li><i>target_dictionary</i> (Gensim-created LSI dictionary)</li>
	<li><i>target_lsi_model</i> (Gensim-created LSI model)</li>
	</ul>
<br>
<li><b>wordvectsim</b>: Same as <i>wordvect</i> but created to calculate similarity scores (from 0-1) for word pairs in a 2-dimensional word vector (e.g., using <i>numpy.apply_along_axis</i>). Input:</li>
	<ul>
	<li><i>word_vector2d</i> (2D string vector or 2D string vector variable)</li>
	<li><i>target_dictionary</i> (Gensim-created LSI dictionary)</li>
	<li><i>target_lsi_model</i> (Gensim-created LSI model)</li>
	</ul>
<br>
<li>Two additional functions/series of functions added (detailed documentation available in each function and will be added here soon):</li>
	<ul>
	<li><b>word2vec_vect_sim_fun</b>: similarity score function for gensim's word2vec</li>
	<li><b>word_pair_similarity_matrix</b>: word-word similarity matrix function for gensim's LSI (LSA) model</li>
	</ul>
</ul>