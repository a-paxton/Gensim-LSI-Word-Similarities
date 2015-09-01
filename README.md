# Gensim-LSI-Word-Similarities
Two simple little functions to create word-word similarities from Gensim's latent semantic indexing in Python. Both functions produce an inverted cosine similarity score (0 = low, 1 = high) between two words in a Gensim-generated LSA/LSI space across the total number of dimensions specified in the creation of the model (i.e., <i>num_topics</i> from <i>gensim.models.LsiModel</i>).

<p>Both require Gensim, Pandas, and SciPy.

<p>Includes two functions:
<ul>
<li><b>wordsim</b>: Create cosine-derived similarity score (from 0-1) between individual words.</li>
<ul>Input:
<li><i>word1</i> (string or string variable)</li>
<li><i>word2</i> (string or string variable)</li>
<li><i>target_dictionary</i> (Gensim-created LSI dictionary)</li>
<li><i>target_lsi_model</i> (Gensim-created LSI model)</li>
</ul>
<li><b>wordvectsim</b>: Same as <i>wordvect</i> but created to calculate similarity scores (from 0-1) for word pairs in a 2-dimensional word vector (e.g., using <i>numpy.apply_along_axis</i>).</li>
<ul>Input:
<li><i>word_vector2d</i> (2D string vector or 2D string vector variable)</li>
<li><i>target_dictionary</i> (Gensim-created LSI dictionary)</li>
<li><i>target_lsi_model</i> (Gensim-created LSI model)</li>
</ul>
</ul>