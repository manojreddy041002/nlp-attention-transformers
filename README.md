# Mallireddy Manoj Kumar reddy
# 700781871
### Q1: NLP Preprocessing Pipeline (`q1_nlp_preprocessing.py`)
Uses NLTK to process the sentence *"NLP techniques are used in virtual assistants like Alexa and Siri."* in three steps:
1. Tokenize the sentence into words and punctuation.
2. Remove English stopwords and punctuation.
3. Apply Porter stemming to each remaining word.

It prints the original tokens, the tokens without stopwords, and the stemmed words.

### Q2: Named Entity Recognition (`q2_ner_spacy.py`)
Uses spaCy's `en_core_web_sm` model to extract named entities from the input sentence. For each entity it prints the text, the label (e.g. PERSON, ORG, GPE, DATE) and the start/end character positions.

### Q3: Scaled Dot-Product Attention (`q3_scaled_dot_product_attention.py`)
Implements Attention(Q, K, V) = softmax(QKᵀ / √d) V with NumPy and prints the attention weights matrix and the final output matrix for the given test inputs.

Output:
```
Attention weights:      [[0.731 0.269]
                         [0.269 0.731]]
Final output:           [[2.076 3.076 4.076 5.076]
                         [3.924 4.924 5.924 6.924]]
```
