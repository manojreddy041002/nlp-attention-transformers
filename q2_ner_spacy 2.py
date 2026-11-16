"""
Part II - Q2: Named Entity Recognition with spaCy
CS5720 Neural Network and Deep Learning - Home Assignment 4

Requires: pip install spacy && python -m spacy download en_core_web_sm
"""
import spacy

# Load the small pretrained English pipeline (includes the NER component)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit("Model missing. Run: python -m spacy download en_core_web_sm")

sentence = (" LeBron James as the NBA player of the United States in Laker "
            "and won the NBA Final MVP in 2020.")

# Run the pipeline; doc.ents holds the detected named entities
doc = nlp(sentence)

print(f"{'Entity':<22}{'Label':<10}{'Start':<7}{'End'}")
for ent in doc.ents:
    # start_char / end_char are character offsets in the original string
    print(f"{ent.text:<22}{ent.label_:<10}{ent.start_char:<7}{ent.end_char}")
