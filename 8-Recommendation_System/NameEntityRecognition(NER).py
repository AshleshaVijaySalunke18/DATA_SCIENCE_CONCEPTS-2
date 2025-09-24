# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 15:30:16 2025

@author: Ashlesha
"""

import spacy
nlp = spacy.load("en_core_web_sm")
nlp.pipe_names

# en_core_web_sm is a small English pipeline model
# that supports tokenization, POS tagging, dependency parsing, NER, etc.
# nlp.pipe_names will show available pipeline components like tok2vec, ner, etc.

###############################

# Process a sample sentence to identify named entities
doc = nlp("Tesla Inc is going to acquire Twitter for $45 billion")

# Iterate over the named entities found in the text
for ent in doc.ents:
    # Print entity text, its label (e.g., ORG, MONEY), and a human-readable explanation
    print(ent.text, " | ", ent.label_, " | ", spacy.explain(ent.label_))

###############################

# doc.ents gives a list of named entities (like Tesla Inc, Twitter, $45 billion)
# ent.label_ returns the entity type (e.g., ORG for organization, MONEY for amount)
# spacy.explain() gives a human-readable explanation.

#############################################
# Now let us apply rendering
from spacy import displacy

displacy.render(doc, style="ent")

...
# This shows a browser-based visualization highlighting
# entities in color.
# Only works in Jupyter or browser environment —
# not in basic Python scripts or terminal.
...
#############################################
# List down all the entities
nlp.pipe_labels['ner']
# Shows all entity labels the NER component 
# can recognize, like PERSON, ORG, MONEY, GPE, etc.

#############################################
doc = nlp("Michael Bloomberg founded Bloomberg in 1982")
for ent in doc.ents:
    print(ent.text, "/", ent.label_, "/", spacy.explain(ent.label_))

...
# doc.ents holds all detected named entities.
# ent.text: the actual entity string (e.g., "Michael Bloomberg")
# ent.label_: the type of entity (e.g., 'PERSON', 'ORG', etc.)
# spacy.explain(ent.label_): gives a description of what the label means

############################

s=doc[2:5]
s
type(s)

"""
This extracts a span from tokens 2 to 4
Slices are useful for extracting phrases
"""
#Manually add Entites
from spacy.tokens import Span

s1 = Span(doc, 0, 1, label="ORG")  # Token 0 to 1 as ORG
s2 = Span(doc, 5, 6, label="ORG")  # Token 5 to 6 as ORG

doc.set_ents([s1, s2], default="unmodified")

'''
you manually define new spans as entities
(span (doc,start,end,label))
set_ents([]) update the doc.em=nts list
default="unmodified " means: keep the  existing ones
as-is,add your custom ones'''
for ent in doc.ents:
    print(ent.text, "|", ent.label_)