# Pubmed-Summarizer
This is a web application for summarizing the PubMed articles giving the option to chose between detailed or brief summary.

 INTRODUCTION
PubMed Article Summarizer is a web application that help the user by summarizing the PubMed articles using state-of-art transformer-based model. It takes text file as input and askes the user if he wants detailed or brief summary. This software uses t5 model for summary text generation.  

DATA EXPLORATION
The data for this specific project is taken from Hugging face that is, PubMed Summarization dataset. This dataset consists of two columns article and abstract. I have changed the name of abstract column to summary.   This dataset id primarily designed for model training and evaluation to make the text summarization process autonomous.

PREPROCESSING
In data preprocessing task all the text is lower cased for consistency in the text. All the whitespaces, number and special characters are removed. All the text is tokenized to remove all the stop words, bringing all the words to their base form stemming and lemmatization is done.    

MODEL SELECTION
t5 model is used for text-to-text generation and it would be best suitable for generating summaries of the articles.  I chose to work with t5-small because as the time is short this variant is smaller in size and   it will be processed speedily. 

FINE TUNING
The model is fine-tuned on the preprocessed dataset to generate appropriate summaries of the provide PubMed articles. The fine-tuning process involves tokenizing the data, defining training parameters, and training the model using the Hugging Face Trainer API. 

WEB APPLICATION DEVELOPMENT
The web application was developed using Streamlit, a Python library for building interactive web applications. Users can upload PubMed articles, select the type of summary (detailed or brief), and generate summaries using the fine-tuned T5 model.

RUNNING THE APPLICATION
•	pip install streamlit transformers
•	cd C:\Users\mahnoor\Documents\Python Scripts\1KDD internship assignment
•	streamlit run app.py

