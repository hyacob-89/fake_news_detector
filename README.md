## Final Project Proposal

### Team Members:
* Anastasia Bolboceanu
* Anika Johnson
* Bernt Stenberg
* Henock Yacob


### Project Description:
Using data sets provided by Clément Bisaill via Kaggle, we hope to develop a Fake News Detector designed to identify and analyze news articles, and their level of truthfulness. The list of articles in this dataset were procured by [Politifact](https://www.politifact.com/) and categorized according to their [“Truth-O-Meter Ratings”](https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/#Truth-O-Meter%20ratings) listed below. Our goal is to perform some ETL in Python and PostgreSQL, and leverage machine learning and NLP techniques to analyze/detect fake news.

#### Truth-O-Meter Ratings:
  * TRUE – The statement is accurate and there’s nothing significant missing.
  * MOSTLY TRUE – The statement is accurate but needs clarification or additional information.
  * HALF TRUE – The statement is partially accurate but leaves out important details or takes things out of context.
  * MOSTLY FALSE – The statement contains an element of truth but ignores critical facts that would give a different impression.
  * FALSE – The statement is not accurate.
  * PANTS ON FIRE – The statement is not accurate and makes a ridiculous claim.


### Data Management:
Amazon Relational Database Service (RDS)


### Requirements:
* Scikit-Learn
* Python
* PostgreSQL
* HTML/CSS/Bootstrap
* JavaScript 
* Google Colab


### Steps:
1. Set up database (S3)
2. Add data to database
3. Clean data (Google Colab)
4. Create classification model (NLP)
5. Pass through a different Fake news dataset (novel to the model’s training) and assess model performance


### Dataset:
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
