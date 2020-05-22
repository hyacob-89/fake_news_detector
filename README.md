#### Final Project - Can NLP Machine Learning Predict Fake News?

### Team Members
* Anastasia Bolboceanu
* Anika Johnson
* Bernt Stenberg
* Henock Yacob


### Project Description
Using data sets provided by Clément Bisaill via Kaggle, we developed a Fake News Detector designed to identify and analyze news articles, and their level of truthfulness. The list of articles in this dataset were procured by [Politifact](https://www.politifact.com/) and categorized according to their [“Truth-O-Meter Ratings”](https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/#Truth-O-Meter%20ratings) listed below. Data was then taken from Politifact Truth-o-Meter for years 2016-2017 categorized as either "True" or "False". Our goal was to perform ETL on the data in Python and PostgreSQL, and leverage machine learning and NLP techniques to analyze/detect fake news.

### Truth-O-Meter Ratings
  * TRUE – The statement is accurate and there’s nothing significant missing.
  * MOSTLY TRUE – The statement is accurate but needs clarification or additional information.
  * HALF TRUE – The statement is partially accurate but leaves out important details or takes things out of context.
  * MOSTLY FALSE – The statement contains an element of truth but ignores critical facts that would give a different impression.
  * FALSE – The statement is not accurate.
  * PANTS ON FIRE – The statement is not accurate and makes a ridiculous claim.

### How does Politifact Fact-check?

"Every fact-check is different, but generally speaking our reporting process includes the following:
a review of what other fact-checkers have found previously;
a thorough Google search;
a search of online databases;
consultation with a variety of experts;
a review of publications
and a final overall review of available evidence." source

### What is NLP?

Natural Language Processing is a machine learning method that breaks down the components of text to understand the grammer and writing style of the text.

### How can We Replicate the Fact-checking Process with Machine Learning?

NLP cannot evaluate the truth of a text. It can however, compare stylistic elements used by reputable news sources like journalistic writing conventions and contrast that with stylistic elements used by unreliable news sources.


### Data Analysis

### NLP Models

* NLTK (Natural Language Toolkit)
* Naive Bayes

### Testing Our Model

To test our model, we gathered additional Politifact articles (25 true and 25 false), starting with the most recent articles posted on the site from in May 2020.  

### Creating our App


### Results/Conclusion

### Limitations of NLP models

1) Our model was developed using data from a specific time period: (2016 to 2017)
* Naives bayes is based on two assumptions: 1) predictors are independent of each other, and 2) past conditions still hold true. When we make predictions from historical data we may get incorrect results if circumstances have changed.
*  Top fake news words may change overtime when connected to real world events that fade from popularity. 

2) Our model identifies text patterns well from humans not machines.
*  [Bayesian Poisoning:](https://en.wikipedia.org/wiki/Bayesian_poisoning) in email spam detection, spammers will try to break machine learning algorithms by attempting to produce a false positive id by introducing positive words that are less likely to indicate spam into their emails. 
* Neural Fake News: this type of fake news uses a Neural Network based model to generate news that replicates the language style used in real news. 

### Practical Application for this type of model:
* "spam filter" which can indicate when a source contains language that indicates it might not be accurate.

### Additional Research Ideas:
*  Create a webscrapping tool that will update our dataset with current news articles from Politfact that will allow us to continue to train our model and keep our model relevant. 
* Develop a new model that can fight Neural Fake News.

-----------------------------
#### Methods

### Data Management:
* Amazon Relational Database Service (RDS)
* Postgres

### Project Components:
* Scikit-Learn
* Python
* PostgreSQL
* HTML/CSS/Bootstrap
* JavaScript 
* Google Colab
* NLTK

------------------------------
#### How to Run this App:

### Prerequisites:
* Install flask
* Install PostgresSQL
* Install Python 3
* Install PySpark
* Install NLTK
* Install Scikit-Learn


### Step-by-Step Process:
1. Clone the repo to your desktop.
2. Open **_PgAdmin_**.
3. Create database called **_fake_news_db_**.
4. Create two new databases called **_fakenews_** and **_truenews_**.
5. Open query tool, open file and run **_fake_news_db.sql_**.
6. Navigate to the repo folder and launch a GitBash or Terminal.
7. type ```source activate PythonData```
8. type ```jupyter notebook```
9. in jupyter notebook, open and run **_ETLnews.ipynb_**.
7. Run **_naive_bayes_model.py_**.
8. Navigate to the **_Web visualization_** folder that contains **_app.py_** and launch a GitBash or Terminal.
9. Type ```source activate PythonData```.
10. Type ```export FLASK_APP=app.py```.
11. Type ```flask run``` and keep that window open.
12. With the Flask server running, load the index page by navigating to http://127.0.0.1:5000/ in your Chrome browser.


### Dataset:
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
