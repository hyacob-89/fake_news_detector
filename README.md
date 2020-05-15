## Final Project Proposal

### Team Members:
* Anastasia Bolboceanu
* Anika Johnson
* Bernt Stenberg
* Henock Yacob


### Project Description:
Our goal is to create a Fake News Detector to identify and analyze news articles' truthfulness. The articles were procured from [Politifact](https://www.politifact.com/), who primarily focus on categorizing news articles according to thier [Truth-O-Meter Ratings](https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/#Truth-O-Meter%20ratings) listed below. We'll leverage some machine leaning and NLP techniques to accomplish accomplish this.

#### Truthfullness Definitions
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
