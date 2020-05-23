# # Install Java, Spark, and Findspark
# !apt-get install openjdk-8-jdk-headless -qq > /dev/null
# !wget -q http://www-us.apache.org/dist/spark/spark-2.4.5/spark-2.4.5-bin-hadoop2.7.tgz
# !tar xf spark-2.4.5-bin-hadoop2.7.tgz
# !pip install -q findspark

# # Set Environment Variables
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.5-bin-hadoop2.7"

# Start a SparkSession
import findspark
findspark.init()

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("FakeNewsPoject_Naive_Bayes").getOrCreate()

from pyspark import SparkFiles
# Load in Fake.csv from S3 into a DataFrame
fake_url = "https://bootcamp-proj-3.s3.us-east-2.amazonaws.com/Fake.csv"
spark.sparkContext.addFile(fake_url)

raw_fake_df = spark.read.csv(SparkFiles.get("Fake.csv"), sep=",", header=True)
# raw_fake_df.show(10)

# Load in True.csv from S3 into a DataFrame
true_url = "https://bootcamp-proj-3.s3.us-east-2.amazonaws.com/True.csv"
spark.sparkContext.addFile(true_url)

raw_true_df = spark.read.csv(SparkFiles.get("True.csv"), sep=",", header=True)
# raw_true_df.show(10)



import pyspark.sql.functions as sf

# Add true/fake categories
add_category_fake = raw_fake_df.withColumn('category',sf.lit('Fake'))
add_category_true = raw_true_df.withColumn('category',sf.lit('True'))


#Append and select data

# import pandas as pd
appended_data = add_category_fake.union(add_category_true)\
                                 .select(['category', 'text'])\
                                .dropna(subset=('text'))

# appended_data.show()

from pyspark.sql.functions import length, trim

# Create a length column to be used as a future feature 
review_data = appended_data.withColumn('length', length(appended_data['text']))\
                            .where("length>=100")\
                            .orderBy('length')\
                            .withColumn("text", trim(appended_data.text))
# review_data.show()

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer

# Create all the features to the data set
pos_neg_to_num = StringIndexer(inputCol='category',outputCol='label')
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="stop_tokens", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

# Create feature vectors
clean_up = VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')

# Create a and run a data processing Pipeline
from pyspark.ml import Pipeline
data_prep_pipeline = Pipeline(stages=[pos_neg_to_num, tokenizer, stopremove, hashingTF, idf, clean_up])

# Fit and transform the pipeline
cleaner = data_prep_pipeline.fit(review_data)
cleaned = cleaner.transform(review_data)

# Show label and resulting features
cleaned.select(['label', 'features'])#.show(10)


from pyspark.ml.classification import NaiveBayes
# Break data down into a training set and a testing set
training, testing = cleaned.randomSplit([0.7, 0.3])

# training.show(10)
# testing.show(10)

# Create a Naive Bayes model and fit training data
nb = NaiveBayes()
predictor = nb.fit(training)

# Tranform the model with the testing data
test_results = predictor.transform(testing)
# test_results.show(10, truncate=False)

# Use the Class Evaluator for a cleaner description
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
# print("Accuracy of model at predicting articles' truthfullness was: %f" % acc)