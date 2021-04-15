# Databricks notebook source
# MAGIC %md
# MAGIC <p><img alt="Colaboratory logo" height="45px" src="/img/colab_favicon.ico" align="left" hspace="10px" vspace="0px"></p>
# MAGIC 
# MAGIC <h1>What is Colaboratory?</h1>
# MAGIC 
# MAGIC Colaboratory, or 'Colab' for short, allows you to write and execute Python in your browser, with 
# MAGIC - Zero configuration required
# MAGIC - Free access to GPUs
# MAGIC - Easy sharing
# MAGIC 
# MAGIC Whether you're a <strong>student</strong>, a <strong>data scientist</strong> or an <strong>AI researcher</strong>, Colab can make your work easier. Watch <a href="https://www.youtube.com/watch?v=inN8seMm7UI">Introduction to Colab</a> to find out more, or just get started below!

# COMMAND ----------

# MAGIC %md
# MAGIC ## <strong>Getting started</strong>
# MAGIC 
# MAGIC The document that you are reading is not a static web page, but an interactive environment called a <strong>Colab notebook</strong> that lets you write and execute code.
# MAGIC 
# MAGIC For example, here is a <strong>code cell</strong> with a short Python script that computes a value, stores it in a variable and prints the result:

# COMMAND ----------

seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day

# COMMAND ----------

# MAGIC %md
# MAGIC To execute the code in the above cell, select it with a click and then either press the play button to the left of the code, or use the keyboard shortcut 'Command/Ctrl+Enter'. To edit the code, just click the cell and start editing.
# MAGIC 
# MAGIC Variables that you define in one cell can later be used in other cells:

# COMMAND ----------

seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week

# COMMAND ----------

# MAGIC %md
# MAGIC Colab notebooks allow you to combine <strong>executable code</strong> and <strong>rich text</strong> in a single document, along with <strong>images</strong>, <strong>HTML</strong>, <strong>LaTeX</strong> and more. When you create your own Colab notebooks, they are stored in your Google Drive account. You can easily share your Colab notebooks with co-workers or friends, allowing them to comment on your notebooks or even edit them. To find out more, see <a href="/notebooks/basic_features_overview.ipynb">Overview of Colab</a>. To create a new Colab notebook you can use the File menu above, or use the following link: <a href="http://colab.research.google.com#create=true">Create a new Colab notebook</a>.
# MAGIC 
# MAGIC Colab notebooks are Jupyter notebooks that are hosted by Colab. To find out more about the Jupyter project, see <a href="https://www.jupyter.org">jupyter.org</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data science
# MAGIC 
# MAGIC With Colab you can harness the full power of popular Python libraries to analyse and visualise data. The code cell below uses <strong>numpy</strong> to generate some random data, and uses <strong>matplotlib</strong> to visualise it. To edit the code, just click the cell and start editing.

# COMMAND ----------

import numpy as np
from matplotlib import pyplot as plt

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)

plt.title("Sample Visualization")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC You can import your own data into Colab notebooks from your Google Drive account, including from spreadsheets, as well as from GitHub and many other sources. To find out more about importing data, and how Colab can be used for data science, see the links below under <a href="#working-with-data">Working with data</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Machine learning
# MAGIC 
# MAGIC With Colab you can import an image dataset, train an image classifier on it, and evaluate the model, all in just <a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb">a few lines of code</a>. Colab notebooks execute code on Google's cloud servers, meaning you can leverage the power of Google hardware, including <a href="#using-accelerated-hardware">GPUs and TPUs</a>, regardless of the power of your machine. All you need is a browser.

# COMMAND ----------

# MAGIC %md
# MAGIC Colab is used extensively in the machine learning community with applications including:
# MAGIC - Getting started with TensorFlow
# MAGIC - Developing and training neural networks
# MAGIC - Experimenting with TPUs
# MAGIC - Disseminating AI research
# MAGIC - Creating tutorials
# MAGIC 
# MAGIC To see sample Colab notebooks that demonstrate machine learning applications, see the <a href="#machine-learning-examples">machine learning examples</a> below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## More resources
# MAGIC 
# MAGIC ### Working with notebooks in Colab
# MAGIC - [Overview of Colaboratory](/notebooks/basic_features_overview.ipynb)
# MAGIC - [Guide to markdown](/notebooks/markdown_guide.ipynb)
# MAGIC - [Importing libraries and installing dependencies](/notebooks/snippets/importing_libraries.ipynb)
# MAGIC - [Saving and loading notebooks in GitHub](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
# MAGIC - [Interactive forms](/notebooks/forms.ipynb)
# MAGIC - [Interactive widgets](/notebooks/widgets.ipynb)
# MAGIC - <img src="/img/new.png" height="20px" align="left" hspace="4px" alt="New"></img>
# MAGIC  [TensorFlow 2 in Colab](/notebooks/tensorflow_version.ipynb)
# MAGIC 
# MAGIC <a name="working-with-data"></a>
# MAGIC ### Working with data
# MAGIC - [Loading data: Drive, Sheets and Google Cloud Storage](/notebooks/io.ipynb) 
# MAGIC - [Charts: visualising data](/notebooks/charts.ipynb)
# MAGIC - [Getting started with BigQuery](/notebooks/bigquery.ipynb)
# MAGIC 
# MAGIC ### Machine learning crash course
# MAGIC These are a few of the notebooks from Google's online machine learning course. See the <a href="https://developers.google.com/machine-learning/crash-course/">full course website</a> for more.
# MAGIC - [Intro to Pandas](/notebooks/mlcc/intro_to_pandas.ipynb)
# MAGIC - [TensorFlow concepts](/notebooks/mlcc/tensorflow_programming_concepts.ipynb)
# MAGIC - [First steps with TensorFlow](/notebooks/mlcc/first_steps_with_tensor_flow.ipynb)
# MAGIC - [Intro to neural nets](/notebooks/mlcc/intro_to_neural_nets.ipynb)
# MAGIC - [Intro to sparse data and embeddings](/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb)
# MAGIC 
# MAGIC <a name="using-accelerated-hardware"></a>
# MAGIC ### Using accelerated hardware
# MAGIC - [TensorFlow with GPUs](/notebooks/gpu.ipynb)
# MAGIC - [TensorFlow with TPUs](/notebooks/tpu.ipynb)

# COMMAND ----------

# MAGIC %md
# MAGIC <a name="machine-learning-examples"></a>
# MAGIC 
# MAGIC ## Machine learning examples
# MAGIC 
# MAGIC To see end-to-end examples of the interactive machine-learning analyses that Colaboratory makes possible, take a look at these tutorials using models from <a href="https://tfhub.dev">TensorFlow Hub</a>.
# MAGIC 
# MAGIC A few featured examples:
# MAGIC 
# MAGIC - <a href="https://tensorflow.org/hub/tutorials/tf2_image_retraining">Retraining an Image Classifier</a>: Build a Keras model on top of a pre-trained image classifier to distinguish flowers.
# MAGIC - <a href="https://tensorflow.org/hub/tutorials/tf2_text_classification">Text Classification</a>: Classify IMDB film reviews as either <em>positive</em> or <em>negative</em>.
# MAGIC - <a href="https://tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization">Style Transfer</a>: Use deep learning to transfer style between images.
# MAGIC - <a href="https://tensorflow.org/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa">Multilingual Universal Sentence Encoder Q&amp;A</a>: Use a machine-learning model to answer questions from the SQuAD dataset.
# MAGIC - <a href="https://tensorflow.org/hub/tutorials/tweening_conv3d">Video Interpolation</a>: Predict what happened in a video between the first and the last frame.
