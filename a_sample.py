# Databricks notebook source
test

# COMMAND ----------

dbutils.widgets.dropdown("a dropdown widget", "1", [str(x) for x in range(1, 10)])

# COMMAND ----------

# MAGIC %md
# MAGIC a change which is not there in other notebook

# COMMAND ----------


