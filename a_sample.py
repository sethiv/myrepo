# Databricks notebook source
test

# COMMAND ----------

dbutils.widgets.dropdown("a dropdown widget", "1", [str(x) for x in range(1, 10)])

# COMMAND ----------


