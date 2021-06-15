# Databricks notebook source
pip install nbdev

# COMMAND ----------

dbutils.widgets.dropdown("a dummy widget", "1", [str(x) for x in range(1, 10)])

# COMMAND ----------


