# Databricks notebook source
pip install nbdev

# COMMAND ----------

dbutils.widgets.dropdown("X123", "1", [str(x) for x in range(1, 10)])

dbutils.widgets.dropdown("1", "1", [str(x) for x in range(1, 10)], "hello this is a widget")

dbutils.widgets.dropdown("x123123", "1", [str(x) for x in range(1, 10)], "hello this is a widget")

dbutils.widgets.dropdown("x1232133123", "1", [str(x) for x in range(1, 10)], "hello this is a widget 2")


# COMMAND ----------


