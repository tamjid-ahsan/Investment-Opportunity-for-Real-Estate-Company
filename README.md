# Phase 4 Project

### Problem 1: Time Series Modeling

If you choose the Time Series option, you will be forecasting real estate prices of various zip codes using data from [Zillow Research](https://www.zillow.com/research/data/). For this project, you will be acting as a consultant for a fictional real-estate investment firm. The firm has asked you what seems like a simple question:

> What are the top 5 best zip codes for us to invest in?

This may seem like a simple question at first glance, but there's more than a little ambiguity here that you'll have to think through in order to provide a solid recommendation. Should your recommendation be focused on profit margins only? What about risk? What sort of time horizon are you predicting against?  Your recommendation will need to detail your rationale and answer any sort of lingering questions like these in order to demonstrate how you define "best".

There are many datasets on the [Zillow Research Page](https://www.zillow.com/research/data/), and making sure you have exactly what you need can be a bit confusing. For simplicity's sake, we have already provided the dataset for you in this repo -- you will find it in the file `time-series/zillow_data.csv`.

The goal of this project is to have you complete a very common real-world task in regard to time series modeling. However, real world problems often come with a significant degree of ambiguity, which requires you to use your knowledge of statistics and data science to think critically about and answer. While the main task in this project is time series modeling, that isn't the overall goal -- it is important to understand that time series modeling is a tool in your toolbox, and the forecasts it provides you are what you'll use to answer important questions.

In short, to pass this project, demonstrating the quality and thoughtfulness of your overall recommendation is at least as important as successfully building a time series model!

#### Starter Jupyter Notebook

For this project, you will be provided with a Jupyter notebook, `time-series/starter_notebook.ipynb`, containing some starter code. If you inspect the Zillow dataset file, you'll notice that the datetimes for each sale are the actual column names -- this is a format you probably haven't seen before. To ensure that you're not blocked by preprocessing, we've provided some helper functions to help simplify getting the data into the correct format. You're not required to use this notebook or keep it in its current format, but we strongly recommend you consider making use of the helper functions so you can spend your time working on the parts of the project that matter.

#### Evaluation

In addition to deciding which quantitative metric(s) you want to target (e.g. minimizing mean squared error), you need to start with a definition of "best investment".  Consider additional metrics like risk vs. profitability, or ROI yield.

