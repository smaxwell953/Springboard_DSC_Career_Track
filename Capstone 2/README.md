<h1>Data Science Career Track - Predicting Housing Prices</h1>
<h2>1. Introduction</h2>
<p>Homeownership has long had social benefits including greater happiness and prosperity levels, higher civic participation rates, and better education outcomes. When considering whether or not to purchase a house, the prospective buyer looks not just at the price but also factors such as square footage and certain numbers of rooms such as bedrooms and bathrooms.  The goal is to determine what factors have the greatest influence on housing prices.</p>

<h2>2. Data Collection</h2>
<p>The dataset for this project comes from <a href="https://kaggle.com/c/house-prices-advanced-regression-techniques/data">Kaggle</a>.</p>

<h2>3. Data Wrangling</h2>
<p>My first step was to isolate only the values that can be useful in a linear regression analysis. These columns were renamed to make them easier to understand. Ratings of quality and condition were converted to numeric values. Values of 'Nan' were replaced with zeroes.</p>

<h2>4. Feature Engineering and Preprocessing</h2>
<p>Feature engineering steps performed included calculating the age of the house, combining half and full bathrooms into one variable, combining all the porch variables into one variable, and combining the first floor, second floor, and basement areas into an overall square footage variable. Any missing values were imputed with the median.</p>

<h2>5. Exploratory Data Analysis</h2>
<p>I plotted the sales prices against the different variables to see if significant correlations could be found between the variable and the sale prices. After plotting a correlation heat map, I could see a negative correlations in the plots of sale price against the age of the house, and positive correlations in the plots of sale price against the greater living area, overall quality, and basement condition.</p>

<h2>6. Machine Learning</h2>
<p>The random forest regressor model is the best of the models I tried because it showed the highest R2 score, with about 89% of the variance explained by the model, and the lowest mean squared error, of 27,274.</p>

<h2>7. Final Report</h2>
<p>This notebook contains my complete analysis and results.</p>
