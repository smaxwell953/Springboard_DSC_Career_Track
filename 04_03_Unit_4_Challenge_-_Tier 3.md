
# Springboard Data Science Career Track Unit 4 Challenge - Tier 3 Complete

## Objectives
Hey! Great job getting through those challenging DataCamp courses. You're learning a lot in a short span of time. 

In this notebook, you're going to apply the skills you've been learning, bridging the gap between the controlled environment of DataCamp and the *slightly* messier work that data scientists do with actual datasets!

Here’s the mystery we’re going to solve: ***which boroughs of London have seen the greatest increase in housing prices, on average, over the last two decades?***


A borough is just a fancy word for district. You may be familiar with the five boroughs of New York… well, there are 32 boroughs within Greater London [(here's some info for the curious)](https://en.wikipedia.org/wiki/London_boroughs). Some of them are more desirable areas to live in, and the data will reflect that with a greater rise in housing prices.

***This is the Tier 3 notebook, which means it's not filled in at all: we'll just give you the skeleton of a project, the brief and the data. It's up to you to play around with it and see what you can find out! Good luck! If you struggle, feel free to look at easier tiers for help; but try to dip in and out of them, as the more independent work you do, the better it is for your learning!***

This challenge will make use of only what you learned in the following DataCamp courses: 
- Prework courses (Introduction to Python for Data Science, Intermediate Python for Data Science)
- Data Types for Data Science
- Python Data Science Toolbox (Part One) 
- pandas Foundations
- Manipulating DataFrames with pandas
- Merging DataFrames with pandas

Of the tools, techniques and concepts in the above DataCamp courses, this challenge should require the application of the following: 
- **pandas**
    - **data ingestion and inspection** (pandas Foundations, Module One) 
    - **exploratory data analysis** (pandas Foundations, Module Two)
    - **tidying and cleaning** (Manipulating DataFrames with pandas, Module Three) 
    - **transforming DataFrames** (Manipulating DataFrames with pandas, Module One)
    - **subsetting DataFrames with lists** (Manipulating DataFrames with pandas, Module One) 
    - **filtering DataFrames** (Manipulating DataFrames with pandas, Module One) 
    - **grouping data** (Manipulating DataFrames with pandas, Module Four) 
    - **melting data** (Manipulating DataFrames with pandas, Module Three) 
    - **advanced indexing** (Manipulating DataFrames with pandas, Module Four) 
- **matplotlib** (Intermediate Python for Data Science, Module One)
- **fundamental data types** (Data Types for Data Science, Module One) 
- **dictionaries** (Intermediate Python for Data Science, Module Two)
- **handling dates and times** (Data Types for Data Science, Module Four)
- **function definition** (Python Data Science Toolbox - Part One, Module One)
- **default arguments, variable length, and scope** (Python Data Science Toolbox - Part One, Module Two) 
- **lambda functions and error handling** (Python Data Science Toolbox - Part One, Module Four) 

## The Data Science Pipeline

This is Tier Three, so we'll get you started. But after that, it's all in your hands! When you feel done with your investigations, look back over what you've accomplished, and prepare a quick presentation of your findings for the next mentor meeting. 

Data Science is magical. In this case study, you'll get to apply some complex machine learning algorithms. But as  [David Spiegelhalter](https://www.youtube.com/watch?v=oUs1uvsz0Ok) reminds us, there is no substitute for simply **taking a really, really good look at the data.** Sometimes, this is all we need to answer our question.

Data Science projects generally adhere to the four stages of Data Science Pipeline:
1. Sourcing and loading 
2. Cleaning, transforming, and visualizing 
3. Modeling 
4. Evaluating and concluding 


### 1. Sourcing and Loading 

Any Data Science project kicks off by importing  ***pandas***. The documentation of this wonderful library can be found [here](https://pandas.pydata.org/). As you've seen, pandas is conveniently connected to the [Numpy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/) libraries. 

***Hint:*** This part of the data science pipeline will test those skills you acquired in the pandas Foundations course, Module One. 

#### 1.1. Importing Libraries


```python
# Let's import the pandas, numpy libraries as pd, and np respectively. 
import pandas as pd
import numpy as np

# Load the pyplot collection of functions from matplotlib, as plt 
import matplotlib.pyplot as plt
```

#### 1.2.  Loading the data
Your data comes from the [London Datastore](https://data.london.gov.uk/): a free, open-source data-sharing portal for London-oriented datasets.


```python
# First, make a variable called url_LondonHousePrices, and assign it the following link, enclosed in quotation-marks as a string:
# https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls

url_LondonHousePrices = "https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls"

# The dataset we're interested in contains the Average prices of the houses, and is actually on a particular sheet of the Excel file. 
# As a result, we need to specify the sheet name in the read_excel() method.
# Put this data into a variable called properties.  
properties = pd.read_excel(url_LondonHousePrices, sheet_name='Average price', index_col= None)
```

### 2. Cleaning, transforming, and visualizing
This second stage is arguably the most important part of any Data Science project. The first thing to do is take a proper look at the data. Cleaning forms the majority of this stage, and can be done both before or after Transformation.

The end goal of data cleaning is to have tidy data. When data is tidy: 

1. Each variable has a column.
2. Each observation forms a row.

Keep the end goal in mind as you move through this process, every step will take you closer. 



***Hint:*** This part of the data science pipeline should test those skills you acquired in: 
- Intermediate Python for data science, all modules.
- pandas Foundations, all modules. 
- Manipulating DataFrames with pandas, all modules.
- Data Types for Data Science, Module Four.
- Python Data Science Toolbox - Part One, all modules

**2.1. Exploring your data** 

Think about your pandas functions for checking out a dataframe. 


```python
#Inspect the dataframe
properties
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>Enfield</th>
      <th>...</th>
      <th>NORTH WEST</th>
      <th>YORKS &amp; THE HUMBER</th>
      <th>EAST MIDLANDS</th>
      <th>WEST MIDLANDS</th>
      <th>EAST OF ENGLAND</th>
      <th>LONDON</th>
      <th>SOUTH EAST</th>
      <th>SOUTH WEST</th>
      <th>Unnamed: 46</th>
      <th>England</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NaT</th>
      <td>E09000001</td>
      <td>E09000002</td>
      <td>E09000003</td>
      <td>E09000004</td>
      <td>E09000005</td>
      <td>E09000006</td>
      <td>E09000007</td>
      <td>E09000008</td>
      <td>E09000009</td>
      <td>E09000010</td>
      <td>...</td>
      <td>E12000002</td>
      <td>E12000003</td>
      <td>E12000004</td>
      <td>E12000005</td>
      <td>E12000006</td>
      <td>E12000007</td>
      <td>E12000008</td>
      <td>E12000009</td>
      <td>nan</td>
      <td>E92000001</td>
    </tr>
    <tr>
      <th>1995-01-01</th>
      <td>91448.98</td>
      <td>50460.23</td>
      <td>93284.52</td>
      <td>64958.09</td>
      <td>71306.57</td>
      <td>81671.48</td>
      <td>120932.89</td>
      <td>69158.16</td>
      <td>79885.89</td>
      <td>72514.69</td>
      <td>...</td>
      <td>43958.48</td>
      <td>44803.43</td>
      <td>45544.52</td>
      <td>48527.52</td>
      <td>56701.60</td>
      <td>74435.76</td>
      <td>64018.88</td>
      <td>54705.16</td>
      <td>nan</td>
      <td>53202.77</td>
    </tr>
    <tr>
      <th>1995-02-01</th>
      <td>82202.77</td>
      <td>51085.78</td>
      <td>93190.17</td>
      <td>64787.92</td>
      <td>72022.26</td>
      <td>81657.56</td>
      <td>119508.86</td>
      <td>68951.10</td>
      <td>80897.07</td>
      <td>73155.20</td>
      <td>...</td>
      <td>43925.42</td>
      <td>44528.81</td>
      <td>46051.57</td>
      <td>49341.29</td>
      <td>56593.59</td>
      <td>72777.94</td>
      <td>63715.02</td>
      <td>54356.15</td>
      <td>nan</td>
      <td>53096.15</td>
    </tr>
    <tr>
      <th>1995-03-01</th>
      <td>79120.70</td>
      <td>51268.97</td>
      <td>92247.52</td>
      <td>64367.49</td>
      <td>72015.76</td>
      <td>81449.31</td>
      <td>120282.21</td>
      <td>68712.44</td>
      <td>81379.86</td>
      <td>72190.44</td>
      <td>...</td>
      <td>44434.87</td>
      <td>45200.47</td>
      <td>45383.82</td>
      <td>49442.18</td>
      <td>56171.18</td>
      <td>73896.84</td>
      <td>64113.61</td>
      <td>53583.08</td>
      <td>nan</td>
      <td>53201.28</td>
    </tr>
    <tr>
      <th>1995-04-01</th>
      <td>77101.21</td>
      <td>53133.51</td>
      <td>90762.87</td>
      <td>64277.67</td>
      <td>72965.63</td>
      <td>81124.41</td>
      <td>120097.90</td>
      <td>68610.05</td>
      <td>82188.90</td>
      <td>71442.92</td>
      <td>...</td>
      <td>44267.78</td>
      <td>45614.34</td>
      <td>46124.23</td>
      <td>49455.93</td>
      <td>56567.90</td>
      <td>74455.29</td>
      <td>64623.22</td>
      <td>54786.02</td>
      <td>nan</td>
      <td>53590.85</td>
    </tr>
    <tr>
      <th>1995-05-01</th>
      <td>84409.15</td>
      <td>53042.25</td>
      <td>90258.00</td>
      <td>63997.14</td>
      <td>73704.05</td>
      <td>81542.62</td>
      <td>119929.28</td>
      <td>68844.92</td>
      <td>82077.06</td>
      <td>70630.78</td>
      <td>...</td>
      <td>44223.62</td>
      <td>44830.99</td>
      <td>45878.00</td>
      <td>50369.66</td>
      <td>56479.80</td>
      <td>75432.03</td>
      <td>64530.36</td>
      <td>54698.84</td>
      <td>nan</td>
      <td>53678.24</td>
    </tr>
    <tr>
      <th>1995-06-01</th>
      <td>94900.51</td>
      <td>53700.35</td>
      <td>90107.23</td>
      <td>64252.32</td>
      <td>74310.48</td>
      <td>82382.83</td>
      <td>121887.46</td>
      <td>69052.51</td>
      <td>81630.66</td>
      <td>71348.31</td>
      <td>...</td>
      <td>44112.96</td>
      <td>45392.64</td>
      <td>45680.00</td>
      <td>50100.43</td>
      <td>56288.95</td>
      <td>75606.25</td>
      <td>65511.01</td>
      <td>54420.16</td>
      <td>nan</td>
      <td>53735.15</td>
    </tr>
    <tr>
      <th>1995-07-01</th>
      <td>110128.04</td>
      <td>52113.12</td>
      <td>91441.25</td>
      <td>63722.70</td>
      <td>74127.04</td>
      <td>82898.52</td>
      <td>124027.58</td>
      <td>69142.48</td>
      <td>82352.22</td>
      <td>71837.54</td>
      <td>...</td>
      <td>44109.59</td>
      <td>45535.00</td>
      <td>46037.67</td>
      <td>49860.01</td>
      <td>57242.30</td>
      <td>75984.24</td>
      <td>65224.88</td>
      <td>54265.86</td>
      <td>nan</td>
      <td>53900.61</td>
    </tr>
    <tr>
      <th>1995-08-01</th>
      <td>112329.44</td>
      <td>52232.20</td>
      <td>92361.32</td>
      <td>64432.60</td>
      <td>73547.04</td>
      <td>82054.37</td>
      <td>125529.80</td>
      <td>68993.43</td>
      <td>82706.66</td>
      <td>72237.95</td>
      <td>...</td>
      <td>44193.67</td>
      <td>45111.46</td>
      <td>45922.54</td>
      <td>49598.46</td>
      <td>56732.41</td>
      <td>75529.34</td>
      <td>64851.60</td>
      <td>54365.71</td>
      <td>nan</td>
      <td>53600.32</td>
    </tr>
    <tr>
      <th>1995-09-01</th>
      <td>104473.11</td>
      <td>51471.61</td>
      <td>93273.12</td>
      <td>64509.55</td>
      <td>73789.54</td>
      <td>81440.43</td>
      <td>120596.85</td>
      <td>69393.50</td>
      <td>82011.08</td>
      <td>71725.22</td>
      <td>...</td>
      <td>44088.08</td>
      <td>44837.86</td>
      <td>45771.66</td>
      <td>49319.70</td>
      <td>56259.29</td>
      <td>74940.81</td>
      <td>64352.47</td>
      <td>54243.99</td>
      <td>nan</td>
      <td>53309.23</td>
    </tr>
    <tr>
      <th>1995-10-01</th>
      <td>108038.12</td>
      <td>51513.76</td>
      <td>92567.38</td>
      <td>64529.94</td>
      <td>73264.05</td>
      <td>81862.16</td>
      <td>117458.49</td>
      <td>68934.61</td>
      <td>80522.66</td>
      <td>72349.14</td>
      <td>...</td>
      <td>43109.99</td>
      <td>44310.00</td>
      <td>45154.72</td>
      <td>49597.94</td>
      <td>55240.29</td>
      <td>74302.08</td>
      <td>64125.81</td>
      <td>54264.61</td>
      <td>nan</td>
      <td>52844.62</td>
    </tr>
    <tr>
      <th>1995-11-01</th>
      <td>117635.61</td>
      <td>50848.68</td>
      <td>90883.16</td>
      <td>63846.03</td>
      <td>72782.01</td>
      <td>82557.78</td>
      <td>115418.84</td>
      <td>68879.75</td>
      <td>79272.35</td>
      <td>71987.87</td>
      <td>...</td>
      <td>43335.25</td>
      <td>44166.43</td>
      <td>44967.22</td>
      <td>48499.96</td>
      <td>55877.85</td>
      <td>74117.79</td>
      <td>64416.21</td>
      <td>53799.22</td>
      <td>nan</td>
      <td>52787.68</td>
    </tr>
    <tr>
      <th>1995-12-01</th>
      <td>127232.45</td>
      <td>50945.19</td>
      <td>91133.90</td>
      <td>63816.94</td>
      <td>72523.67</td>
      <td>82966.33</td>
      <td>118739.01</td>
      <td>68407.15</td>
      <td>79699.69</td>
      <td>72390.76</td>
      <td>...</td>
      <td>43373.93</td>
      <td>44457.99</td>
      <td>45648.71</td>
      <td>49029.69</td>
      <td>55523.84</td>
      <td>75177.56</td>
      <td>64260.60</td>
      <td>53081.17</td>
      <td>nan</td>
      <td>52921.62</td>
    </tr>
    <tr>
      <th>1996-01-01</th>
      <td>108998.64</td>
      <td>50828.11</td>
      <td>91111.01</td>
      <td>63995.99</td>
      <td>72806.35</td>
      <td>82210.29</td>
      <td>119560.48</td>
      <td>68006.01</td>
      <td>80214.27</td>
      <td>71544.18</td>
      <td>...</td>
      <td>42599.17</td>
      <td>43460.63</td>
      <td>44618.55</td>
      <td>47951.87</td>
      <td>55033.24</td>
      <td>75341.92</td>
      <td>64057.18</td>
      <td>53373.22</td>
      <td>nan</td>
      <td>52333.23</td>
    </tr>
    <tr>
      <th>1996-02-01</th>
      <td>93356.70</td>
      <td>51440.75</td>
      <td>92429.53</td>
      <td>64503.66</td>
      <td>73084.06</td>
      <td>81418.51</td>
      <td>123370.79</td>
      <td>68031.99</td>
      <td>80836.42</td>
      <td>71070.19</td>
      <td>...</td>
      <td>43080.53</td>
      <td>43814.78</td>
      <td>44506.13</td>
      <td>48646.10</td>
      <td>55315.55</td>
      <td>76086.64</td>
      <td>63770.10</td>
      <td>53255.27</td>
      <td>nan</td>
      <td>52535.58</td>
    </tr>
    <tr>
      <th>1996-03-01</th>
      <td>93706.70</td>
      <td>51907.07</td>
      <td>91409.82</td>
      <td>64787.98</td>
      <td>72779.77</td>
      <td>81633.34</td>
      <td>124723.00</td>
      <td>68269.87</td>
      <td>81011.38</td>
      <td>71544.14</td>
      <td>...</td>
      <td>43162.74</td>
      <td>43881.26</td>
      <td>45189.85</td>
      <td>48811.37</td>
      <td>55063.21</td>
      <td>75690.65</td>
      <td>63808.53</td>
      <td>53608.10</td>
      <td>nan</td>
      <td>52682.61</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>120542.79</td>
      <td>51724.03</td>
      <td>92394.19</td>
      <td>65285.92</td>
      <td>72369.72</td>
      <td>81480.64</td>
      <td>129158.52</td>
      <td>68897.63</td>
      <td>81041.06</td>
      <td>72517.33</td>
      <td>...</td>
      <td>43390.13</td>
      <td>44539.20</td>
      <td>45704.09</td>
      <td>48753.48</td>
      <td>55822.46</td>
      <td>76767.25</td>
      <td>64984.69</td>
      <td>54761.74</td>
      <td>nan</td>
      <td>53331.16</td>
    </tr>
    <tr>
      <th>1996-05-01</th>
      <td>112050.08</td>
      <td>51735.73</td>
      <td>91058.91</td>
      <td>65080.67</td>
      <td>72994.83</td>
      <td>82372.46</td>
      <td>135335.59</td>
      <td>68831.80</td>
      <td>81821.02</td>
      <td>73630.50</td>
      <td>...</td>
      <td>43924.20</td>
      <td>44948.42</td>
      <td>45728.21</td>
      <td>50097.95</td>
      <td>56139.60</td>
      <td>77214.88</td>
      <td>65343.89</td>
      <td>54871.01</td>
      <td>nan</td>
      <td>53821.83</td>
    </tr>
    <tr>
      <th>1996-06-01</th>
      <td>114226.19</td>
      <td>50761.40</td>
      <td>91978.19</td>
      <td>65119.76</td>
      <td>73497.60</td>
      <td>82946.43</td>
      <td>143861.82</td>
      <td>69051.78</td>
      <td>82562.44</td>
      <td>74158.32</td>
      <td>...</td>
      <td>43871.93</td>
      <td>45127.17</td>
      <td>46148.83</td>
      <td>49845.52</td>
      <td>56600.92</td>
      <td>78138.84</td>
      <td>65927.29</td>
      <td>55075.51</td>
      <td>nan</td>
      <td>54092.71</td>
    </tr>
    <tr>
      <th>1996-07-01</th>
      <td>97546.66</td>
      <td>50621.11</td>
      <td>92444.09</td>
      <td>65150.64</td>
      <td>75551.18</td>
      <td>84088.81</td>
      <td>147974.35</td>
      <td>69400.43</td>
      <td>84005.38</td>
      <td>74456.07</td>
      <td>...</td>
      <td>44306.93</td>
      <td>45650.22</td>
      <td>46577.12</td>
      <td>50429.82</td>
      <td>57072.97</td>
      <td>78798.97</td>
      <td>67000.66</td>
      <td>55958.45</td>
      <td>nan</td>
      <td>54703.69</td>
    </tr>
    <tr>
      <th>1996-08-01</th>
      <td>114178.96</td>
      <td>51104.69</td>
      <td>95515.96</td>
      <td>65960.20</td>
      <td>76913.74</td>
      <td>84400.03</td>
      <td>142462.07</td>
      <td>70479.67</td>
      <td>84824.41</td>
      <td>75285.78</td>
      <td>...</td>
      <td>44595.28</td>
      <td>45530.85</td>
      <td>47066.52</td>
      <td>51188.98</td>
      <td>57951.02</td>
      <td>79058.30</td>
      <td>67332.88</td>
      <td>56223.56</td>
      <td>nan</td>
      <td>55106.64</td>
    </tr>
    <tr>
      <th>1996-09-01</th>
      <td>108138.34</td>
      <td>51892.72</td>
      <td>97489.86</td>
      <td>66500.65</td>
      <td>78667.11</td>
      <td>84675.93</td>
      <td>136551.96</td>
      <td>70696.54</td>
      <td>85674.21</td>
      <td>75748.78</td>
      <td>...</td>
      <td>44638.56</td>
      <td>45131.27</td>
      <td>46938.27</td>
      <td>50847.10</td>
      <td>57609.04</td>
      <td>79752.07</td>
      <td>67196.35</td>
      <td>56443.75</td>
      <td>nan</td>
      <td>55025.87</td>
    </tr>
    <tr>
      <th>1996-10-01</th>
      <td>98934.21</td>
      <td>52533.15</td>
      <td>97844.46</td>
      <td>66659.34</td>
      <td>77990.87</td>
      <td>85513.19</td>
      <td>133849.81</td>
      <td>71406.06</td>
      <td>85595.17</td>
      <td>75933.60</td>
      <td>...</td>
      <td>44473.38</td>
      <td>44874.12</td>
      <td>46520.35</td>
      <td>50979.37</td>
      <td>57668.70</td>
      <td>79504.43</td>
      <td>67132.87</td>
      <td>56785.88</td>
      <td>nan</td>
      <td>54930.78</td>
    </tr>
    <tr>
      <th>1996-11-01</th>
      <td>99706.78</td>
      <td>52216.04</td>
      <td>97155.03</td>
      <td>66353.76</td>
      <td>77914.21</td>
      <td>85780.90</td>
      <td>134128.29</td>
      <td>71397.86</td>
      <td>86230.21</td>
      <td>75976.55</td>
      <td>...</td>
      <td>44454.70</td>
      <td>45639.16</td>
      <td>47287.40</td>
      <td>51597.83</td>
      <td>58420.92</td>
      <td>80223.97</td>
      <td>67966.66</td>
      <td>57181.83</td>
      <td>nan</td>
      <td>55496.60</td>
    </tr>
    <tr>
      <th>1996-12-01</th>
      <td>106426.65</td>
      <td>53853.51</td>
      <td>97174.30</td>
      <td>66486.44</td>
      <td>78261.58</td>
      <td>86049.26</td>
      <td>134749.18</td>
      <td>72097.39</td>
      <td>86863.02</td>
      <td>76878.71</td>
      <td>...</td>
      <td>44640.78</td>
      <td>45251.17</td>
      <td>47641.71</td>
      <td>51418.03</td>
      <td>58964.05</td>
      <td>81416.76</td>
      <td>68771.60</td>
      <td>56951.17</td>
      <td>nan</td>
      <td>55755.28</td>
    </tr>
    <tr>
      <th>1997-01-01</th>
      <td>116343.26</td>
      <td>54459.03</td>
      <td>98558.78</td>
      <td>67350.56</td>
      <td>79663.96</td>
      <td>86846.01</td>
      <td>135956.30</td>
      <td>72165.22</td>
      <td>88091.73</td>
      <td>77692.81</td>
      <td>...</td>
      <td>44094.93</td>
      <td>44984.37</td>
      <td>46775.09</td>
      <td>51119.62</td>
      <td>59081.20</td>
      <td>83065.74</td>
      <td>69007.75</td>
      <td>57751.25</td>
      <td>nan</td>
      <td>55788.50</td>
    </tr>
    <tr>
      <th>1997-02-01</th>
      <td>111142.24</td>
      <td>55451.49</td>
      <td>100075.90</td>
      <td>67981.18</td>
      <td>81213.22</td>
      <td>88200.87</td>
      <td>141531.33</td>
      <td>72207.30</td>
      <td>89942.46</td>
      <td>78080.72</td>
      <td>...</td>
      <td>44691.09</td>
      <td>44759.02</td>
      <td>47345.83</td>
      <td>51443.02</td>
      <td>59683.67</td>
      <td>83579.43</td>
      <td>69675.04</td>
      <td>57920.55</td>
      <td>nan</td>
      <td>56199.25</td>
    </tr>
    <tr>
      <th>1997-03-01</th>
      <td>107487.77</td>
      <td>55004.77</td>
      <td>101129.88</td>
      <td>67823.81</td>
      <td>82321.89</td>
      <td>89764.82</td>
      <td>143444.34</td>
      <td>72229.56</td>
      <td>91027.38</td>
      <td>78402.24</td>
      <td>...</td>
      <td>45054.48</td>
      <td>45314.75</td>
      <td>47721.23</td>
      <td>52104.01</td>
      <td>59868.75</td>
      <td>84575.77</td>
      <td>70721.15</td>
      <td>58622.91</td>
      <td>nan</td>
      <td>56784.27</td>
    </tr>
    <tr>
      <th>1997-04-01</th>
      <td>108480.13</td>
      <td>55279.86</td>
      <td>101200.79</td>
      <td>68470.51</td>
      <td>82771.00</td>
      <td>90000.43</td>
      <td>143591.55</td>
      <td>73527.90</td>
      <td>91990.81</td>
      <td>79603.50</td>
      <td>...</td>
      <td>44945.85</td>
      <td>46306.10</td>
      <td>48694.04</td>
      <td>52548.19</td>
      <td>60357.72</td>
      <td>86087.56</td>
      <td>71954.71</td>
      <td>59538.07</td>
      <td>nan</td>
      <td>57577.27</td>
    </tr>
    <tr>
      <th>1997-05-01</th>
      <td>115453.11</td>
      <td>54936.94</td>
      <td>103286.56</td>
      <td>69353.48</td>
      <td>82729.65</td>
      <td>91229.57</td>
      <td>148617.04</td>
      <td>74829.55</td>
      <td>93697.20</td>
      <td>81700.69</td>
      <td>...</td>
      <td>45703.83</td>
      <td>46634.12</td>
      <td>49018.60</td>
      <td>53616.47</td>
      <td>61736.88</td>
      <td>88543.68</td>
      <td>73359.13</td>
      <td>59732.32</td>
      <td>nan</td>
      <td>58497.92</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2017-11-01</th>
      <td>790158.90</td>
      <td>293796.31</td>
      <td>541562.51</td>
      <td>336238.75</td>
      <td>485284.86</td>
      <td>445251.05</td>
      <td>846353.79</td>
      <td>375639.90</td>
      <td>479332.23</td>
      <td>397553.39</td>
      <td>...</td>
      <td>157803.69</td>
      <td>157087.89</td>
      <td>183974.06</td>
      <td>188640.08</td>
      <td>287142.75</td>
      <td>476290.47</td>
      <td>320212.15</td>
      <td>249345.03</td>
      <td>nan</td>
      <td>241086.06</td>
    </tr>
    <tr>
      <th>2017-12-01</th>
      <td>778001.53</td>
      <td>292914.71</td>
      <td>538717.71</td>
      <td>340598.38</td>
      <td>494515.16</td>
      <td>445503.79</td>
      <td>844415.69</td>
      <td>374440.85</td>
      <td>469145.69</td>
      <td>395284.09</td>
      <td>...</td>
      <td>158863.20</td>
      <td>158435.42</td>
      <td>184941.68</td>
      <td>190466.25</td>
      <td>289022.32</td>
      <td>476848.02</td>
      <td>320732.72</td>
      <td>249308.76</td>
      <td>nan</td>
      <td>242377.77</td>
    </tr>
    <tr>
      <th>2018-01-01</th>
      <td>802128.85</td>
      <td>291547.86</td>
      <td>531832.20</td>
      <td>342095.71</td>
      <td>493882.35</td>
      <td>446736.58</td>
      <td>859592.51</td>
      <td>372312.13</td>
      <td>473630.80</td>
      <td>395037.61</td>
      <td>...</td>
      <td>155595.52</td>
      <td>155887.94</td>
      <td>184573.70</td>
      <td>187845.36</td>
      <td>288946.92</td>
      <td>479772.29</td>
      <td>320977.64</td>
      <td>252532.67</td>
      <td>nan</td>
      <td>241061.24</td>
    </tr>
    <tr>
      <th>2018-02-01</th>
      <td>783266.49</td>
      <td>292777.26</td>
      <td>531736.20</td>
      <td>341254.38</td>
      <td>493621.09</td>
      <td>445944.32</td>
      <td>875078.39</td>
      <td>371040.29</td>
      <td>476730.57</td>
      <td>397704.16</td>
      <td>...</td>
      <td>157518.68</td>
      <td>155693.57</td>
      <td>187234.83</td>
      <td>190682.02</td>
      <td>288285.96</td>
      <td>477859.67</td>
      <td>321374.77</td>
      <td>250793.15</td>
      <td>nan</td>
      <td>241989.47</td>
    </tr>
    <tr>
      <th>2018-03-01</th>
      <td>740799.12</td>
      <td>291723.50</td>
      <td>538120.12</td>
      <td>339631.62</td>
      <td>482967.37</td>
      <td>441430.18</td>
      <td>853395.73</td>
      <td>366045.16</td>
      <td>483878.09</td>
      <td>397593.47</td>
      <td>...</td>
      <td>155624.71</td>
      <td>156465.65</td>
      <td>184845.35</td>
      <td>190077.32</td>
      <td>286498.68</td>
      <td>472357.07</td>
      <td>319875.87</td>
      <td>250615.04</td>
      <td>nan</td>
      <td>240427.64</td>
    </tr>
    <tr>
      <th>2018-04-01</th>
      <td>732350.84</td>
      <td>291183.84</td>
      <td>541502.22</td>
      <td>338476.39</td>
      <td>485433.90</td>
      <td>440858.62</td>
      <td>866438.05</td>
      <td>368281.43</td>
      <td>481101.19</td>
      <td>396942.47</td>
      <td>...</td>
      <td>157768.19</td>
      <td>157430.50</td>
      <td>187276.27</td>
      <td>191375.43</td>
      <td>287655.92</td>
      <td>477253.40</td>
      <td>321352.61</td>
      <td>250253.71</td>
      <td>nan</td>
      <td>242396.10</td>
    </tr>
    <tr>
      <th>2018-05-01</th>
      <td>796398.61</td>
      <td>290239.33</td>
      <td>533088.42</td>
      <td>336410.03</td>
      <td>480836.48</td>
      <td>437880.01</td>
      <td>837927.51</td>
      <td>370540.74</td>
      <td>473140.41</td>
      <td>398208.02</td>
      <td>...</td>
      <td>158822.80</td>
      <td>159541.50</td>
      <td>188826.00</td>
      <td>191299.06</td>
      <td>288222.39</td>
      <td>478485.47</td>
      <td>321824.57</td>
      <td>251132.04</td>
      <td>nan</td>
      <td>243445.48</td>
    </tr>
    <tr>
      <th>2018-06-01</th>
      <td>789277.78</td>
      <td>294235.79</td>
      <td>524701.99</td>
      <td>341480.06</td>
      <td>492743.21</td>
      <td>436971.00</td>
      <td>850612.56</td>
      <td>373885.72</td>
      <td>476827.69</td>
      <td>396699.80</td>
      <td>...</td>
      <td>160313.53</td>
      <td>160484.45</td>
      <td>189258.61</td>
      <td>193886.04</td>
      <td>289468.62</td>
      <td>479931.38</td>
      <td>324665.15</td>
      <td>252603.32</td>
      <td>nan</td>
      <td>244962.42</td>
    </tr>
    <tr>
      <th>2018-07-01</th>
      <td>809696.97</td>
      <td>294952.74</td>
      <td>523397.05</td>
      <td>345692.83</td>
      <td>505027.72</td>
      <td>440996.44</td>
      <td>837711.75</td>
      <td>374361.16</td>
      <td>478375.67</td>
      <td>397849.41</td>
      <td>...</td>
      <td>162345.25</td>
      <td>162019.52</td>
      <td>191230.53</td>
      <td>196297.20</td>
      <td>293936.26</td>
      <td>484724.11</td>
      <td>327090.55</td>
      <td>256892.52</td>
      <td>nan</td>
      <td>247980.90</td>
    </tr>
    <tr>
      <th>2018-08-01</th>
      <td>784143.72</td>
      <td>295907.84</td>
      <td>533714.99</td>
      <td>347531.06</td>
      <td>519753.71</td>
      <td>444572.79</td>
      <td>827363.20</td>
      <td>372035.48</td>
      <td>485596.17</td>
      <td>395213.32</td>
      <td>...</td>
      <td>163091.66</td>
      <td>164087.51</td>
      <td>192609.86</td>
      <td>182447.39</td>
      <td>294035.04</td>
      <td>479549.86</td>
      <td>328060.73</td>
      <td>258465.43</td>
      <td>nan</td>
      <td>248619.95</td>
    </tr>
    <tr>
      <th>2018-09-01</th>
      <td>800874.53</td>
      <td>296423.97</td>
      <td>534951.41</td>
      <td>346150.77</td>
      <td>510929.80</td>
      <td>449292.94</td>
      <td>817464.42</td>
      <td>371792.94</td>
      <td>485266.18</td>
      <td>394729.20</td>
      <td>...</td>
      <td>163068.07</td>
      <td>162179.66</td>
      <td>194049.16</td>
      <td>183861.05</td>
      <td>292573.89</td>
      <td>476545.00</td>
      <td>327053.82</td>
      <td>258838.53</td>
      <td>nan</td>
      <td>248248.48</td>
    </tr>
    <tr>
      <th>2018-10-01</th>
      <td>802869.07</td>
      <td>299648.49</td>
      <td>537283.90</td>
      <td>344680.66</td>
      <td>493489.48</td>
      <td>447534.11</td>
      <td>808320.99</td>
      <td>370363.35</td>
      <td>481233.32</td>
      <td>397497.69</td>
      <td>...</td>
      <td>163846.15</td>
      <td>161759.44</td>
      <td>191624.54</td>
      <td>183987.33</td>
      <td>292886.46</td>
      <td>479774.52</td>
      <td>324284.25</td>
      <td>256883.01</td>
      <td>nan</td>
      <td>247675.72</td>
    </tr>
    <tr>
      <th>2018-11-01</th>
      <td>764206.74</td>
      <td>302605.92</td>
      <td>535059.84</td>
      <td>344172.30</td>
      <td>479103.68</td>
      <td>445563.04</td>
      <td>807224.55</td>
      <td>368735.29</td>
      <td>482904.57</td>
      <td>394431.87</td>
      <td>...</td>
      <td>163172.77</td>
      <td>161418.65</td>
      <td>192969.11</td>
      <td>184621.30</td>
      <td>293281.80</td>
      <td>474550.69</td>
      <td>323125.23</td>
      <td>256804.10</td>
      <td>nan</td>
      <td>246896.48</td>
    </tr>
    <tr>
      <th>2018-12-01</th>
      <td>811694.19</td>
      <td>301113.55</td>
      <td>533810.05</td>
      <td>343667.47</td>
      <td>476355.65</td>
      <td>443139.80</td>
      <td>867795.25</td>
      <td>364188.83</td>
      <td>481921.88</td>
      <td>397673.59</td>
      <td>...</td>
      <td>162823.21</td>
      <td>162266.59</td>
      <td>191780.90</td>
      <td>185061.02</td>
      <td>290885.66</td>
      <td>473454.32</td>
      <td>321264.83</td>
      <td>256811.90</td>
      <td>nan</td>
      <td>246518.47</td>
    </tr>
    <tr>
      <th>2019-01-01</th>
      <td>865636.10</td>
      <td>297180.89</td>
      <td>528638.96</td>
      <td>338707.82</td>
      <td>479105.71</td>
      <td>440623.57</td>
      <td>867808.29</td>
      <td>365096.06</td>
      <td>483335.78</td>
      <td>395616.18</td>
      <td>...</td>
      <td>161408.62</td>
      <td>160341.43</td>
      <td>191237.46</td>
      <td>195497.84</td>
      <td>288658.86</td>
      <td>470067.49</td>
      <td>322226.29</td>
      <td>254632.08</td>
      <td>nan</td>
      <td>244640.83</td>
    </tr>
    <tr>
      <th>2019-02-01</th>
      <td>894519.91</td>
      <td>293838.65</td>
      <td>523678.59</td>
      <td>339186.15</td>
      <td>478494.21</td>
      <td>439716.32</td>
      <td>864013.62</td>
      <td>364405.54</td>
      <td>476647.60</td>
      <td>397254.17</td>
      <td>...</td>
      <td>162066.31</td>
      <td>159283.67</td>
      <td>190989.73</td>
      <td>197367.77</td>
      <td>289831.80</td>
      <td>466068.22</td>
      <td>320094.71</td>
      <td>255243.43</td>
      <td>nan</td>
      <td>244427.01</td>
    </tr>
    <tr>
      <th>2019-03-01</th>
      <td>853451.08</td>
      <td>294064.76</td>
      <td>516531.22</td>
      <td>336679.64</td>
      <td>475429.65</td>
      <td>435996.30</td>
      <td>808859.99</td>
      <td>367440.28</td>
      <td>467017.21</td>
      <td>393925.71</td>
      <td>...</td>
      <td>160669.84</td>
      <td>160771.42</td>
      <td>190586.19</td>
      <td>195057.96</td>
      <td>286802.29</td>
      <td>464162.21</td>
      <td>317129.13</td>
      <td>253050.94</td>
      <td>nan</td>
      <td>242982.46</td>
    </tr>
    <tr>
      <th>2019-04-01</th>
      <td>738796.72</td>
      <td>295498.22</td>
      <td>512343.34</td>
      <td>339330.12</td>
      <td>486252.78</td>
      <td>431643.36</td>
      <td>820812.02</td>
      <td>363627.44</td>
      <td>461457.96</td>
      <td>393880.47</td>
      <td>...</td>
      <td>162263.51</td>
      <td>161862.40</td>
      <td>191550.34</td>
      <td>195523.09</td>
      <td>288011.41</td>
      <td>469494.40</td>
      <td>319063.02</td>
      <td>253155.27</td>
      <td>nan</td>
      <td>244728.16</td>
    </tr>
    <tr>
      <th>2019-05-01</th>
      <td>719217.56</td>
      <td>295092.09</td>
      <td>503911.19</td>
      <td>336980.59</td>
      <td>481803.42</td>
      <td>428092.90</td>
      <td>838678.63</td>
      <td>364097.31</td>
      <td>465752.39</td>
      <td>388590.88</td>
      <td>...</td>
      <td>164139.39</td>
      <td>162271.88</td>
      <td>192072.57</td>
      <td>197211.40</td>
      <td>288859.07</td>
      <td>463834.09</td>
      <td>318757.47</td>
      <td>252789.05</td>
      <td>nan</td>
      <td>245141.56</td>
    </tr>
    <tr>
      <th>2019-06-01</th>
      <td>761525.87</td>
      <td>293889.31</td>
      <td>512694.06</td>
      <td>339323.70</td>
      <td>474820.91</td>
      <td>430001.89</td>
      <td>871957.45</td>
      <td>363420.09</td>
      <td>474844.30</td>
      <td>390707.68</td>
      <td>...</td>
      <td>163927.46</td>
      <td>163780.58</td>
      <td>192143.90</td>
      <td>196666.42</td>
      <td>288717.39</td>
      <td>470878.17</td>
      <td>321041.37</td>
      <td>252945.78</td>
      <td>nan</td>
      <td>246014.28</td>
    </tr>
    <tr>
      <th>2019-07-01</th>
      <td>756407.02</td>
      <td>297426.44</td>
      <td>514667.76</td>
      <td>338346.40</td>
      <td>473849.04</td>
      <td>434256.63</td>
      <td>890288.41</td>
      <td>365874.78</td>
      <td>475591.80</td>
      <td>388053.47</td>
      <td>...</td>
      <td>166137.05</td>
      <td>165118.80</td>
      <td>193548.75</td>
      <td>197937.55</td>
      <td>291699.69</td>
      <td>478857.58</td>
      <td>322863.09</td>
      <td>258448.66</td>
      <td>nan</td>
      <td>248652.26</td>
    </tr>
    <tr>
      <th>2019-08-01</th>
      <td>813769.59</td>
      <td>299421.26</td>
      <td>528576.61</td>
      <td>337522.60</td>
      <td>488784.36</td>
      <td>442189.47</td>
      <td>863170.61</td>
      <td>364540.25</td>
      <td>474627.27</td>
      <td>393032.64</td>
      <td>...</td>
      <td>167490.22</td>
      <td>165441.00</td>
      <td>195267.89</td>
      <td>200857.89</td>
      <td>290622.52</td>
      <td>472872.27</td>
      <td>323121.71</td>
      <td>258240.72</td>
      <td>nan</td>
      <td>249293.89</td>
    </tr>
    <tr>
      <th>2019-09-01</th>
      <td>810454.64</td>
      <td>304778.05</td>
      <td>526669.78</td>
      <td>333339.95</td>
      <td>501532.60</td>
      <td>441057.50</td>
      <td>838169.61</td>
      <td>365225.81</td>
      <td>473161.88</td>
      <td>386806.01</td>
      <td>...</td>
      <td>166796.48</td>
      <td>165407.34</td>
      <td>194807.97</td>
      <td>200969.07</td>
      <td>291944.46</td>
      <td>477779.44</td>
      <td>324357.13</td>
      <td>258750.94</td>
      <td>nan</td>
      <td>249770.51</td>
    </tr>
    <tr>
      <th>2019-10-01</th>
      <td>826227.06</td>
      <td>304579.06</td>
      <td>525678.42</td>
      <td>332919.67</td>
      <td>494770.26</td>
      <td>439178.40</td>
      <td>804713.12</td>
      <td>364412.83</td>
      <td>477369.14</td>
      <td>392720.54</td>
      <td>...</td>
      <td>166640.19</td>
      <td>166494.03</td>
      <td>194674.41</td>
      <td>200115.78</td>
      <td>291195.94</td>
      <td>474155.82</td>
      <td>323278.30</td>
      <td>258879.51</td>
      <td>nan</td>
      <td>249151.91</td>
    </tr>
    <tr>
      <th>2019-11-01</th>
      <td>776894.40</td>
      <td>306390.26</td>
      <td>522639.01</td>
      <td>333657.48</td>
      <td>432187.70</td>
      <td>436079.63</td>
      <td>825336.31</td>
      <td>367584.56</td>
      <td>475491.60</td>
      <td>393255.11</td>
      <td>...</td>
      <td>165808.12</td>
      <td>164499.35</td>
      <td>194855.23</td>
      <td>200895.65</td>
      <td>288781.44</td>
      <td>468982.76</td>
      <td>322511.72</td>
      <td>256874.86</td>
      <td>nan</td>
      <td>247950.71</td>
    </tr>
    <tr>
      <th>2019-12-01</th>
      <td>737275.03</td>
      <td>301283.44</td>
      <td>519306.24</td>
      <td>336301.87</td>
      <td>427126.35</td>
      <td>438681.81</td>
      <td>807124.19</td>
      <td>369567.88</td>
      <td>469662.03</td>
      <td>400182.39</td>
      <td>...</td>
      <td>164892.63</td>
      <td>165818.11</td>
      <td>194552.67</td>
      <td>198644.06</td>
      <td>290813.76</td>
      <td>478576.32</td>
      <td>321971.53</td>
      <td>256576.85</td>
      <td>nan</td>
      <td>248250.31</td>
    </tr>
    <tr>
      <th>2020-01-01</th>
      <td>747610.87</td>
      <td>303653.40</td>
      <td>518542.25</td>
      <td>334765.06</td>
      <td>423160.84</td>
      <td>435532.43</td>
      <td>815512.46</td>
      <td>371226.72</td>
      <td>466490.65</td>
      <td>390589.26</td>
      <td>...</td>
      <td>165093.52</td>
      <td>164752.36</td>
      <td>196199.08</td>
      <td>202084.82</td>
      <td>290334.93</td>
      <td>478489.13</td>
      <td>323421.57</td>
      <td>257822.15</td>
      <td>nan</td>
      <td>248950.46</td>
    </tr>
    <tr>
      <th>2020-02-01</th>
      <td>777640.01</td>
      <td>304265.15</td>
      <td>519121.24</td>
      <td>337759.69</td>
      <td>467144.91</td>
      <td>435704.45</td>
      <td>825249.40</td>
      <td>371356.53</td>
      <td>465613.85</td>
      <td>389901.43</td>
      <td>...</td>
      <td>166290.98</td>
      <td>164949.21</td>
      <td>194213.96</td>
      <td>200196.37</td>
      <td>290308.73</td>
      <td>479628.13</td>
      <td>319827.02</td>
      <td>257101.38</td>
      <td>nan</td>
      <td>248231.66</td>
    </tr>
    <tr>
      <th>2020-03-01</th>
      <td>844988.51</td>
      <td>304098.64</td>
      <td>527747.37</td>
      <td>339215.32</td>
      <td>461397.54</td>
      <td>434624.83</td>
      <td>870106.85</td>
      <td>370871.54</td>
      <td>478565.23</td>
      <td>395082.12</td>
      <td>...</td>
      <td>168160.92</td>
      <td>165684.57</td>
      <td>198283.71</td>
      <td>201368.71</td>
      <td>291355.54</td>
      <td>488184.74</td>
      <td>326700.98</td>
      <td>262444.14</td>
      <td>nan</td>
      <td>251538.79</td>
    </tr>
    <tr>
      <th>2020-04-01</th>
      <td>867840.63</td>
      <td>283139.44</td>
      <td>526552.50</td>
      <td>346839.54</td>
      <td>494454.52</td>
      <td>443481.92</td>
      <td>870626.76</td>
      <td>386737.93</td>
      <td>502107.07</td>
      <td>400695.13</td>
      <td>...</td>
      <td>167808.52</td>
      <td>165561.19</td>
      <td>200513.17</td>
      <td>202093.37</td>
      <td>295639.93</td>
      <td>480425.39</td>
      <td>327412.72</td>
      <td>255891.14</td>
      <td>nan</td>
      <td>250873.98</td>
    </tr>
  </tbody>
</table>
<p>305 rows × 48 columns</p>
</div>




```python
#Find the shape of the dataset
properties.shape
```




    (305, 48)




```python
#Check data types of columns
properties.dtypes
```




    City of London           object
    Barking & Dagenham       object
    Barnet                   object
    Bexley                   object
    Brent                    object
    Bromley                  object
    Camden                   object
    Croydon                  object
    Ealing                   object
    Enfield                  object
    Greenwich                object
    Hackney                  object
    Hammersmith & Fulham     object
    Haringey                 object
    Harrow                   object
    Havering                 object
    Hillingdon               object
    Hounslow                 object
    Islington                object
    Kensington & Chelsea     object
    Kingston upon Thames     object
    Lambeth                  object
    Lewisham                 object
    Merton                   object
    Newham                   object
    Redbridge                object
    Richmond upon Thames     object
    Southwark                object
    Sutton                   object
    Tower Hamlets            object
    Waltham Forest           object
    Wandsworth               object
    Westminster              object
    Unnamed: 33             float64
    Inner London             object
    Outer London             object
    Unnamed: 36             float64
    NORTH EAST               object
    NORTH WEST               object
    YORKS & THE HUMBER       object
    EAST MIDLANDS            object
    WEST MIDLANDS            object
    EAST OF ENGLAND          object
    LONDON                   object
    SOUTH EAST               object
    SOUTH WEST               object
    Unnamed: 46             float64
    England                  object
    dtype: object




```python
#Check for NaNs in dataframe
properties.isnull().values.any()
```




    True




```python
#Check first five rows
properties.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City of London</th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>Croydon</th>
      <th>Ealing</th>
      <th>Enfield</th>
      <th>...</th>
      <th>NORTH WEST</th>
      <th>YORKS &amp; THE HUMBER</th>
      <th>EAST MIDLANDS</th>
      <th>WEST MIDLANDS</th>
      <th>EAST OF ENGLAND</th>
      <th>LONDON</th>
      <th>SOUTH EAST</th>
      <th>SOUTH WEST</th>
      <th>Unnamed: 46</th>
      <th>England</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NaT</th>
      <td>E09000001</td>
      <td>E09000002</td>
      <td>E09000003</td>
      <td>E09000004</td>
      <td>E09000005</td>
      <td>E09000006</td>
      <td>E09000007</td>
      <td>E09000008</td>
      <td>E09000009</td>
      <td>E09000010</td>
      <td>...</td>
      <td>E12000002</td>
      <td>E12000003</td>
      <td>E12000004</td>
      <td>E12000005</td>
      <td>E12000006</td>
      <td>E12000007</td>
      <td>E12000008</td>
      <td>E12000009</td>
      <td>nan</td>
      <td>E92000001</td>
    </tr>
    <tr>
      <th>1995-01-01</th>
      <td>91448.98</td>
      <td>50460.23</td>
      <td>93284.52</td>
      <td>64958.09</td>
      <td>71306.57</td>
      <td>81671.48</td>
      <td>120932.89</td>
      <td>69158.16</td>
      <td>79885.89</td>
      <td>72514.69</td>
      <td>...</td>
      <td>43958.48</td>
      <td>44803.43</td>
      <td>45544.52</td>
      <td>48527.52</td>
      <td>56701.60</td>
      <td>74435.76</td>
      <td>64018.88</td>
      <td>54705.16</td>
      <td>nan</td>
      <td>53202.77</td>
    </tr>
    <tr>
      <th>1995-02-01</th>
      <td>82202.77</td>
      <td>51085.78</td>
      <td>93190.17</td>
      <td>64787.92</td>
      <td>72022.26</td>
      <td>81657.56</td>
      <td>119508.86</td>
      <td>68951.10</td>
      <td>80897.07</td>
      <td>73155.20</td>
      <td>...</td>
      <td>43925.42</td>
      <td>44528.81</td>
      <td>46051.57</td>
      <td>49341.29</td>
      <td>56593.59</td>
      <td>72777.94</td>
      <td>63715.02</td>
      <td>54356.15</td>
      <td>nan</td>
      <td>53096.15</td>
    </tr>
    <tr>
      <th>1995-03-01</th>
      <td>79120.70</td>
      <td>51268.97</td>
      <td>92247.52</td>
      <td>64367.49</td>
      <td>72015.76</td>
      <td>81449.31</td>
      <td>120282.21</td>
      <td>68712.44</td>
      <td>81379.86</td>
      <td>72190.44</td>
      <td>...</td>
      <td>44434.87</td>
      <td>45200.47</td>
      <td>45383.82</td>
      <td>49442.18</td>
      <td>56171.18</td>
      <td>73896.84</td>
      <td>64113.61</td>
      <td>53583.08</td>
      <td>nan</td>
      <td>53201.28</td>
    </tr>
    <tr>
      <th>1995-04-01</th>
      <td>77101.21</td>
      <td>53133.51</td>
      <td>90762.87</td>
      <td>64277.67</td>
      <td>72965.63</td>
      <td>81124.41</td>
      <td>120097.90</td>
      <td>68610.05</td>
      <td>82188.90</td>
      <td>71442.92</td>
      <td>...</td>
      <td>44267.78</td>
      <td>45614.34</td>
      <td>46124.23</td>
      <td>49455.93</td>
      <td>56567.90</td>
      <td>74455.29</td>
      <td>64623.22</td>
      <td>54786.02</td>
      <td>nan</td>
      <td>53590.85</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



**2.2. Cleaning the data**

You might find you need to transpose your dataframe, check out what its row indexes are, and reset the index. You  also might find you need to assign the values of the first row to your column headings  . (Hint: recall the .columns feature of DataFrames, as well as the iloc[] method).

Don't be afraid to use StackOverflow for help  with this.


```python
#Transpose dataframe
properties_t = properties.transpose()

#Check row indexes
properties_t.head()
properties_t.index
```




    Index(['City of London', 'Barking & Dagenham', 'Barnet', 'Bexley', 'Brent',
           'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich',
           'Hackney', 'Hammersmith & Fulham', 'Haringey', 'Harrow', 'Havering',
           'Hillingdon', 'Hounslow', 'Islington', 'Kensington & Chelsea',
           'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham',
           'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton',
           'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 'Westminster',
           'Unnamed: 33', 'Inner London', 'Outer London', 'Unnamed: 36',
           'NORTH EAST', 'NORTH WEST', 'YORKS & THE HUMBER', 'EAST MIDLANDS',
           'WEST MIDLANDS', 'EAST OF ENGLAND', 'LONDON', 'SOUTH EAST',
           'SOUTH WEST', 'Unnamed: 46', 'England'],
          dtype='object')




```python
#Reset index
properties_t_ri = properties_t.reset_index()
properties_t_ri.index
```




    RangeIndex(start=0, stop=48, step=1)




```python
properties_t_ri.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>NaT</th>
      <th>1995-01-01 00:00:00</th>
      <th>1995-02-01 00:00:00</th>
      <th>1995-03-01 00:00:00</th>
      <th>1995-04-01 00:00:00</th>
      <th>1995-05-01 00:00:00</th>
      <th>1995-06-01 00:00:00</th>
      <th>1995-07-01 00:00:00</th>
      <th>1995-08-01 00:00:00</th>
      <th>...</th>
      <th>2019-07-01 00:00:00</th>
      <th>2019-08-01 00:00:00</th>
      <th>2019-09-01 00:00:00</th>
      <th>2019-10-01 00:00:00</th>
      <th>2019-11-01 00:00:00</th>
      <th>2019-12-01 00:00:00</th>
      <th>2020-01-01 00:00:00</th>
      <th>2020-02-01 00:00:00</th>
      <th>2020-03-01 00:00:00</th>
      <th>2020-04-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>91448.98</td>
      <td>82202.77</td>
      <td>79120.70</td>
      <td>77101.21</td>
      <td>84409.15</td>
      <td>94900.51</td>
      <td>110128.04</td>
      <td>112329.44</td>
      <td>...</td>
      <td>756407.02</td>
      <td>813769.59</td>
      <td>810454.64</td>
      <td>826227.06</td>
      <td>776894.40</td>
      <td>737275.03</td>
      <td>747610.87</td>
      <td>777640.01</td>
      <td>844988.51</td>
      <td>867840.63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>50460.23</td>
      <td>51085.78</td>
      <td>51268.97</td>
      <td>53133.51</td>
      <td>53042.25</td>
      <td>53700.35</td>
      <td>52113.12</td>
      <td>52232.20</td>
      <td>...</td>
      <td>297426.44</td>
      <td>299421.26</td>
      <td>304778.05</td>
      <td>304579.06</td>
      <td>306390.26</td>
      <td>301283.44</td>
      <td>303653.40</td>
      <td>304265.15</td>
      <td>304098.64</td>
      <td>283139.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>93284.52</td>
      <td>93190.17</td>
      <td>92247.52</td>
      <td>90762.87</td>
      <td>90258.00</td>
      <td>90107.23</td>
      <td>91441.25</td>
      <td>92361.32</td>
      <td>...</td>
      <td>514667.76</td>
      <td>528576.61</td>
      <td>526669.78</td>
      <td>525678.42</td>
      <td>522639.01</td>
      <td>519306.24</td>
      <td>518542.25</td>
      <td>519121.24</td>
      <td>527747.37</td>
      <td>526552.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>64958.09</td>
      <td>64787.92</td>
      <td>64367.49</td>
      <td>64277.67</td>
      <td>63997.14</td>
      <td>64252.32</td>
      <td>63722.70</td>
      <td>64432.60</td>
      <td>...</td>
      <td>338346.40</td>
      <td>337522.60</td>
      <td>333339.95</td>
      <td>332919.67</td>
      <td>333657.48</td>
      <td>336301.87</td>
      <td>334765.06</td>
      <td>337759.69</td>
      <td>339215.32</td>
      <td>346839.54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>71306.57</td>
      <td>72022.26</td>
      <td>72015.76</td>
      <td>72965.63</td>
      <td>73704.05</td>
      <td>74310.48</td>
      <td>74127.04</td>
      <td>73547.04</td>
      <td>...</td>
      <td>473849.04</td>
      <td>488784.36</td>
      <td>501532.60</td>
      <td>494770.26</td>
      <td>432187.70</td>
      <td>427126.35</td>
      <td>423160.84</td>
      <td>467144.91</td>
      <td>461397.54</td>
      <td>494454.52</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 306 columns</p>
</div>




```python
properties_t_ri.columns
```




    Index([            'index',                 NaT, 1995-01-01 00:00:00,
           1995-02-01 00:00:00, 1995-03-01 00:00:00, 1995-04-01 00:00:00,
           1995-05-01 00:00:00, 1995-06-01 00:00:00, 1995-07-01 00:00:00,
           1995-08-01 00:00:00,
           ...
           2019-07-01 00:00:00, 2019-08-01 00:00:00, 2019-09-01 00:00:00,
           2019-10-01 00:00:00, 2019-11-01 00:00:00, 2019-12-01 00:00:00,
           2020-01-01 00:00:00, 2020-02-01 00:00:00, 2020-03-01 00:00:00,
           2020-04-01 00:00:00],
          dtype='object', length=306)



**2.3. Cleaning the data (part 2)**

You might we have to **rename** a couple columns. How do you do this? The clue's pretty bold...


```python
properties_t_ri = properties_t_ri.rename(columns = {'index': 'Boroughs', pd.NaT: 'PostalCode'})
properties_t_ri.columns
```




    Index([         'Boroughs',        'PostalCode', 1995-01-01 00:00:00,
           1995-02-01 00:00:00, 1995-03-01 00:00:00, 1995-04-01 00:00:00,
           1995-05-01 00:00:00, 1995-06-01 00:00:00, 1995-07-01 00:00:00,
           1995-08-01 00:00:00,
           ...
           2019-07-01 00:00:00, 2019-08-01 00:00:00, 2019-09-01 00:00:00,
           2019-10-01 00:00:00, 2019-11-01 00:00:00, 2019-12-01 00:00:00,
           2020-01-01 00:00:00, 2020-02-01 00:00:00, 2020-03-01 00:00:00,
           2020-04-01 00:00:00],
          dtype='object', length=306)



**2.4.Transforming the data**

Remember what Wes McKinney said about tidy data? 

You might need to **melt** your DataFrame here. 


```python
properties_melt = pd.melt(properties_t_ri, id_vars = ['Boroughs','PostalCode'])
properties_melt.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boroughs</th>
      <th>PostalCode</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.57</td>
    </tr>
  </tbody>
</table>
</div>




```python
properties_melt_2 = properties_melt.rename(columns = {'variable' : 'Month', 'value' : 'AveragePrice'})
properties_melt_2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boroughs</th>
      <th>PostalCode</th>
      <th>Month</th>
      <th>AveragePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.57</td>
    </tr>
  </tbody>
</table>
</div>



Remember to make sure your column data types are all correct. Average prices, for example, should be floating point numbers... 


```python
pd.to_numeric(properties_melt_2['AveragePrice']) # convert everything to float values
pd.options.display.float_format = '{:.2f}'.format
properties_melt_2.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boroughs</th>
      <th>PostalCode</th>
      <th>Month</th>
      <th>AveragePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.57</td>
    </tr>
  </tbody>
</table>
</div>



**2.5. Cleaning the data (part 3)**

Do we have an equal number of observations in the ID, Average Price, Month, and London Borough columns? Remember that there are only 32 London Boroughs. How many entries do you have in that column? 

Check out the contents of the London Borough column, and if you find null values, get rid of them however you see fit. 


```python
#Get rid of null values
properties_clean = properties_melt_2.dropna(axis=0)
properties_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boroughs</th>
      <th>PostalCode</th>
      <th>Month</th>
      <th>AveragePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.52</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.57</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Bromley</td>
      <td>E09000006</td>
      <td>1995-01-01</td>
      <td>81671.48</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Camden</td>
      <td>E09000007</td>
      <td>1995-01-01</td>
      <td>120932.89</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Croydon</td>
      <td>E09000008</td>
      <td>1995-01-01</td>
      <td>69158.16</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ealing</td>
      <td>E09000009</td>
      <td>1995-01-01</td>
      <td>79885.89</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Enfield</td>
      <td>E09000010</td>
      <td>1995-01-01</td>
      <td>72514.69</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Greenwich</td>
      <td>E09000011</td>
      <td>1995-01-01</td>
      <td>62300.10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Hackney</td>
      <td>E09000012</td>
      <td>1995-01-01</td>
      <td>61296.53</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Hammersmith &amp; Fulham</td>
      <td>E09000013</td>
      <td>1995-01-01</td>
      <td>124902.86</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Haringey</td>
      <td>E09000014</td>
      <td>1995-01-01</td>
      <td>76287.57</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harrow</td>
      <td>E09000015</td>
      <td>1995-01-01</td>
      <td>84769.53</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Havering</td>
      <td>E09000016</td>
      <td>1995-01-01</td>
      <td>68000.14</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Hillingdon</td>
      <td>E09000017</td>
      <td>1995-01-01</td>
      <td>73834.83</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Hounslow</td>
      <td>E09000018</td>
      <td>1995-01-01</td>
      <td>72231.71</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Islington</td>
      <td>E09000019</td>
      <td>1995-01-01</td>
      <td>92516.49</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kensington &amp; Chelsea</td>
      <td>E09000020</td>
      <td>1995-01-01</td>
      <td>182694.83</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Kingston upon Thames</td>
      <td>E09000021</td>
      <td>1995-01-01</td>
      <td>80875.85</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Lambeth</td>
      <td>E09000022</td>
      <td>1995-01-01</td>
      <td>67770.99</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Lewisham</td>
      <td>E09000023</td>
      <td>1995-01-01</td>
      <td>60491.26</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Merton</td>
      <td>E09000024</td>
      <td>1995-01-01</td>
      <td>82070.61</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Newham</td>
      <td>E09000025</td>
      <td>1995-01-01</td>
      <td>53539.32</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Redbridge</td>
      <td>E09000026</td>
      <td>1995-01-01</td>
      <td>72189.58</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Richmond upon Thames</td>
      <td>E09000027</td>
      <td>1995-01-01</td>
      <td>109326.12</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Southwark</td>
      <td>E09000028</td>
      <td>1995-01-01</td>
      <td>67885.20</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sutton</td>
      <td>E09000029</td>
      <td>1995-01-01</td>
      <td>71536.97</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Tower Hamlets</td>
      <td>E09000030</td>
      <td>1995-01-01</td>
      <td>59865.19</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14559</th>
      <td>Havering</td>
      <td>E09000016</td>
      <td>2020-04-01</td>
      <td>378746.98</td>
    </tr>
    <tr>
      <th>14560</th>
      <td>Hillingdon</td>
      <td>E09000017</td>
      <td>2020-04-01</td>
      <td>405066.32</td>
    </tr>
    <tr>
      <th>14561</th>
      <td>Hounslow</td>
      <td>E09000018</td>
      <td>2020-04-01</td>
      <td>396029.79</td>
    </tr>
    <tr>
      <th>14562</th>
      <td>Islington</td>
      <td>E09000019</td>
      <td>2020-04-01</td>
      <td>706263.09</td>
    </tr>
    <tr>
      <th>14563</th>
      <td>Kensington &amp; Chelsea</td>
      <td>E09000020</td>
      <td>2020-04-01</td>
      <td>1348950.96</td>
    </tr>
    <tr>
      <th>14564</th>
      <td>Kingston upon Thames</td>
      <td>E09000021</td>
      <td>2020-04-01</td>
      <td>477449.66</td>
    </tr>
    <tr>
      <th>14565</th>
      <td>Lambeth</td>
      <td>E09000022</td>
      <td>2020-04-01</td>
      <td>513396.68</td>
    </tr>
    <tr>
      <th>14566</th>
      <td>Lewisham</td>
      <td>E09000023</td>
      <td>2020-04-01</td>
      <td>400986.80</td>
    </tr>
    <tr>
      <th>14567</th>
      <td>Merton</td>
      <td>E09000024</td>
      <td>2020-04-01</td>
      <td>529856.66</td>
    </tr>
    <tr>
      <th>14568</th>
      <td>Newham</td>
      <td>E09000025</td>
      <td>2020-04-01</td>
      <td>356915.37</td>
    </tr>
    <tr>
      <th>14569</th>
      <td>Redbridge</td>
      <td>E09000026</td>
      <td>2020-04-01</td>
      <td>435763.14</td>
    </tr>
    <tr>
      <th>14570</th>
      <td>Richmond upon Thames</td>
      <td>E09000027</td>
      <td>2020-04-01</td>
      <td>683247.81</td>
    </tr>
    <tr>
      <th>14571</th>
      <td>Southwark</td>
      <td>E09000028</td>
      <td>2020-04-01</td>
      <td>508543.07</td>
    </tr>
    <tr>
      <th>14572</th>
      <td>Sutton</td>
      <td>E09000029</td>
      <td>2020-04-01</td>
      <td>376808.05</td>
    </tr>
    <tr>
      <th>14573</th>
      <td>Tower Hamlets</td>
      <td>E09000030</td>
      <td>2020-04-01</td>
      <td>480983.84</td>
    </tr>
    <tr>
      <th>14574</th>
      <td>Waltham Forest</td>
      <td>E09000031</td>
      <td>2020-04-01</td>
      <td>440354.13</td>
    </tr>
    <tr>
      <th>14575</th>
      <td>Wandsworth</td>
      <td>E09000032</td>
      <td>2020-04-01</td>
      <td>607227.47</td>
    </tr>
    <tr>
      <th>14576</th>
      <td>Westminster</td>
      <td>E09000033</td>
      <td>2020-04-01</td>
      <td>1034488.20</td>
    </tr>
    <tr>
      <th>14578</th>
      <td>Inner London</td>
      <td>E13000001</td>
      <td>2020-04-01</td>
      <td>584234.12</td>
    </tr>
    <tr>
      <th>14579</th>
      <td>Outer London</td>
      <td>E13000002</td>
      <td>2020-04-01</td>
      <td>431880.49</td>
    </tr>
    <tr>
      <th>14581</th>
      <td>NORTH EAST</td>
      <td>E12000001</td>
      <td>2020-04-01</td>
      <td>125937.87</td>
    </tr>
    <tr>
      <th>14582</th>
      <td>NORTH WEST</td>
      <td>E12000002</td>
      <td>2020-04-01</td>
      <td>167808.52</td>
    </tr>
    <tr>
      <th>14583</th>
      <td>YORKS &amp; THE HUMBER</td>
      <td>E12000003</td>
      <td>2020-04-01</td>
      <td>165561.19</td>
    </tr>
    <tr>
      <th>14584</th>
      <td>EAST MIDLANDS</td>
      <td>E12000004</td>
      <td>2020-04-01</td>
      <td>200513.17</td>
    </tr>
    <tr>
      <th>14585</th>
      <td>WEST MIDLANDS</td>
      <td>E12000005</td>
      <td>2020-04-01</td>
      <td>202093.37</td>
    </tr>
    <tr>
      <th>14586</th>
      <td>EAST OF ENGLAND</td>
      <td>E12000006</td>
      <td>2020-04-01</td>
      <td>295639.93</td>
    </tr>
    <tr>
      <th>14587</th>
      <td>LONDON</td>
      <td>E12000007</td>
      <td>2020-04-01</td>
      <td>480425.39</td>
    </tr>
    <tr>
      <th>14588</th>
      <td>SOUTH EAST</td>
      <td>E12000008</td>
      <td>2020-04-01</td>
      <td>327412.72</td>
    </tr>
    <tr>
      <th>14589</th>
      <td>SOUTH WEST</td>
      <td>E12000009</td>
      <td>2020-04-01</td>
      <td>255891.14</td>
    </tr>
    <tr>
      <th>14591</th>
      <td>England</td>
      <td>E92000001</td>
      <td>2020-04-01</td>
      <td>250873.98</td>
    </tr>
  </tbody>
</table>
<p>13680 rows × 4 columns</p>
</div>



**2.6. Visualizing the data**

To visualize the data, why not subset on a particular London Borough? Maybe do a line plot of Month against Average Price?


```python
greenwich_prices = properties_clean[properties_clean['Boroughs'] == 'Greenwich']
greenwich_prices.plot.line(x='Month', y='AveragePrice')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15b279f8a20>




![png](output_29_1.png)


To limit the number of data points you have, you might want to extract the year from every month value your *Month* column. 

To this end, you *could* apply a ***lambda function***. Your logic could work as follows:
1. look through the `Month` column
2. extract the year from each individual value in that column 
3. store that corresponding year as separate column. 

Whether you go ahead with this is up to you. Just so long as you answer our initial brief: which boroughs of London have seen the greatest house price increase, on average, over the past two decades? 


```python
properties_clean['Year'] = properties_clean['Month'].apply(lambda t: t.year)
```

    C:\Users\saraa\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    


```python
properties_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boroughs</th>
      <th>PostalCode</th>
      <th>Month</th>
      <th>AveragePrice</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>City of London</td>
      <td>E09000001</td>
      <td>1995-01-01</td>
      <td>91448.98</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>E09000002</td>
      <td>1995-01-01</td>
      <td>50460.23</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barnet</td>
      <td>E09000003</td>
      <td>1995-01-01</td>
      <td>93284.52</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bexley</td>
      <td>E09000004</td>
      <td>1995-01-01</td>
      <td>64958.09</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brent</td>
      <td>E09000005</td>
      <td>1995-01-01</td>
      <td>71306.57</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
properties_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 13680 entries, 0 to 14591
    Data columns (total 5 columns):
    Boroughs        13680 non-null object
    PostalCode      13680 non-null object
    Month           13680 non-null datetime64[ns]
    AveragePrice    13680 non-null object
    Year            13680 non-null int64
    dtypes: datetime64[ns](1), int64(1), object(3)
    memory usage: 641.2+ KB
    


```python
properties_clean['AveragePrice']=pd.to_numeric(properties_clean['AveragePrice']) # convert everything to float values
pd.options.display.float_format = '{:.2f}'.format
```

    C:\Users\saraa\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    


```python
properties_clean = properties_clean.groupby(by=['Boroughs', 'Year']).mean()
```


```python
properties_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>AveragePrice</th>
    </tr>
    <tr>
      <th>Boroughs</th>
      <th>Year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Barking &amp; Dagenham</th>
      <th>1995</th>
      <td>51817.97</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>51718.19</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>55974.26</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>60285.82</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>65320.93</td>
    </tr>
  </tbody>
</table>
</div>



**3. Modeling**

Consider creating a function that will calculate a ratio of house prices, comparing the price of a house in 2018 to the price in 1998.

Consider calling this function create_price_ratio.

You'd want this function to:
1. Take a filter of dfg, specifically where this filter constrains the London_Borough, as an argument. For example, one admissible argument should be: dfg[dfg['London_Borough']=='Camden'].
2. Get the Average Price for that Borough, for the years 1998 and 2018.
4. Calculate the ratio of the Average Price for 1998 divided by the Average Price for 2018.
5. Return that ratio.

Once you've written this function, you ultimately want to use it to iterate through all the unique London_Boroughs and work out the ratio capturing the difference of house prices between 1998 and 2018.

Bear in mind: you don't have to write a function like this if you don't want to. If you can solve the brief otherwise, then great! 

***Hint***: This section should test the skills you acquired in:
- Python Data Science Toolbox - Part One, all modules


```python
def create_price_ratio(properties_clean):
    avg1998 = float(properties_clean['AveragePrice'][properties_clean['Year']==1998])
    avg2018 = float(properties_clean['AveragePrice'][properties_clean['Year']==2018])
    ratio = [avg2018 / avg1998]
    return ratio
```


```python
properties_clean_ri = properties_clean.reset_index()
```


```python
create_price_ratio(properties_clean_ri[properties_clean_ri['Boroughs']=='Barking & Dagenham'])
```




    [4.89661861291754]




```python
properties_clean_ri.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Boroughs</th>
      <th>Year</th>
      <th>AveragePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking &amp; Dagenham</td>
      <td>1995</td>
      <td>51817.97</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barking &amp; Dagenham</td>
      <td>1996</td>
      <td>51718.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Barking &amp; Dagenham</td>
      <td>1997</td>
      <td>55974.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Barking &amp; Dagenham</td>
      <td>1998</td>
      <td>60285.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Barking &amp; Dagenham</td>
      <td>1999</td>
      <td>65320.93</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_boroughs = {}

for b in properties_clean_ri['Boroughs'].unique():
    borough = properties_clean_ri[properties_clean_ri['Boroughs'] == b]
    all_boroughs[b] = create_price_ratio(borough)
print(all_boroughs)
```

    {'Barking & Dagenham': [4.89661861291754], 'Barnet': [4.358195917538044], 'Bexley': [4.248977046127877], 'Brent': [4.8945544971392865], 'Bromley': [4.0947846853338765], 'Camden': [4.935353408884261], 'City of London': [5.301620377587609], 'Croydon': [4.201100280024767], 'EAST MIDLANDS': [3.6327346720877034], 'EAST OF ENGLAND': [4.166900547724156], 'Ealing': [4.311450902121834], 'Enfield': [4.26347158349581], 'England': [3.8104529783974], 'Greenwich': [4.763036347329193], 'Hackney': [6.198285561008662], 'Hammersmith & Fulham': [4.137798101936229], 'Haringey': [5.134624964136042], 'Harrow': [4.0591964329643195], 'Havering': [4.325230371335308], 'Hillingdon': [4.2002730803844575], 'Hounslow': [3.976409106143329], 'Inner London': [5.170857506254785], 'Islington': [4.844048012802298], 'Kensington & Chelsea': [5.082465066092464], 'Kingston upon Thames': [4.270549521484271], 'LONDON': [4.679776249632861], 'Lambeth': [4.957751163514063], 'Lewisham': [5.449221041059685], 'Merton': [4.741273313294604], 'NORTH EAST': [2.828080506434263], 'NORTH WEST': [3.3634156376540654], 'Newham': [5.305390437201879], 'Outer London': [4.418949809440314], 'Redbridge': [4.471182006097364], 'Richmond upon Thames': [4.005161895721457], 'SOUTH EAST': [3.8283877112840563], 'SOUTH WEST': [3.795497124092444], 'Southwark': [5.516485302379376], 'Sutton': [4.118522608573157], 'Tower Hamlets': [4.62670104006116], 'WEST MIDLANDS': [3.3112621931400104], 'Waltham Forest': [5.83475580932281], 'Wandsworth': [4.757709347739269], 'Westminster': [5.353565392605413], 'YORKS & THE HUMBER': [3.356065086028382]}
    


```python
properties_clean_ri = pd.DataFrame(all_boroughs)
properties_clean_ri
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Barking &amp; Dagenham</th>
      <th>Barnet</th>
      <th>Bexley</th>
      <th>Brent</th>
      <th>Bromley</th>
      <th>Camden</th>
      <th>City of London</th>
      <th>Croydon</th>
      <th>EAST MIDLANDS</th>
      <th>EAST OF ENGLAND</th>
      <th>...</th>
      <th>SOUTH EAST</th>
      <th>SOUTH WEST</th>
      <th>Southwark</th>
      <th>Sutton</th>
      <th>Tower Hamlets</th>
      <th>WEST MIDLANDS</th>
      <th>Waltham Forest</th>
      <th>Wandsworth</th>
      <th>Westminster</th>
      <th>YORKS &amp; THE HUMBER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4.90</td>
      <td>4.36</td>
      <td>4.25</td>
      <td>4.89</td>
      <td>4.09</td>
      <td>4.94</td>
      <td>5.30</td>
      <td>4.20</td>
      <td>3.63</td>
      <td>4.17</td>
      <td>...</td>
      <td>3.83</td>
      <td>3.80</td>
      <td>5.52</td>
      <td>4.12</td>
      <td>4.63</td>
      <td>3.31</td>
      <td>5.83</td>
      <td>4.76</td>
      <td>5.35</td>
      <td>3.36</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 45 columns</p>
</div>




```python
properties_ratios_T = properties_clean_ri.T
properties_ratios = properties_ratios_T.reset_index()
properties_ratios.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking &amp; Dagenham</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>4.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>4.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>4.09</td>
    </tr>
  </tbody>
</table>
</div>




```python
properties_ratios.rename(columns={'index':'Borough', 0:'2018'}, inplace=True)
properties_ratios.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Barking &amp; Dagenham</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Barnet</td>
      <td>4.36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bexley</td>
      <td>4.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brent</td>
      <td>4.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bromley</td>
      <td>4.09</td>
    </tr>
  </tbody>
</table>
</div>




```python
top10 = properties_ratios.sort_values(by='2018',ascending=False).head(10)
print(top10)
```

                     Borough  2018
    14               Hackney  6.20
    41        Waltham Forest  5.83
    37             Southwark  5.52
    27              Lewisham  5.45
    43           Westminster  5.35
    31                Newham  5.31
    6         City of London  5.30
    21          Inner London  5.17
    16              Haringey  5.13
    23  Kensington & Chelsea  5.08
    


```python
bottom10 = properties_ratios.sort_values(by='2018',ascending=True).head(10)
print(bottom10)
```

                     Borough  2018
    29            NORTH EAST  2.83
    40         WEST MIDLANDS  3.31
    44    YORKS & THE HUMBER  3.36
    30            NORTH WEST  3.36
    8          EAST MIDLANDS  3.63
    36            SOUTH WEST  3.80
    12               England  3.81
    35            SOUTH EAST  3.83
    20              Hounslow  3.98
    34  Richmond upon Thames  4.01
    


```python
print(properties_ratios.describe())
```

           2018
    count 45.00
    mean   4.49
    std    0.70
    min    2.83
    25%    4.09
    50%    4.36
    75%    4.94
    max    6.20
    

### 4. Conclusion
What can you conclude? Type out your conclusion below. 

Look back at your notebook. Think about how you might summarize what you have done, and prepare a quick presentation on it to your mentor at your next meeting. 

We hope you enjoyed this practical project. It should have consolidated your data hygiene and pandas skills by looking at a real-world problem involving just the kind of dataset you might encounter as a budding data scientist. Congratulations, and looking forward to seeing you at the next step in the course! 

<h3>My Findings</h3>
From the descriptive statistics, the average ratio of 2018 average price to 1998 average price overall is almost 4.5, with a median ratio of 4.36 and a standard deviation of 0.70. The smallest ratio is 2.83, about 2 standard deviations below the median, and the largest is 6.20, just over 2 standard deviations above the mean. This means that the average house prices in most boroughs increased by 4 or 5 times from 1998 to 2018, while the top 10 saw housing prices rise by 6 times and the lower 10 saw housing prices rise by just under 3 to 4 times.
