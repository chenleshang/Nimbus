# Nimbus: Model-based Pricing for Machine Learning in a Data Marketplace

Interactive model based pricing UI Various domains such as business intelligence and journalism have made many achievements with help of data analytics based on machine learning (ML). While a lot of work has studied how to reduce the cost of training, storing, and deploying ML models, there is little work on eliminating the data collection and purchase cost. Existing data markets provide only simplistic mechanism allowing the sale of fixed datasets with fixed price, which potentially hurts not only ML model availability to buyers with limited budget, but market expansion and thus sellers’ revenue as well. In this work, we demonstrate Nimbus, a data market framework for ML model exchange. Instead of pricing data, Nimbus prices ML models directly, which we call model-based pricing (MBP). Through interactive interfaces, the audience can play the role of sellers to vend their own ML models with different price requirements, as well as the role of buyers to purchase ML model instances with different accuracy/budget constraints. We will further demonstrate how much gain of sellers’ revenue and buyers’ affordability Nimbus can achieve with low runtime cost via both real time and offline results.

Reference: 
[[Our Demo Paper](https://github.com/chenleshang/Nimbus/blob/master/NimbusDemo_SIGMOD.pdf)]
[[Our Poster](https://github.com/chenleshang/Nimbus/blob/master/Nimbus_Poster.pdf)]
[[Theories](https://arxiv.org/pdf/1805.11450.pdf)]

Requirements: 

`pip install flask`

`pip install bokeh`

## Seller Upload
This is the view port for seller uploading files. 

### How to run seller upload
In ./SellerUpload, run these command in order: 

`export FLASK_APP=sellerupload.py`

`flask run --port 5000`

Then open `localhost:5000`

## Seller
This is the view port for seller. 

### How to use the Bokeh app: 
Type in the following command from the root directory: 

`bokeh server Seller --port 5006`

Then open `localhost:5006` in browser. 

## Buyer
This is the view port for buyer. 

### How to use the Bokeh app: 
Type in the following command from the root directory: 

`bokeh server Buyer --port 5007`

Then open `localhost:5007` in browser. 

## How to enable remote access for Flask and Bokeh
Flask: `flask run --host=0.0.0.0`

Bokeh: `bokeh serve buyer --port <port> --allow-websocket-origin <your-ip>:<port>`
