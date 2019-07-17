# Nimbus
Interactive model based pricing UI

Reference: 
[[Our Demo Paper](https://github.com/chenleshang/Nimbus/blob/master/NimbusDemo_SIGMOD.pdf)]
[[Our Poster](https://github.com/chenleshang/Nimbus/blob/master/Nimbus_Poster.pdf)]
[[Full Paper](https://arxiv.org/pdf/1805.11450.pdf)]

Requirements: 

`pip install flask`

`pip install bokeh`

## Seller Upload
This is the view port for seller uploading files. 

### How to run seller upload
In ./SellerUpload, run these command in order: 

`export FLASK_APP=flasktest.py`

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
