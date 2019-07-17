# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 22:58:59 2018

@author: Leshang
"""

''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
import bokeh
from bokeh.events import ButtonClick
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox, column
from bokeh.models import ColumnDataSource, Select
from bokeh.models.widgets import Slider, TextInput, DataTable, DateFormatter, TableColumn, Button
from bokeh.plotting import figure
from bokeh.models import HoverTool, CustomJS, Div, TapTool, Range1d
from bokeh import events

from datetime import date
from random import randint
import pandas as pd
import os
from os.path import dirname, join
import sqlite3
import pickle

# Basic Settings
DATABASE = './SellerUpload/database.db'
DATABASE_INIT_FILE = './SellerUpload/schema.sql'
UPLOAD_FOLDER = './SellerUpload/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

# Set up data
#N = 200
#x = np.linspace(0, 4*np.pi, N)
#y = np.sin(x)

t = np.arange(0.1, 100.0, 0.1)
#        x = t
y = (t < 30) * 4.0 / 3 * t + (t >= 30) * (t < 50) * ( 1.0 / 2 * t + 25) + (t >= 50) * (45.0 / 50 * t + 5)

#        y = (t < 30) * 1.0 / (4.0 / 3 * t) + \
#            (t >= 30) * 1.0 / (t < 50) * ( 1.0 / 2 * t + 25) + \
#            (t >= 50) * 1.0 / (45.0 / 50 * t + 5)

#        y = 100 - 100.0 / x
x = 1.0 / t
xorder = np.argsort(x)
x_new = x[xorder]
xlen = int(x.shape[0] * 0.9 )

x_new = x_new[0:xlen]
y_new = y[xorder]
y_new = y_new[0:xlen]

#        t[0] = 3

x=x_new
y=y_new
#posx = 0
#posy = 0
source = ColumnDataSource(data=dict(error_approx=[], price=[]))
#position = ColumnDataSource(data = dict (posx = posx, posy = posy))

# Set up plot
#plot = figure(plot_height=420, plot_width=400, title="Price - Error Curve",
#              tools="crosshair,pan,reset,save,wheel_zoom,hover,tap",
#              x_range=[0, 0.10], y_range=[0, 100])

plot = figure(plot_height=390, plot_width=400, title="",
              tools="crosshair,pan,reset,save,wheel_zoom,box_zoom,hover,tap",
              x_range=[0.05, 0.15], y_range=[0, 1.2], toolbar_location='above',toolbar_sticky=True )

#plot.line('x', 'y', source=source, line_width=6, line_alpha=0.6, legend="Prive vs Err")
plot.line('error_approx', 'price', source=source, line_width=6, line_alpha=0.6)

# setting up axis name 
plot.xaxis.axis_label = 'Error'
plot.yaxis.axis_label = 'Price'
plot.xaxis.axis_label_text_font_size = "20pt"
plot.yaxis.axis_label_text_font_size = "20pt"
plot.title.text_font_size = "20pt"

plot.yaxis.major_label_text_font_size = "15pt"
plot.xaxis.major_label_text_font_size = "15pt"

# Get index to be returned on tap plot
def my_tap_handler(attr,old,new):
    index = source.selected.indices
    print(index)
    


# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)

modelList = ['Logistic Regression', 'Linear Regression']
kernelList = ['Default']
datasetList = ['Synthetic']#['Digit Recognition', 'Boston House-prices', 'Iris']
select = Select(title="Select ML model: ", value="Linear Regression", options=modelList, height=50, width=360)
kernelSelect = Select(title="Select kernel type: ", value="Default", options=kernelList, height=50, width=360)
datasetSelect = Select(title="Choose Dataset: ", value="Synthetic", options=datasetList, height=50, width=360)



#select3 = Select(title="Select ML model: ", value="Linear Regression", options=modelList)
#text_xpos = TextInput(title="POSITION", value='UNDEFINED')
# Set up callbacks
#def update_title(attrname, old, new):
#    plot.title.text = text.value
#
#text.on_change('value', update_title)

#def update_data(attrname, old, new):
#
#    # Get the current slider values
#    a = amplitude.value
#    b = offset.value
#    w = phase.value
#    k = freq.value
#
#    #TODO: Generate the new curve
##    x = np.linspace(0, 4*np.pi, N)
##    y = a*np.sin(k*x + w) + b
#
#    source.data = dict(x=x, y=y)
#
#for w in [offset, amplitude, phase, freq]:
#    w.on_change('value', update_data)

## backup code on tap callback
#txt = Div(text='Text will be updated with current time anytime anything '\
#          'happens to the ColumnDataSource')
#empty_line = Div()

#source_code = """
#var time = new Date()
#
#txt.text = time.getHours() + ":" + time.getMinutes() + ":" + time.getSeconds() + " x: " + cb_obj.x + " y: " + cb_obj.y;
#console.log(cb_obj);
#"""
#
#tapcallback = CustomJS(args={'txt':txt},code=source_code)
#plot.js_on_event(events.Tap, tapcallback)

# Set up table
data_to_fill = dict(
        dates=[date(2014, 3, i+1) for i in range(20)],
        downloads=[randint(0, 100) for i in range(20)],
    )
data_to_fill = dict(
        att1=np.random.randn(20),
        att2=np.random.randn(20),
        att3=np.random.randn(20),
    )
source_table = ColumnDataSource(data=dict(dates=[], downloads=[]))

table_columns = [
        TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        TableColumn(field="downloads", title="Downloads"),
    ]
table_columns = [
        TableColumn(field="att1", title="Attribute1", width=100),
        TableColumn(field="att2", title="Attribute2", width=100),
        TableColumn(field="att3", title="Attribute3", width=100)
    ]
data_table = DataTable(source=source_table, columns=table_columns, editable = True, fit_columns=False,
                       width=360, height=132)


# Set up layouts and add to document
#inputs = widgetbox(text, offset, amplitude, phase, freq, txt, select, kernelSelect, select2, select3)
# Update dataset lists
def update_dataset_lists(event):
    conn = sqlite3.connect(DATABASE)
    query_dataset = '''
        select distinct datasetname
        from csv
        '''
    cursor = conn.execute(query_dataset)
#    print cursor.fetchall()
    
    datasetSelect.options = [item[0] for item in cursor.fetchall()]
    if not datasetSelect.value in datasetSelect.options:
        datasetSelect.value = datasetSelect.options[0]
    
    if event is not None:
        datatype = 'trainx'
        query_datasample = '''
            select distinct username, filepath
            from csv
            where filetype=? and datasetname=? 
        '''
        cursor = conn.execute(query_datasample, (datatype, datasetSelect.value))
        (dataset_username, trainx_filepath) = cursor.fetchone()
        
        source_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, dataset_username, datatype, trainx_filepath), header=None)
        source_df.columns = ["attr{0:d}".format(Ci) for Ci in source_df.columns]
        print(os.path.join(UPLOAD_FOLDER, dataset_username, datatype, trainx_filepath))
        print(source_df)
        data_table.columns = [TableColumn(field=Ci, title=Ci, width=100) for Ci in source_df.columns]
        data_table.source.data = source_df[:10].to_dict(orient='list')
    #conn.close()
    print datasetSelect.options
    
update_dataset_lists(None)

def show_data_curves(event):
    # update source data directly
    # TODO: TBD
    print('show data curves')
    plots_dict = { 'price': source}
    figures_dict = { 'price': plot}
    dataset_types = [ 'price' ]
    conn = sqlite3.connect(DATABASE)
    print(datasetSelect.value)
    dataset_filepath={}
    for datatype in dataset_types:
            #need to change log reg to logreg, remove spacing in the name to find table
        try:
            source_df = pd.read_sql(
                    "select error_approx, price from price_error_curve_{:s}_{:s}_{:s}".format(
                            datasetSelect.value, select.value.replace(" ", ""), kernelSelect.value) ,
                    conn)
            plots_dict[datatype].data=source_df[['error_approx', 'price']].to_dict(orient='list')
#            figures_dict[datatype].x_range = Range1d(source_df['error_approx'].min(), source_df['error_approx'].max())
#            figures_dict[datatype].y_range = Range1d(source_df['price'].min(), source_df['price'].max())
            
            print(source_df)
#            print(source_df['error_approx'].min(), source_df['error_approx'].max())
#            print(source_df['price'].min(), source_df['price'].max())
            
            figures_dict[datatype].x_range.start = source_df['error_approx'].min() - 0.1 * (source_df['error_approx'].max() - source_df['error_approx'].min())
            figures_dict[datatype].x_range.end = source_df['error_approx'].max() + 0.1 * (source_df['error_approx'].max() - source_df['error_approx'].min())
            figures_dict[datatype].y_range.start = source_df[datatype].min() - 0.1 * (source_df[datatype].max() - source_df[datatype].min())
            figures_dict[datatype].y_range.end = source_df[datatype].max() + 0.1 * (source_df[datatype].max() - source_df[datatype].min())
#                print(plots_dict[datatype].data)
        except Exception as e:
            print(e)
            return
    #conn.close()
            
# Button definition and actions
def show_dataset(event):
#    source_table.data = (data_to_fill)
#    data_table.source.data = data_to_fill
    
    #update data
    update_dataset_lists(event)
    show_data_curves(event)
    
showDataBtn = Button(label='Show the Sample DataSet', width=360, height=50, button_type="primary")
showDataBtn.on_event(ButtonClick, show_dataset)

# Add table
table_title = Div(text='The dataset is listed here: ')
table_output = widgetbox(table_title, data_table)

#Add all things in a row
param_inst_txt = Div(text='Specific all parameters here along with the dataset.'\
                     'Click on \'show dataset\' will reveal a subset of the sample data. ', 
                     height = 50)
inputs = widgetbox(select, kernelSelect, datasetSelect, width=400)

data_table_top_right = widgetbox(data_table, showDataBtn, width=400)


# Add illustration on Optimization: 
param_txt = Div(text = 'Specify Learning Task', style={'font-size': '160%'}, 
                width=800, height=5)


### 
# OPTIMIZATION
###
opti_title = Div(text = 'Select Model Instance', style={'font-size': '160%'}, width=400, height=20)
#pe_curve_title = Div(text = 'Price - Error Curve', style={'font-size': '160%'}, width=400, height=5)

opti_txt = Div(text = 'Minimizing Error Subjuct To: ', width=800, height=10)
# Calc on budget
budget_txt = TextInput(value = '0', title = 'Budget <=: ', width=385)

budget_button = Button(label='Target on Budget', width=350, button_type="primary")
#budget_button.on_click(calc_budget)
#opti_inputs = widgetbox(opti_txt, width=300)

# Calc on Accuracy
accu_txt = TextInput(value = '0', title = 'Error >=: ', width=385)
accu_button = Button(label='Target on Error', width=350, button_type="primary")

# Select Point
opti_constraints_div = Div(text = 'Or pick a point in Figure to get value. ', width=400, height=10)

picked_model = {'model': None, 'payed': False}
# Budget or Error, search
def search_model_by_budget(event):
#    global picked_model
    budget_value = float(budget_txt.value)
    conn = sqlite3.connect(DATABASE)
    try:
        get_model_query = '''
        select price.error, price.error_approx, price.variance_approx, price.price, model.modelpath 
        from price_error_curve_{:s}_{:s}_{:s} as price, noisymodel as model
        where price.datasetname = model.datasetname and 
        price.modeltype = model.modeltype and
        price.kerneltype = model.kerneltype and
        price.variance_approx = model.noise and
        price.price <= {:f}
        order by price.price DESC
        LIMIT 1
                        '''.format(
                        datasetSelect.value, select.value.replace(" ", ""), kernelSelect.value, budget_value)
                        
        noise_error_df = pd.read_sql(get_model_query, conn)
        
        picked_model['model'] = pickle.load( open( os.path.join(UPLOAD_FOLDER, 'noisymodel', noise_error_df['modelpath'][0]) , 'rb')  )
        picked_model['payed'] = False
        picked_model['detaildf'] = noise_error_df
#        print(noise_error_df['error'].min(), noise_error_df['error'].max())
#        print(noise_error_df['price'].min(), noise_error_df['price'].max())
        print( noise_error_df[['error_approx', 'price']])
        opti_constraints_div.text = "Model Picked. Price: {:f}. Error: {:f}".format(noise_error_df['price'][0], noise_error_df['error_approx'][0])
#                print(plots_dict[datatype].data)
    except Exception as e:
        print(e)
        return
    #conn.close()

budget_button.on_event(ButtonClick, search_model_by_budget)

def search_model_by_err(event):
#    global picked_model
    error_value = float(accu_txt.value)
    conn = sqlite3.connect(DATABASE)
    try:
        get_model_query = '''
        select price.error, price.error_approx, price.variance_approx, price.price, model.modelpath 
        from price_error_curve_{:s}_{:s}_{:s} as price, noisymodel as model
        where price.datasetname = model.datasetname and 
        price.modeltype = model.modeltype and
        price.kerneltype = model.kerneltype and
        price.variance_approx = model.noise and
        price.error_approx >= {:f}
        order by price.error_approx ASC
        LIMIT 1
                        '''.format(
                        datasetSelect.value, select.value.replace(" ", ""), kernelSelect.value, error_value)
                        
        noise_error_df = pd.read_sql(get_model_query, conn)
        if( len(noise_error_df) > 0):
            picked_model['model'] = pickle.load( open( os.path.join(UPLOAD_FOLDER, 'noisymodel', noise_error_df['modelpath'][0]) , 'rb')  )
            picked_model['payed'] = False
            picked_model['detaildf'] = noise_error_df
#        print(noise_error_df['error'].min(), noise_error_df['error'].max())
#        print(noise_error_df['price'].min(), noise_error_df['price'].max())
            print( noise_error_df[['error_approx', 'price']])
            opti_constraints_div.text = "Model Picked. Price: {:f}. Error: {:f}".format(noise_error_df['price'][0], noise_error_df['error_approx'][0])
#                print(plots_dict[datatype].data)
    except Exception as e:
        print(e)
        return
    #conn.close()
    
    
accu_button.on_event(ButtonClick, search_model_by_err)


#point_x_txt = Div(text = 'Point Error: 0    Point Accuracy: 0', width=385)
#point_y_txt = Div(text = 'Select Point Accuracy = 0', width=385)
#point_x_txt = TextInput(value = '0', title = 'Point Error: ', width=385)
#point_y_txt = TextInput(value = '0', title = 'Point Price: ', width=385)

#opti_widget = widgetbox(opti_txt, budget_txt, budget_button, accu_txt, accu_button, 
#                        opti_constraints_div, point_x_txt, point_y_txt, width=400)
#opti_widget = widgetbox(opti_txt, budget_txt, budget_button, accu_txt, accu_button, 
#                        opti_constraints_div, point_x_txt, point_y_txt, width=400)
opti_widget = widgetbox(opti_txt, budget_txt, budget_button, accu_txt, accu_button, 
                        opti_constraints_div, width=400)
#opti_widget = widgetbox(trial_row1, budget_button, accu_txt, accu_button, 
#                        opti_constraints_div, point_x_txt, point_y_txt, width=400)

# Retrieve Hover
hover = plot.select(dict(type=HoverTool))
#hover = bokeh.models.HoverTool(callback = hovercallback)
hover.mode = 'vline'
hover.tooltips= [("index", "$index"),
#    ("(x,y)", "($x, $y)"),
    ("(error, price)", "(@error_approx, @price)")]

#point_x = point_x_txt, point_y = point_y_txt
TapCallback = CustomJS(args={'point_x': accu_txt, 'point_y': budget_txt}, code="""
        console.log(cb_obj);
        //console.log(cb_data);
        //text_xpos.value = cb_obj['x'];
        point_x.value = cb_obj.x.toString()
        point_y.value = cb_obj.y.toString()
        
    """)
#tap = plot.select(dict(type=TapTool))
#tap.callback = TapCallback
plot.js_on_event(events.Tap, TapCallback)
#plot.add_tools(bokeh.models.TapTool(callback=TapCallback))


### 
# BUY OUR MODEL
###
buy_title = Div(text = 'Purchase Selected Model', style={'font-size': '160%'}, width=800, height=5)

urname_txt = TextInput(value = 'Leshang Chen', title = 'Your Name: ', width=385)
cc_txt = TextInput(value = 'XXXX-XXXX-XXXX-XXXX', title = 'Credit Card #: ', width=385)
addr_txt = TextInput(value = '1210 W. Dayton Street', title = 'Billing Address: ', width=385)
pay_button = Button(label='Pay and Get the Model', width=350, button_type="primary")




modelhere_div = Div(text='Your Model Here: ', width=385)
#download_button = Button(label='Download the Model', width=360, button_type="primary")


model_widget = widgetbox(urname_txt, cc_txt, addr_txt, pay_button, width=400)

# right side: model table
model_param_to_fill = {
        "attr1": np.random.randn(20),
        "attr2": np.random.randn(20),
        "attr3": np.random.randn(20),
    }
model_source_table = ColumnDataSource(data=dict(attr1=[], attr2=[], attr3=[]))


model_table_columns = [
        TableColumn(field="attr1", title="Parameter 1"),
        TableColumn(field="attr2", title="Parameter 2"),
        TableColumn(field="attr3", title="Parameter 3")
    ]
model_table = DataTable(source=model_source_table, columns=model_table_columns, editable = True, fit_columns=False, width=360, height=148)


def pay_and_get_model(event):
    if(picked_model['model'] != None and not picked_model['payed']):
        buyer_name = urname_txt.value
        buyer_cc = cc_txt.value
        buyer_addr = addr_txt.value
        conn = sqlite3.connect(DATABASE)
        print('connect DB')
        var_approx = picked_model['detaildf']['variance_approx'][0]
        err_approx = picked_model['detaildf']['error_approx'][0]
        price = picked_model['detaildf']['price'][0]
        
        print('insert into DB')
        insertquery = '''
            insert into buyerstat 
            (buyername, buyercc, buyeraddr, datasetname, modeltype, kerneltype, error_approx, variance_approx, price) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        conn.execute(insertquery, (buyer_name, buyer_cc, buyer_addr, 
                                   datasetSelect.value, select.value.replace(" ", ""),  kernelSelect.value,
                                   err_approx, var_approx, price) )
        
        conn.commit()
        
        picked_model['payed'] = True
        #conn.close()
        ### PAYED
    
    if(picked_model['payed']):
        print('payed')
        param = picked_model['model'].coef_
        bias = picked_model['model'].intercept_
        print(param)
        print(bias)
        if(isinstance (bias, (int, long, float))):
            
            bias = np.array([bias])
            print(bias)
            model_params = np.array([np.concatenate((bias, param))])
#            attrshape, = model_params.shape
#            rowshape = 1
        else:
            model_params = np.c_[bias, param]
            
        rowshape, attrshape = model_params.shape
        print(rowshape, attrshape)
        print(model_params)
        columns = ['bias']
        for ci in np.arange(1, attrshape ):
            columns.append('weight{0:d}'.format(ci) ) 
        source_df = pd.DataFrame(model_params, columns = columns)
        
#        print(os.path.join(UPLOAD_FOLDER, dataset_username, datatype, trainx_filepath))
        print(source_df)
        model_table.columns = [TableColumn(field=ci, title=ci, width=100) for ci in source_df.columns]
        model_table.source.data = source_df.to_dict(orient='list')
        print(source_df.to_dict(orient='list'))
#        charge_model()
    
pay_button.on_event(ButtonClick, pay_and_get_model)


# Define the download button to download data from the table
download_button = Button(label="Download the Model", width=360, button_type="success")
download_button.callback = CustomJS(args=dict(source=model_table.source),
                           code=open(join(dirname(__file__), "download.js")).read())

model_widget_right = widgetbox(model_table, modelhere_div, 
                         download_button, width=360)

#def show_model_param(event):
##    source_table.data = (data_to_fill)
#    model_table.source.data = model_param_to_fill
#    
#pay_button.on_event(ButtonClick, show_model_param)

# Adding all widgets into the main port
curdoc().add_root(row(param_txt, width = 800))
#curdoc().add_root(row(param_txt, width = 800))
curdoc().add_root(row(inputs, data_table_top_right, width=800, sizing_mode='fixed'))
#curdoc().add_root(row(opti_title, pe_curve_title, width=800))
curdoc().add_root(row(opti_title, width=800))
curdoc().add_root(row(opti_widget, plot, width = 800,sizing_mode='fixed'))
#curdoc().add_root(row(empty_line, width=800))
curdoc().add_root(row(buy_title, width=800))
curdoc().add_root(row(model_widget, model_widget_right, width=800))
curdoc().title = "Model Based Pricing: Buyer Terminal"