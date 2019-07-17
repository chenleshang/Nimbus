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
from bokeh.models.widgets import Slider, TextInput, DataTable, DateFormatter, TableColumn, Button, Toggle
from bokeh.plotting import figure
from bokeh.models import HoverTool, CustomJS, Div
from bokeh import events
from bokeh.models import PrintfTickFormatter, DatetimeTickFormatter, NumeralTickFormatter

from datetime import date
from random import randint
import sqlite3
import pandas as pd
import os
from bokeh.models import Range1d
from bokeh.models.tools import CrosshairTool, PanTool, TapTool, WheelZoomTool, HoverTool, ResetTool, ToolbarBox, Toolbar

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

# transform to error, not 1/ncp
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

#init data to null
x = []
y = []
#print(x, len(x))
#x = [0.1*np.exp(-0.02 * x) for x in np.arange(0.1, 100.0, 0.1)]

source = ColumnDataSource(data=dict(error_approx=x, price=y))

t_revnue = np.arange(0, 50*3600, 1*3600)/4
# TODO(leshang): can this be improved?
s_revnue = np. array([   0.        ,   56.6652806 ,   56.6652806 ,  108.60845449,
            108.60845449,  149.15589183,  205.82117244,  205.82117244,
            239.61070355,  329.33073118,  329.33073118,  329.33073118,
            329.33073118,  329.33073118,  329.33073118,  329.33073118,
            381.27390506,  381.27390506,  381.27390506,  381.27390506,
            381.27390506,  421.8213424 ,  421.8213424 ,  473.76451629,
            473.76451629,  535.15190361,  596.53929093,  648.48246482,
            648.48246482,  705.14774542,  757.09091931,  757.09091931,
            757.09091931,  757.09091931,  757.09091931,  813.75619991,
            813.75619991,  865.6993738 ,  865.6993738 ,  922.3646544 ,
            922.3646544 ,  922.3646544 ,  983.75204172, 1040.41732233,
           1040.41732233, 1040.41732233, 1040.41732233, 1080.96475967,
           1080.96475967, 1080.96475967])/100
    
#source_revenue = ColumnDataSource(data=dict(x=t_revnue, y=s_revnue))
source_revenue = ColumnDataSource(data=dict(x=[], y=[]))
#position = ColumnDataSource(data = dict (posx = posx, posy = posy))

######################################### Price vs Error Plot ##############################################################
#tools = [CrosshairTool(), PanTool(), TapTool(), WheelZoomTool(), HoverTool(), ResetTool()]
#"crosshair,pan,reset,save,wheel_zoom,hover,tap"
#fig = Figure(toolbar_location=None)
#fig.tools= tools
#toolBarBox = ToolbarBox()
#toolBarBox.toolbar = Toolbar(tools=tools)
#toolBarBox.toolbar_location = "right"


plot = figure(plot_height=400, plot_width=417, title="Price - Error Curve",
              tools="crosshair,pan,reset,save,wheel_zoom,box_zoom, hover,tap",
              x_range=[0.05, 0.15], y_range=[0, 1.2], toolbar_location='above', toolbar_sticky=False)

plot.line('error_approx', 'price', source=source, line_width=6, line_alpha=0.6)

# setting up axis name 
plot.xaxis.axis_label = 'Error'
plot.yaxis.axis_label = 'Price'

plot.xaxis.axis_label_text_font_size = "20pt"
plot.yaxis.axis_label_text_font_size = "20pt"
plot.title.text_font_size = "20pt"

plot.yaxis.major_label_text_font_size = "15pt"
plot.xaxis.major_label_text_font_size = "15pt"

#plot.add_layout(toolBarBox)
###########################################################################################################################

######################################### Online Revenue Curve ############################################################
plot_online_revenue = figure(plot_height=400, plot_width=417, title="Online Revenue Curve",
              tools="crosshair, pan, reset, save, wheel_zoom, box_zoom, hover, tap",
              x_range=[0.00, 51*3600/4], y_range=[0, 50], toolbar_location='above', toolbar_sticky=False)
#plot_online_revenue = figure(plot_height=400, plot_width=417, title="Online Revenue Curve",
#              tools="crosshair, pan, reset, save, wheel_zoom, box_zoom, hover, tap",
#              x_range=[0.05, 51*3600/4], y_range=[0, 12], toolbar_location='above', toolbar_sticky=False)

plot_online_revenue.line('x', 'y', source=source_revenue, line_width=6, line_alpha=0.6)

# setting up axis name 
plot_online_revenue.xaxis.axis_label = 'Time'
plot_online_revenue.yaxis.axis_label = 'Revenue'

plot_online_revenue.xaxis.axis_label_text_font_size = "20pt"
plot_online_revenue.yaxis.axis_label_text_font_size = "20pt"
plot_online_revenue.title.text_font_size = "20pt"

plot_online_revenue.yaxis.major_label_text_font_size = "15pt"
plot_online_revenue.xaxis.major_label_text_font_size = "11pt"

plot_online_revenue.legend.location = "bottom_right"

#plot_online_revenue.xaxis[0].formatter = NumeralTickFormatter(format='00:00:00')
plot_online_revenue.xaxis[0].formatter = NumeralTickFormatter(format='00:00')
###########################################################################################################################

# Set up widgets
text = TextInput(title="title", value='my sine wave')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)

######################################### Left Top of the Interface #####################################################
param_txt = Div(text = 'Specify Learning Task', style={'font-size': '160%'}, 
                width=800, height=14)

section_titles_left = Div(text = 'Estimate Buyer Info', style={'font-size': '160%'}, 
                width=440, height=14)
section_titles_right = Div(text = 'Monitor the Market', style={'font-size': '160%'}, 
                width=400, height=14)

modelList = ['Logistic Regression', 'Linear Regression']#, 'SVM', 'Neural Networks']
kernelList = ['Default'] #, 'Linear', 'Polynomial', 'Gaussian'
loss_funcs = {'Logistic Regression': ['0-1 Loss'], 
                'Linear Regression': ['Squared Loss']}
datasetList = ['Synthetic']#['Digit Recognition', 'Boston House-prices', 'Iris']
#loss_funcs = ['ReLU', 'LogisticLoss', 'Sigmoid', 'HingeLoss']

select_models = Select(title="Select ML model: ", value="Linear Regression", options=modelList, height=44, width=420)
select_loss_funcs = Select(title="Loss Function: ", value="Squared Loss", options=loss_funcs['Linear Regression'], height=44, width=420)
kernel_select = Select(title="Select kernel type: ", value="Default", options=kernelList, height=44, width=420)
dataset_select = Select(title="Choose Dataset: ", value="Synthetic", options=datasetList, height=44, width=420)


# Update select options
def update_model_select_options(attrname, old, new):
    model_value = select_models.value
    print(model_value)
    select_loss_funcs.options = loss_funcs[model_value]
    print(select_loss_funcs.options)
#    if(model_value == 'Logistic Regression'):
    if(select_loss_funcs.value not in select_loss_funcs.options):
        select_loss_funcs.value = select_loss_funcs.options[0]
        
select_models.on_change('value', update_model_select_options)

# Update dataset lists
def update_dataset_lists(event):
    conn = sqlite3.connect(DATABASE)
    query_dataset = '''
        select distinct datasetname
        from csv
        '''
    cursor = conn.execute(query_dataset)
#    print cursor.fetchall()
    
    dataset_select.options = [item[0] for item in cursor.fetchall()]
    if not dataset_select.value in dataset_select.options:
        dataset_select.value = dataset_select.options[0]
    
    if event is not None:
        datatype = 'trainx'
        query_datasample = '''
            select distinct username, filepath
            from csv
            where filetype=? and datasetname=? 
        '''
        cursor = conn.execute(query_datasample, (datatype, dataset_select.value))
        (dataset_username, trainx_filepath) = cursor.fetchone()
        
        source_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, dataset_username, datatype, trainx_filepath), header=None)
        source_df.columns = ["attr{0:d}".format(Ci) for Ci in source_df.columns]
        print(os.path.join(UPLOAD_FOLDER, dataset_username, datatype, trainx_filepath))
        print(source_df)
        data_table.columns = [TableColumn(field=Ci, title=Ci, width=100) for Ci in source_df.columns]
        data_table.source.data = source_df[:20].to_dict(orient='list')
        print(source_df[:20].to_dict(orient='list'))
        
    #conn.close()
    print dataset_select.options
    
update_dataset_lists(None)

# Button definition and actions
def show_dataset(event):
#    source_table.data = (data_to_fill)
#    data_table.source.data = data_to_fill
    
    #update data
    update_dataset_lists(event)
    show_data_curves(event)
    
showDataBtn = Button(label='Show Dataset Example', width=410, height=50, button_type="primary")
showDataBtn.on_event(ButtonClick, show_dataset)
#dataset_select.on_event(ButtonClick, update_dataset_lists)
#Add all things in a row
#param_inst_txt = Div(text='Specific all parameters here along with the dataset.'\
#                     'Click on \'show dataset\' will reveal a subset of the sample data. ', 
#                     height = 44)
inputs = widgetbox(select_models, kernel_select, select_loss_funcs, dataset_select, width=440)
#########################################################################################################################


# Retrieve Hover
hover = plot.select(dict(type=HoverTool))
hover.mode = 'vline'

#def update_data(attrname, old, new):
#
#    # Get the current slider values
#    a = amplitude.value
#    b = offset.value
#    w = phase.value
#    k = freq.value
#
#    source.data = dict(x=x, y=y)
#
#for w in [offset, amplitude, phase, freq]:
#    w.on_change('value', update_data)

txt = Div(text='Text will be updated with current time anytime anything '\
          'happens to the ColumnDataSource')
empty_line = Div()

source_code = """
var time = new Date()

txt.text = time.getHours() + ":" + time.getMinutes() + ":" + time.getSeconds() + " x: " + cb_obj.x + " y: " + cb_obj.y;
console.log(cb_obj);
"""

tapcallback = CustomJS(args={'txt':txt},code=source_code)
plot.js_on_event(events.Tap, tapcallback)

# Set up table
data_to_fill = dict(
        att1=np.random.randn(1),
        att2=np.random.randn(1),
        att3=np.random.randn(1),
    )
source_table = ColumnDataSource(data=dict(dates=[], downloads=[]))
#source_table = ColumnDataSource(data=data_to_fill)

table_columns = [
        TableColumn(field="att1", title="Attribute1", width=100),
        TableColumn(field="att2", title="Attribute2", width=100),
        TableColumn(field="att3", title="Attribute3", width=100)
    ]
data_table = DataTable(source=source_table, columns=table_columns, editable = True, fit_columns=False,
                       width=410, height=175)

data_table_top_right = widgetbox(data_table, showDataBtn, width=380)

#data_table.source.data = data_to_fill
# Set up layouts and add to document
#inputs = widgetbox(text, offset, amplitude, phase, freq, txt, select, kernelSelect, select2, select3)

# Add table
#table_title = Div(text='The dataset is listed here: ')
#table_output = widgetbox(table_title, data_table)
#table_output = widgetbox(data_table)


### 
# OPTIMIZATION
###
price_vs_err_title = Div(text = 'Price vs Error', style={'font-size': '200%'}, width=400, height=18)
online_revenue_title = Div(text = 'Online Revenue', style={'font-size': '200%'}, width=400, height=18)

opti_txt = Div(text = 'Maximize Accuracy Subjuct To: ', width=800, height=20)
# Calc on budget
budget_txt = TextInput(value = '0', title = 'Budget <= ', width=295)
budget_button = Button(label='Target on Budget', width=290)
#budget_button.on_click(calc_budget)
#opti_inputs = widgetbox(opti_txt, width=300)

# Calc on Accuracy
accu_txt = TextInput(value = '0', title = 'Accuracy >= ', width=295)
accu_button = Button(label='Target on Accuracy', width=290)
#budget_button.on_click(calc_budget)

# Select Point
opti_constraints_div = Div(text = 'Or Pick a point in Figure: ', width=380)


demand_title = Div(text = 'Demand Curve', style={'font-size': '200%'}, width=400, height=18)
distribution_title = Div(text = 'Distribution Curve', style={'font-size': '200%'}, width=400, height=18)


############################################ D&D Part of Code ##############################################
#a_t = np.arange(20, 121, 5)
#a_t = a_t / 1200.0
a_t = [0.1*np.exp(-0.02 * idx) for idx in np.arange(1, 101, 5)]

distribution = np.array([0.000850764399164716, 0.00219982972472225, 0.00514682469021963, 0.0108958279546975, 0.0208714033131340, \
                        0.0361754227796170, 0.0567343563605320, 0.0805098840015564, 0.103376737352580, 0.120106633357756, \
                        0.126264632132041, 0.120106633357756, 0.103376737352580, 0.0805098840015564, 0.0567343563605320, \
                        0.0361754227796170, 0.0208714033131340, 0.0108958279546975, 0.00514682469021963, 0.00219982972472225, \
                        0.000850764399164716])
value = np.array([0.181360412865646,
                1.30097589089282,
                4.29449005047003,
                11.1824939607035,
                27.0316248945261,
                41.5402831330502,
                42.2334300636103,
                42.6388950883851,
                42.9265771191702,
                44.2275530100631,
                47.2210671696403,
                54.1090710798737,
                69.9582020136964,
                84.4668602522204,
                85.1600071827805,
                85.5654722075553,
                85.8531542383405,
                87.1541301292333,
                90.1476442888105,
                97.0356481990439,
                112.884779132867])/100


x_range = [0.1*np.exp(-0.02 * idx) for idx in np.arange(1, 101, 5)]

# init data to null 
distribution = []
value = []
a_t = []
x_range = []

source_dist = ColumnDataSource(data=dict(error=a_t, demand=distribution))
source_value = ColumnDataSource(data=dict(error=x_range, value=value))
#position = ColumnDataSource(data = dict (posx = posx, posy = posy))

######################################### Demand Plot #######################################################
plot_dist = figure(plot_height=400, plot_width=417, title="Demand - Error Curve",
              tools="crosshair,pan,reset,save,wheel_zoom,box_zoom,hover,tap",
              x_range=[0.05, 0.15], y_range=[0, 0.15], toolbar_location='above', toolbar_sticky=False)

plot_dist.line('error', 'demand', source=source_dist, line_width=6, line_alpha=0.6)

# setting up axis name 
plot_dist.xaxis.axis_label = 'Error'
plot_dist.yaxis.axis_label = 'Demand'

plot_dist.xaxis.axis_label_text_font_size = "20pt"
plot_dist.yaxis.axis_label_text_font_size = "20pt"
plot_dist.title.text_font_size = "20pt"

plot_dist.yaxis.major_label_text_font_size = "15pt"
plot_dist.xaxis.major_label_text_font_size = "15pt"
#############################################################################################################

######################################### Distribution Curve ################################################
plot_value = figure(plot_height=400, plot_width=417, title="Value - Error Curve",
#plot_value = figure(plot_height=250, plot_width=417,
              tools="crosshair,pan,reset,save,wheel_zoom,box_zoom,hover,tap",
              x_range=[0.05, 0.15], y_range=[0, 1.2], toolbar_location='above', toolbar_sticky=False)

plot_value.line('error', 'value', source=source_value, line_width=6, line_alpha=0.6)

# setting up axis name 
plot_value.xaxis.axis_label = 'Error'
plot_value.yaxis.axis_label = 'Value'

plot_value.xaxis.axis_label_text_font_size = "20pt"
plot_value.yaxis.axis_label_text_font_size = "20pt"
plot_value.title.text_font_size = "20pt"

plot_value.yaxis.major_label_text_font_size = "15pt"
plot_value.xaxis.major_label_text_font_size = "15pt"

plot_value.legend.location = "bottom_right"
#############################################################################################################

############################################## Bottom buttons ###############################################
upload_mark_research_input_button = Button(label='Upload Market Research Input', width=410, height=50, button_type="primary")
enter_mark_toggle = Toggle(label='Enter the Market', width=400, height=50, button_type="warning")

def show_data_curves(event):
    # update source data directly
    # TODO: TBD
    plots_dict = {'demand': source_dist, 'value':source_value, 'price': source}
    figures_dict = {'demand': plot_dist, 'value':plot_value, 'price': plot}
    dataset_types = ['demand', 'value', 'price' ]
    conn = sqlite3.connect(DATABASE)
    print(dataset_select.value)
    dataset_filepath={}
    for datatype in dataset_types:
        if datatype != 'price': # value, demand
            query_fillin = (datatype, dataset_select.value)
            query_dataset = '''
                select distinct username, filepath
                from csv
                where filetype=? and datasetname=?
                '''
            cursor = conn.execute(query_dataset, query_fillin)
            (dataset_username, dataset_filepath[datatype]) = cursor.fetchone()
        
            print(dataset_username, dataset_filepath)
            source_df = pd.read_csv(
                    os.path.join(UPLOAD_FOLDER, dataset_username, datatype, dataset_filepath[datatype]
                    ))[['error', datatype]] #only read those columns
            print(source_df)
            plots_dict[datatype].data=source_df.to_dict(orient='list')
            
            figures_dict[datatype].x_range.start = source_df['error'].min() - 0.1 * (source_df['error'].max() - source_df['error'].min())
            figures_dict[datatype].x_range.end = source_df['error'].max() + 0.1 * (source_df['error'].max() - source_df['error'].min())
            figures_dict[datatype].y_range.start = source_df[datatype].min() - 0.1 * (source_df[datatype].max() - source_df[datatype].min())
            figures_dict[datatype].y_range.end = source_df[datatype].max() + 0.1 * (source_df[datatype].max() - source_df[datatype].min())
        else: # price: 
            #need to change log reg to logreg, remove spacing in the name to find table
            try:
                source_df = pd.read_sql(
                        "select error_approx, price from price_error_curve_{:s}_{:s}_{:s}".format(
                                dataset_select.value, select_models.value.replace(" ", ""), kernel_select.value) ,
                        conn)
                plots_dict[datatype].data=source_df[['error_approx', 'price']].to_dict(orient='list')
#                figures_dict[datatype].x_range = Range1d(source_df['error'].min(), source_df['error'].max())
#                figures_dict[datatype].y_range = Range1d(source_df['price'].min(), source_df['price'].max())
                
                print(source_df['error_approx'].min(), source_df['error_approx'].max())
                print(source_df['price'].min(), source_df['price'].max())
                
                figures_dict[datatype].x_range.start = source_df['error_approx'].min() - 0.1 * (source_df['error_approx'].max() - source_df['error_approx'].min())
                figures_dict[datatype].x_range.end = source_df['error_approx'].max() + 0.1 * (source_df['error_approx'].max() - source_df['error_approx'].min())
                figures_dict[datatype].y_range.start = source_df[datatype].min() - 0.1 * (source_df[datatype].max() - source_df[datatype].min())
                figures_dict[datatype].y_range.end = source_df[datatype].max() + 0.1 * (source_df[datatype].max() - source_df[datatype].min())
#                print(plots_dict[datatype].data)
            except Exception as e:
                print(e)
                return
    
#    print cursor.fetchall()
    #conn.close()

live_count = {'count': 0, 'time': 0}
def update_livestream():
#    t_revnue =
#    s_revnue = 

    if(enter_mark_toggle.active): 
        live_count['time'] += 10
        conn = sqlite3.connect(DATABASE)
        query_revenue = '''
            select sum(price)
            from buyerstat
            where datasetname = ? and modeltype = ? and kerneltype = ?
            group by datasetname
        '''
        cursor = conn.execute(query_revenue, (dataset_select.value, (select_models.value).replace(" ", ""), kernel_select.value) )
        print((dataset_select.value, (select_models.value).replace(" ", ""), kernel_select.value) )
        (revenue,) = cursor.fetchone()
        print(revenue)
        if revenue is not None:
            newrevenue = dict(x = [live_count['time']], y = [revenue])
            source_revenue.stream(newrevenue, rollover=50)
        
        #conn.close()
        
    else: ## plot fake curve
        if live_count['count'] < 50:
            newdata = dict( x = [t_revnue[live_count['count']]],  y = [s_revnue[live_count['count']]])
            source_revenue.stream(newdata, rollover=50)
            live_count['count'] += 1
        
def refresh_revenue_curve(event):
    print('Toggle!')
    if(enter_mark_toggle.active ):
        source_revenue.data = dict(x = [], y = [])
        plot_online_revenue.x_range.start = 0
        plot_online_revenue.x_range.end = 500
    else:
        source_revenue.data = dict(x = [], y = [])
#        plot_online_revenue.x_range=Range1d(0, 51*3600/4)
        plot_online_revenue.x_range.start = 0
        plot_online_revenue.x_range.end = 51*3600/4
        
enter_mark_toggle.active = False     
enter_mark_toggle.on_click(refresh_revenue_curve)   
# Set callbacks for the market research button
upload_mark_research_input_button.on_event(ButtonClick, show_data_curves)

umri_button = widgetbox(upload_mark_research_input_button, width=440)
entmark_button = widgetbox(enter_mark_toggle, width=420)
#############################################################################################################

# Adding all widgets into the main port
curdoc().add_root(row(param_txt, width = 800))
#curdoc().add_root(row(div, width=800))
curdoc().add_root(row(inputs, data_table_top_right, width=800, sizing_mode='fixed'))
#curdoc().add_root(row(price_vs_err_title, online_revenue_title, width=800, sizing_mode='fixed'))
#curdoc().add_root(row(plot, plot_online_revenue, width = 800, sizing_mode='fixed'))
#curdoc().add_root(row(demand_title, distribution_title, width=800))
curdoc().add_root(row(section_titles_left, section_titles_right, width=800))
curdoc().add_root(row(plot_value, plot, width = 800, sizing_mode='fixed'))
curdoc().add_root(row(plot_dist, plot_online_revenue, width=800, sizing_mode='fixed'))
curdoc().add_root(row(umri_button, entmark_button, width=800, sizing_mode='fixed'))
curdoc().add_periodic_callback(update_livestream, 10000)
curdoc().title = "Model Based Pricing: Seller Terminal"