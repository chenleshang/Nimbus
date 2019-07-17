#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:15:28 2019

@author: leshang
"""
#%%
from __future__ import print_function
from flask import Flask, session, redirect, url_for, escape, request, render_template, flash, Response
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory
import sqlite3
from flask import g
import pandas as pd
import logging
import sys
import time
from md5 import md5
from werkzeug.datastructures import FileStorage
from calcnoise import calc_optimal_model, calc_error_noise_curve, calc_price_accu
from numpy import genfromtxt
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import threading 
import random
import numpy as np
#%%

logging.basicConfig(level=logging.DEBUG)

DATABASE = 'database.db'
DATABASE_INIT_FILE = 'schema.sql'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.debug = True
app.env='development'
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
exporting_threads = {}
#db = None

# DB
def get_db():
    try:
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE)
    except: 
#        global db
#        if db is None:
        db = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None) 
    if db is not None:
        db.close()
   
def init_db(): # 此函数要使用 数据库建模文件 初始化数据库：建立post表（Initial Schemas）。此函数在命令行中使用，仅一次，再次使用会删除数据库中已有数据。
    with app.app_context(): # 官网文档说了原因：不是在Web应用中使用，而是在Python Shell中使用时需要此语句（Connect on Demand）。
        db = get_db()
        with app.open_resource(DATABASE_INIT_FILE, mode='r') as f: # with语句用法！
            db.cursor().executescript(f.read().decode('utf8')) # 执行建模文件中的脚本
        db.commit() # 提交事务

def init_db_common(): # 此函数要使用 数据库建模文件 初始化数据库：建立post表（Initial Schemas）。此函数在命令行中使用，仅一次，再次使用会删除数据库中已有数据。
     # 官网文档说了原因：不是在Web应用中使用，而是在Python Shell中使用时需要此语句（Connect on Demand）。
    db = get_db()
    with open(DATABASE_INIT_FILE, mode='r') as f: # with语句用法！
        db.cursor().executescript(f.read().decode('utf8')) # 执行建模文件中的脚本
    db.commit() # 提交事务
        
# Upload File 
def get_all_file_entries(testall = True):
    check_files = ['demand', 'value']
    if(testall):
        check_files = ['trainx', 'trainy', 'testx', 'testy', 'demand', 'value']
    return check_files

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_files(files, testall):
    if all(files[entry] for entry in get_all_file_entries(testall)):
        return all(allowed_file(files[eachfile].filename) for eachfile in files)
    return False

#def exact_allowed_files(filename):
#    return filename in ['data.csv', 'demandcurve.csv', 'valuecurve.csv']
       
# check if all six files are uploaded, set testtype to others if only want to check demand curve and value curve.    
def upload_form_incomplete(files, testall = True):
    return any(item not in files for item in get_all_file_entries(testall))

def any_file_missing(files, testall=True):
    return any(files[checkfile].filename == '' for checkfile in get_all_file_entries(testall))
    
@app.route('/')
def index():
    if 'username' in session:
        return '''Logged in as %s. <a href='upload'>Upload</a>
        <a href='showdb'>Show</a>
        <a href='logout'>Logout</a>  
        ''' % escape(session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        You are not logged in. Log in here: 
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    '''

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        testallfiles = True
        # check if the post has dataset name
        if ('datasetname' not in request.form )| (request.form['datasetname'].strip() == ''):
            flash('no dataset name specified. ')
            return redirect(request.url)
        # check if the post request has the file part
        files = request.files
#        flash('yes')
       
        if upload_form_incomplete(files, testallfiles): # delete 'test2' when trying 6 files
            flash('No file part')
#            print('No file part')
            return redirect(request.url)
        print('yes')
        dataset_name = request.form['datasetname'].strip()
        # if user does not select file, browser also
        # submit an empty part without filename
        if any_file_missing(files, testallfiles):# delete 'test2' when trying 6 files
            flash('No selected file')
            return redirect(request.url)
        
        if allowed_files(files, testallfiles) :# delete 'test2' when trying 6 files
#            data_path = secure_filename(files['dataset'].filename)
#            seed = str(round(time.time() * 1000))
            trainx_type = 'trainx'
            trainy_type = 'trainy'
            testx_type='testx'
            testy_type='testy'
            demandcurve_type = 'demand'
            valuecurve_type = 'value'
            
            if testallfiles:
                trainx_file = files[trainx_type]
                trainy_file = files[trainy_type]
                testx_file = files[testx_type]
                testy_file = files[testy_type]
            demandcurve_file = files[demandcurve_type]
            valuecurve_file = files[valuecurve_type]
            
            if testallfiles:
                trainx_path = md5(trainx_file.read() + str(round(time.time() * 1000))).hexdigest() + '.csv'
                trainy_path = md5(trainy_file.read() + str(round(time.time() * 1000))).hexdigest() + '.csv'
                testx_path = md5(testx_file.read() + str(round(time.time() * 1000))).hexdigest() + '.csv'
                testy_path = md5(testy_file.read() + str(round(time.time() * 1000))).hexdigest() + '.csv'
            demandcurve_path = md5(demandcurve_file.read() + str(round(time.time() * 1000))).hexdigest() + '.csv'
            valuecurve_path = md5(valuecurve_file.read() + str(round(time.time() * 1000))).hexdigest() + '.csv'
#            demandcurve_path = secure_filename(files['demandcurve'].filename)
#            valuecurve_path = secure_filename(files['valuecurve'].filename)

            if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], session['username'])):
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username']))
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], trainx_type))
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], trainy_type))
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], testx_type))
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], testy_type))
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], demandcurve_type))
                os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], valuecurve_type))
#            data_content = files['data'].read()
#            demandcurve_content = files['demandcurve'].read()
#            valuecurve_content = files['valuecurve'].read()
            if testallfiles:
                trainx_file.seek(0)
                trainy_file.seek(0)
                testx_file.seek(0)
                testy_file.seek(0)
            demandcurve_file.seek(0)
            valuecurve_file.seek(0)
            
            if testallfiles:
                trainx_file.save(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], trainx_type, trainx_path))
                trainy_file.save(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], trainy_type, trainy_path))
                testx_file.save (os.path.join(app.config['UPLOAD_FOLDER'], session['username'], testx_type , testx_path ))
                testy_file.save (os.path.join(app.config['UPLOAD_FOLDER'], session['username'], testy_type , testy_path ))
            demandcurve_file.save(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], demandcurve_type, demandcurve_path))
            valuecurve_file.save(os.path.join(app.config['UPLOAD_FOLDER'], session['username'], valuecurve_type, valuecurve_path))
            
            
            
            sqlinsert = '''
            INSERT INTO csv
            (username, datasetname, filetype, filepath)
            VALUES (?, ?, ?, ?); '''
            db = get_db()
            if testallfiles:
                sqlcontents = [
                    (session['username'], dataset_name, trainx_type, trainx_path),
                    (session['username'], dataset_name, trainy_type, trainy_path),
                    (session['username'], dataset_name, testx_type , testx_path ),
                    (session['username'], dataset_name, testy_type , testy_path ),
                    (session['username'], dataset_name, demandcurve_type, demandcurve_path),
                    (session['username'], dataset_name, valuecurve_type, valuecurve_path)]
            else: 
                sqlcontents = [
                    (session['username'], dataset_name, demandcurve_type, demandcurve_path),
                    (session['username'], dataset_name, valuecurve_type, valuecurve_path)]
            db.executemany(sqlinsert, sqlcontents)
#            sqlinsert = '''
#            INSERT INTO user
#            (username, password)
#            VALUES (?, ?); '''
#            db.execute(sqlinsert, [session['username'], filename])
            db.commit()
#            time.sleep(10)
#            print([session['username'], filename])
#            print(sqlinsert)

#            return redirect(url_for('uploaded_file',
#                                    filename=filename))
            
            # Perform calculations
            if(request.form['task'] == 'Regression'):
                model_types = ['LinearRegression']
                variances = np.arange(1, 1001)/100.0
            else:
                model_types = ['LogisticRegression']
                variances = np.arange(1, 1001)/1000.0
                
            model_funcs = {'LogisticRegression': LogisticRegression,
                           'LinearRegression': LinearRegression }
            kernel_types = ['Default']
            
            trainx_file.seek(0)
            trainy_file.seek(0)
            testx_file.seek(0)
            testy_file.seek(0)
                
            # Train data and label
            trainX = genfromtxt(trainx_file, delimiter=',')
            trainY = genfromtxt(trainy_file, delimiter=',')
            # Test data and label
            TestX = genfromtxt(testx_file, delimiter=',')
            TestY = genfromtxt(testy_file, delimiter=',')
    
            demandcurve_file.seek(0)
            valuecurve_file.seek(0)
            demand_error_curve = pd.read_csv(demandcurve_file)
            value_error_curve =pd.read_csv(valuecurve_file)
            print('var: ', variances)
#            variances = np.arange(0, 101)/500.0
#            for model_type in model_types:
#                for kernel_type in kernel_types:
#                    optimal_model = calc_optimal_model(model_funcs[model_type], model_type, kernel_type, dataset_name, 
#                           trainX, trainY, TestX, TestY, db, app.config['UPLOAD_FOLDER'],
#                           filetype = 'bestmodel', username = session['username'])
#                    
#                    noise_err_curve = calc_error_noise_curve(model_funcs[model_type], model_type, kernel_type, dataset_name, #'testall',typo 
#                       trainX, trainY, TestX, TestY, 
#                       optimal_model, variances, db, app.config['UPLOAD_FOLDER'])
#                        
#                    price_error_curve = calc_price_accu(demand_error_curve, value_error_curve, noise_err_curve, 
#                                        dataset_name, model_type, kernel_type, db, app.config['UPLOAD_FOLDER'])
                    
#                    plt.figure()
#                    plt.plot(1/ price_error_curve.variance_approx, price_error_curve.price)
#                
#            db.commit()
            
            global exporting_threads
            thread_id = random.randint(0, 10000)
            exporting_threads[thread_id] = ExportingThread( model_funcs, model_types, kernel_types, dataset_name,
                 trainX, trainY, TestX, TestY, demand_error_curve, value_error_curve, variances, session['username'])
            exporting_threads[thread_id].start()
#            DATABASE.close()
                    
#            return '''
#                <title>Upload new File</title>
#                Upload Successfully.  <a href="/">Return</a>
#                '''
#            global exporting_threads
#
#            thread_id = random.randint(0, 10000)
#            exporting_threads[thread_id] = ExportingThread()
#            exporting_threads[thread_id].start()
    
            return render_template('progress.html', thread_id = thread_id) 
                
        return '''
                <title>Upload new File</title>
                Illegal Upload.  <a href="/upload">Retry</a>
                '''
        
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <a>Only CSV allowed. </a> <br/><br/>
    <form method=post enctype=multipart/form-data>
      Name of this dataset: <br/><input type=text name=datasetname> <br/><br/>
      Type of the task: <br/>  <select name="task">
          <option value="Classification">Classification</option>
          <option value="Regression">Regression</option>
        	</select> <br/> <br/>
      TrainX: <br/><input type=file name=trainx> <br/><br/>
      TrainY: <br/><input type=file name=trainy> <br/><br/>
      TestX: <br/><input type=file name=testx> <br/><br/>
      TestY: <br/><input type=file name=testy> <br/><br/>
      Demand Curve: <br/><input type=file name=demand> <br/><br/>
      Value Curve: <br/><input type=file name=value> <br/><br/>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
#    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'],session['username']), 
#                               filename)
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']), 
                               filename)
    
@app.route('/showdb')
def show_database_table():
    readsql = '''
    select *
    from csv
    '''
    df = pd.read_sql_query(readsql, get_db())
#    print(os.getcwd())
#    print(df)
#    print(df)
    return render_template('simple.html',  
                           tables=[df.to_html(classes=["table", "table-bordered", "table-striped", "table-hover"], header="true") ], 
                           titles=None)
#    return df.to_html(classes='data', header='true',table_id='table')

@app.route('/testdb')
def testdb():
    get_db()
    return "After test"  



class ExportingThread(threading.Thread):
    def __init__(self, model_funcs, model_types, kernel_types, dataset_name, 
                 trainX, trainY, TestX, TestY, demand_error_curve,value_error_curve, variances, username):
        super(ExportingThread, self).__init__()
#        self.progress = 10
        self.model_types = model_types
        self.model_funcs = model_funcs
        self.kernel_types = kernel_types
        self.dataset_name = dataset_name
        self.trainX = trainX
        self.trainY = trainY
        self.TestX = TestX
        self.TestY = TestY
        self.demand_error_curve = demand_error_curve
        self.value_error_curve = value_error_curve
        self.variances = variances
        self.username = username
        self.progress = {'progress': 10, 'stepprogress': 10}
#        self.db = db
        
    def run(self):
        # Your exporting stuff goes here ...
#        for _ in range(10):
#            time.sleep(1)
#            self.progress += 10
#            print(self.progress)
        db = get_db()
        count = 0
        for self.model_type in self.model_types:
            count += 1
            for self.kernel_type in self.kernel_types:
                
                
                optimal_model = calc_optimal_model(self.model_funcs[self.model_type], self.model_type, self.kernel_type, self.dataset_name, 
                       self.trainX, self.trainY, self.TestX, self.TestY, db, app.config['UPLOAD_FOLDER'],
                       filetype = 'bestmodel', username = 'username')
                
#                self.progress['progress'] += 10
                self.progress['progress'] = 20
                self.progress['stepprogress'] = 20                
                
                noise_err_curve = calc_error_noise_curve(self.model_funcs[self.model_type], self.model_type, self.kernel_type, 
                                                         self.dataset_name, #'testall',typo 
                   self.trainX, self.trainY, self.TestX, self.TestY, 
                   optimal_model, self.variances, db, app.config['UPLOAD_FOLDER'], self.progress)
                
#                self.progress['progress'] = 90
#                self.progress['stepprogress'] = 90
                price_error_curve = calc_price_accu(self.demand_error_curve, self.value_error_curve, noise_err_curve, 
                                    self.dataset_name, self.model_type, self.kernel_type, db, app.config['UPLOAD_FOLDER'])
            
#            self.progress['stepprogress'] = int(1.0 * count / len(self.model_types) * 100 + 10)
            
        self.progress['progress'] = 100
        self.progress['stepprogress'] = 100
                
        db.close()
                

@app.route('/start')
def start_test():
    global exporting_threads

    thread_id = random.randint(0, 10000)
    exporting_threads[thread_id] = ExportingThread()
    exporting_threads[thread_id].start()

#    return 'task id: #%s' % thread_id
    return render_template('progress.html', thread_id = thread_id)

#@app.route('/')
#def stop():
#	return '''<a href='/start'>Start</a>'''

@app.route('/progress/<int:thread_id>')
def progress(thread_id):
    global exporting_threads
    
    if(thread_id not in exporting_threads):
        def terminateth(thread_id):
            yield "data:100\n\n"
        return Response(terminateth(thread_id), mimetype= 'text/event-stream')
    
    def generate(thread_id):
        while exporting_threads[thread_id].progress['progress'] <= 100:
            print(exporting_threads[thread_id].progress)
            yield "data:" + str(exporting_threads[thread_id].progress['progress']) + "\n\n"
            time.sleep(1)
            if(exporting_threads[thread_id].progress['progress']):
                break;

    return Response(generate(thread_id), mimetype= 'text/event-stream')

if __name__ == '__main__':
	app.run(port=5000, debug=True)