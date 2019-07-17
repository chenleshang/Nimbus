DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS csv;
DROP TABLE IF EXISTS bestmodel;
DROP TABLE IF EXISTS noisymodel;
DROP TABLE IF EXISTS pricecurve;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);


CREATE TABLE csv (
    username TEXT NOT NULL,
    datasetname TEXT NOT NULL,
    filetype TEXT NOT NULL, 
    filepath TEXT NOT NULL,
    CONSTRAINT compkey PRIMARY KEY (datasetname, filetype)
);


CREATE TABLE bestmodel (
    username TEXT,
    datasetname TEXT NOT NULL,
    modeltype TEXT NOT NULL,
    kerneltype TEXT NOT NULL,
    modelpath TEXT NOT NULL
    -- CONSTRAINT compkey PRIMARY KEY (datasetname, modeltype, kerneltype)
);

CREATE TABLE noisymodel (
    username TEXT,
    datasetname TEXT NOT NULL,
    modeltype TEXT NOT NULL,
    kerneltype TEXT NOT NULL, 
    modelpath TEXT NOT NULL,
    noise REAL NOT NULL,
    error REAL NOT NULL
    -- CONSTRAINT compkey PRIMARY KEY (datasetname, modeltype, kerneltype, noise)
);

CREATE TABLE pricecurve (
  datasetname TEXT NOT NULL,
  modeltype TEXT NOT NULL,
  kerneltype TEXT NOT NULL,
  error REAL NOT NULL,
  error_approx REAL NOT NULL,
  variance_approx REAL NOT NULL,
  price REAL NOT NULL
  -- CONSTRAINT compkey PRIMARY KEY (datasetname, modeltype, kerneltype, noise)
);

CREATE TABLE buyerstat (
  buyername TEXT NOT NULL,
  buyercc TEXT NOT NULL,
  buyeraddr TEXT NOT NULL,
  datasetname TEXT NOT NULL,
  modeltype TEXT NOT NULL,
  kerneltype TEXT NOT NULL,
  error_approx REAL NOT NULL,
  variance_approx REAL NOT NULL,
  price REAL NOT NULL
);
