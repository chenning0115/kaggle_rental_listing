import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_json("data/train.json")
test_df = pd.read_json("data/test.json")

from sklearn.cluster import Birch
from math import *
def lonlatfeature(train_df,test_df,features_to_use):
    R = 6373.0
    location_dict = {
    'manhatten_loc' : [40.7527, -73.9943],
    'brooklyn_loc' : [45.0761,-73.9442],
    'bronx_loc' : [40.8448,-73.8648],
    'queens_loc' : [40.7282,-73.7949],
    'staten_loc' : [40.5795,-74.1502]}

    def getfeature(train_df):
        for location in location_dict.keys():
            lat1 = train_df['latitude'].apply(radians)
            lon1 = train_df['longitude'].apply(radians)
            lat2 = radians(location_dict[location][0])
            lon2 = radians(location_dict[location][1])
            dlon = lon2 - lon1
            dlat = lat2 - lat1

            def power(x):
                return x**2

            a = (dlat/2).apply(sin).apply(power) + lat1.apply(cos) * cos(lat2) * (dlon/2).apply(sin).apply(power)
            c = 2 * a.apply(sqrt).apply(sin)

            ### Add a new column called distance
            train_df['distance_' + location] = R * c
            features_to_use.append('distance_' + location)
        return train_df
    train_df = getfeature(train_df)
    test_df = getfeature(test_df)
    features_to_use = list(set(features_to_use))
    return train_df,test_df,features_to_use

def brand_feature(train_df,test_df,features_to_use):
    import math
    def cart2rho(x, y):
        rho = np.sqrt(x**2 + y**2)
        return rho


    def cart2phi(x, y):
        phi = np.arctan2(y, x)
        return phi


    def rotation_x(row, alpha):
        x = row['latitude']
        y = row['longitude']
        return x*math.cos(alpha) + y*math.sin(alpha)


    def rotation_y(row, alpha):
        x = row['latitude']
        y = row['longitude']
        return y*math.cos(alpha) - x*math.sin(alpha)


    def add_rotation(degrees, df):
        namex = "rot" + str(degrees) + "_X"
        namey = "rot" + str(degrees) + "_Y"

        df['num_' + namex] = df.apply(lambda row: rotation_x(row, math.pi/(180/degrees)), axis=1)
        df['num_' + namey] = df.apply(lambda row: rotation_y(row, math.pi/(180/degrees)), axis=1)
        features_to_use.extend(['num_' + namex,'num_' + namey])
        return df

    def operate_on_coordinates(tr_df, te_df):
        for df in [tr_df, te_df]:
            #polar coordinates system
            df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
            df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
            #rotations
            for angle in [15,30,45,60]:
                df = add_rotation(angle, df)
        features_to_use.extend(["num_rho","num_phi"])
        return tr_df, te_df

    train_df, test_df = operate_on_coordinates(train_df, test_df)
    import re

    def cap_share(x):
        return sum(1 for c in x if c.isupper())/float(len(x)+1)

    for df in [train_df, test_df]:
        # do you think that users might feel annoyed BY A DESCRIPTION THAT IS SHOUTING AT THEM?
        df['num_cap_share'] = df['description'].apply(cap_share)
        
        # how long in lines the desc is?
        df['num_nr_of_lines'] = df['description'].apply(lambda x: x.count('<br /><br />'))
       
        # is the description redacted by the website?        
        df['num_redacted'] = 0
        df['num_redacted'].ix[df['description'].str.contains('website_redacted')] = 1

        
        # can we contact someone via e-mail to ask for the details?
        df['num_email'] = 0
        df['num_email'].ix[df['description'].str.contains('@')] = 1
        
        #and... can we call them?
        
        reg = re.compile(".*?(\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}).*?", re.S)
        def try_and_find_nr(description):
            if reg.match(description) is None:
                return 0
            return 1

        df['num_phone_nr'] = df['description'].apply(try_and_find_nr)
    features_to_use.extend(['num_cap_share','num_nr_of_lines','num_redacted','num_email','num_phone_nr'])
    features_to_use = list(set(features_to_use))
    return train_df,test_df,features_to_use


def feature_cluster(train_df,test_df,features_to_use):
    train_df['sign'] = 'train'
    test_df['sign'] = 'test'
    all_data = pd.concat([train_df,test_df])
    def cluster_latlon(n_clusters, data):  
        #split the data between "around NYC" and "other locations" basically our first two clusters 
        data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
        data_e=data[~((data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9))]
        #put it in matrix form
        coords=data_c.as_matrix(columns=['latitude', "longitude"])
        
        brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)

        brc.fit(coords)
        clusters=brc.predict(coords)
        data_c["cluster_"+str(n_clusters)]=clusters
        data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
        data=pd.concat([data_c,data_e])
        return data 
    all_data = cluster_latlon(7,all_data)
    train_df = all_data[all_data['sign'] == 'train']
    test_df = all_data[all_data['sign'] == 'test']
    features_to_use.append("cluster_7")
    return train_df,test_df,features_to_use

def setdata(df_train,df_test,features_to_use):
    image_date = pd.read_csv("data/listing_image_time.csv")

    # rename columns so you can join tables later on
    image_date.columns = ["listing_id", "time_stamp"]

    # reassign the only one timestamp from April, all others from Oct/Nov
    image_date.loc[80240,"time_stamp"] = 1478129766 

    image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
    image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
    image_date["img_date_month"]            = image_date["img_date"].dt.month
    image_date["img_date_week"]             = image_date["img_date"].dt.week
    image_date["img_date_day"]              = image_date["img_date"].dt.day
    image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
    image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
    image_date["img_date_hour"]             = image_date["img_date"].dt.hour
    image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

    df_train = pd.merge(df_train, image_date, on="listing_id", how="left")
    df_test = pd.merge(df_test, image_date, on="listing_id", how="left")
    features_to_use.extend(['img_days_passed','img_date_month','img_date_week','img_date_day','img_date_dayofweek','img_date_dayofyear',
        'img_date_hour','img_date_monthBeginMidEnd'])
    return df_train,df_test,features_to_use

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=321, num_rounds=2000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.02
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        watchlist = [ (xgtrain,'train')]
        model = xgb.train(plst, xgtrain, num_rounds,watchlist,early_stopping_rounds=20)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

test_df["bathrooms"].loc[19671] = 1.5
test_df["bathrooms"].loc[22977] = 2.0
test_df["bathrooms"].loc[63719] = 2.0
test_df["bathrooms"].loc[17808] = 2.0
test_df["bathrooms"].loc[22737] = 2.0
test_df["bathrooms"].loc[837] = 2.0
test_df["bedrooms"].loc[100211] = 5.0
test_df["bedrooms"].loc[15504] = 4.0
train_df["price"] = train_df["price"].clip(upper=31000)
test_df["price"] = test_df["price"].clip(upper=31000)


train_df["logprice"] = np.log(train_df["price"])
test_df["logprice"] = np.log(test_df["price"])

train_df["price_t"] =train_df["price"]/(train_df["bedrooms"]+1)
test_df["price_t"] = test_df["price"]/(test_df["bedrooms"] +1)

train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

train_df['price_per_room'] = train_df['price']/(train_df['room_sum']+1)
test_df['price_per_room'] = test_df['price']/(test_df['room_sum']+1)

train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

# train_df["created"] = pd.to_datetime(train_df["created"])
# test_df["created"] = pd.to_datetime(test_df["created"])
# train_df["created_year"] = train_df["created"].dt.year
# test_df["created_year"] = test_df["created"].dt.year
# train_df["created_month"] = train_df["created"].dt.month
# test_df["created_month"] = test_df["created"].dt.month
# train_df["created_day"] = train_df["created"].dt.day
# test_df["created_day"] = test_df["created"].dt.day
# train_df["created_hour"] = train_df["created"].dt.hour
# test_df["created_hour"] = test_df["created"].dt.hour

train_df["pos"] = train_df.longitude.round(3).astype(str) + '_' + train_df.latitude.round(3).astype(str)
test_df["pos"] = test_df.longitude.round(3).astype(str) + '_' + test_df.latitude.round(3).astype(str)

vals = train_df['pos'].value_counts()
dvals = vals.to_dict()
train_df["density"] = train_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
test_df["density"] = test_df['pos'].apply(lambda x: dvals.get(x, vals.min()))

features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price","price_t","price_per_room", "logprice","density",
"num_photos", "num_features", "listing_id","num_description_words"]
train_df,test_df,features_to_use = setdata(train_df,test_df,features_to_use)
# train_df,test_df,features_to_use = feature_cluster(train_df,test_df,features_to_use)
train_df,test_df,features_to_use = lonlatfeature(train_df,test_df,features_to_use)
train_df,test_df,features_to_use = brand_feature(train_df,test_df,features_to_use)
index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)
d=[np.nan]*len(train_df)
e=[np.nan]*len(train_df)
f=[np.nan]*len(train_df)

for i in range(5):
    building_level={}
    price_level={}
    num_feature_level={}
    num_photo_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
        price_level[j]=[0]
        num_feature_level[j]=[0]
        num_photo_level[j]=[0]
    
    test_index=index[int((i*train_df.shape[0])/5):int(((i+1)*train_df.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    
    for j in train_index:
        temp=train_df.iloc[j]

        price_level[temp['manager_id']][0] += temp['price']
        num_feature_level[temp['manager_id']][0] += temp['num_features']
        num_photo_level[temp['manager_id']][0] += temp['num_photos']

        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
            
    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
            d[j]=price_level[temp['manager_id']][0] * 1.0
            e[j]=num_feature_level[temp['manager_id']][0] * 1.0
            f[j]=num_photo_level[temp['manager_id']][0] * 1.0
            
train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c
train_df['manager_level_price']=d
train_df['manager_level_feature']=e
train_df['manager_level_photo']=f

a=[]
b=[]
c=[]
d=[]
e=[]
f=[]

building_level={}
price_level={}
num_feature_level={}
num_photo_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]
    price_level[j]=[0]
    num_feature_level[j]=[0]
    num_photo_level[j]=[0]

for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    
    price_level[temp['manager_id']][0] += temp['price']
    num_feature_level[temp['manager_id']][0] += temp['num_features']
    num_photo_level[temp['manager_id']][0] += temp['num_photos']

    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
        d.append(np.nan)
        e.append(np.nan)
        f.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
        d.append(price_level[i][0]*1.0)
        e.append(num_feature_level[i][0]*1.0)
        f.append(num_photo_level[i][0]*1.0)
test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c
test_df['manager_level_price']=d
test_df['manager_level_feature']=e
test_df['manager_level_photo']=f

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')

#features_to_use.append('manager_level_price')
#features_to_use.append('manager_level_feature')
#features_to_use.append('manager_level_photo')

categorical = ["display_address", "manager_id", "building_id"]
for f in categorical:
        if train_df[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

import re
def clean(x):
    cleaned = re.sub(r"(?s)<.*?>", " ", x)
    # Keep only regular chars:
    cleaned = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", cleaned)
    # Remove unicode chars
    cleaned = re.sub("\\\\u(.){4}", " ", cleaned)
    # Remove extra whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    regex = re.compile('[^a-zA-Z ]')
    # For user clarity, broken it into three steps
    i = regex.sub(' ', cleaned).lower()
    i = i.split(" ")
    #i= [stemmer.stem(l) for l in i]
    i= " ".join([l.strip() for l in i if (len(l)>2) ]) # Keeping words that have length greater than 2
    return i
train_df['description'] = train_df["description"].apply(lambda x: clean(x))
test_df['description'] = test_df["description"].apply(lambda x: clean(x))


tfidf1=TfidfVectorizer(min_df=20, max_features=150, strip_accents='unicode',lowercase =True,
    analyzer='word', token_pattern=r'\w{16,}', ngram_range=(1, 2), stop_words = 'english') 
tr_sparse = tfidf1.fit_transform(train_df["description"])
te_sparse = tfidf1.transform(test_df["description"])
tfidf_columns1 = ['desc_'+i for i in tfidf1.vocabulary_.keys()]
tfidf_train = pd.DataFrame(tr_sparse.todense(),columns=tfidf_columns1)
tfidf_test = pd.DataFrame(te_sparse.todense(),columns=tfidf_columns1)
features_to_use.extend(tfidf_columns1)
train_df = pd.concat([train_df,tfidf_train],axis=1)
test_df = pd.concat([test_df,tfidf_test],axis = 1)

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])
tfidf_columns = ['features_'+i for i in tfidf.vocabulary_.keys()]
tfidf_train = pd.DataFrame(tr_sparse.todense(),columns=tfidf_columns)
tfidf_test = pd.DataFrame(te_sparse.todense(),columns=tfidf_columns)
features_to_use.extend(tfidf_columns)
train_df = pd.concat([train_df,tfidf_train],axis=1)
test_df = pd.concat([test_df,tfidf_test],axis = 1)
train_X = train_df[features_to_use]
test_X = test_df[features_to_use]


target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
print('start to train...')
# preds, model = runXGB(train_X, train_y, train_X,test_y=train_y, num_rounds=2000)
k_fold = StratifiedKFold()
train_predict = [[0.0] * 3 for i in range(train_X.shape[0])]
test_predict = [[0.0] * 3 for i in range(test_X.shape[0])]
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.02
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 321
num_rounds = 1450

def prepare_submission(model):
    xgtest = xgb.DMatrix(test_X)
    preds = model.predict(xgtest) 
    sub = pd.DataFrame(data = {'listing_id': test_X['listing_id'].ravel()})
    sub['high'] = preds[:, 0]
    sub['medium'] = preds[:, 1]
    sub['low'] = preds[:, 2]
    sub.to_csv("c_submission.csv", index = False, header = True)
    for r in range(preds.shape[0]):
        for c in range(3):
            test_predict[r][c] = preds[r][c]
            

xgtrain = xgb.DMatrix(train_X, label=train_y)
testing = xgb.DMatrix(test_X)
clf = xgb.train(param, xgtrain, num_rounds)
predictions = clf.predict(testing)
preds=predictions.reshape(test_X.shape[0], 3)
for r in range(len(preds)):
    for c in range(3):
        test_predict[r][c] = preds[r][c]
prepare_submission(clf)

for train, test in k_fold.split(train_X, y=train_y):
    x_for_train = train_X.iloc[train]
    y_for_train = train_y[train]
    x_for_test = train_X.iloc[test]
    y_for_test = train_y[test]
    xgtrain = xgb.DMatrix(x_for_train, label=y_for_train)
    xgtest = xgb.DMatrix(x_for_test, label=y_for_test)
    watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
    clf = xgb.train(param, xgtrain, num_rounds, watchlist)
    temp_predict = clf.predict(xgtest)
    for r in range(len(test)):
        for c in range(3):
            train_predict[test[r]][c] = temp_predict[r][c]
    
    #break



# Submission
train_file="c_train_stacknet_rental_test.csv"
test_file="c_test_stacknet_rental_test.csv"
stack_x_xgb = np.column_stack((train_X.values,train_predict))
stack_x_test_xgb = np.column_stack((test_X.values,test_predict))
stack_x = np.column_stack((train_y,stack_x_xgb))
stack_test_x = np.column_stack((test_X['listing_id'].ravel(),stack_x_test_xgb))


np.savetxt(train_file, stack_x, delimiter=",", fmt='%f')
np.savetxt(test_file, stack_test_x, delimiter=",", fmt='%f')
