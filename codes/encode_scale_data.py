import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
import keras

train_df = pd.read_csv('./data/train.csv')
lectures = pd.read_csv('./data/lectures.csv')
questions = pd.read_csv('./data/questions.csv')

# 准备用户看过的lectures具有的tag的表user_lec
tdf = train_df[train_df['content_type_id'] == 1]
user_id = tdf['user_id'].unique()
col = ['user_id','lec_tag']
user_lec = pd.DataFrame(columns = col)
idx = 0
for uid in user_id:
    tmptdf = tdf[tdf['user_id'] == uid]
    tlist=[]
    for i in range(tmptdf.shape[0]):
        cid = tmptdf.iloc[i]['content_id']
        if tlist == None:
            tlist = lectures[lectures['lecture_id'] == cid]['tag'].tolist()
        else:
            tlist = tlist + lectures[lectures['lecture_id'] == cid]['tag'].tolist()
    user_lec.loc[idx] = {col[0]:uid, col[1]:tlist}
    idx += 1
    
# 选取问题集，删去多余列
tdf = train_df[train_df['content_type_id'] == 0]
tdf = tdf.drop(['content_type_id','row_id'],1)

# merge question、user_lec和tdf三个表
tqdf = questions.drop(['bundle_id'],1)
tqdf.fillna(0, inplace = True)

def csnm(a,b):
    ans = 0
    for i in a:
        if i in b:
            ans += 1
    return ans

# 将标签转化为学习过的比例（考虑改进，加入遇到过且看过解释的标签）
def trans_tags(data):
    # 更名：为了方便
    data.rename(columns = {"prior_question_had_explanation": "explanation"}, inplace = True)
    data.rename(columns = {"prior_question_elapsed_time": "pe_time"}, inplace = True)
    
    # 数据清理
    data.fillna(0, inplace = True)
    data['explanation'] = data.explanation.apply(lambda x: False if x == 0 else x)
    data['explanation'] = data.explanation.apply(lambda x: False if type(x) == float else x)
    
    # 方便merge
    data.rename(columns = {"content_id": "question_id"}, inplace=True)    
    
    # 合并tags
    data = pd.merge(data, 
                   tqdf,
                   'outer',
                   sort = False)
    data = pd.merge(data,user_lec,'outer',sort = False)
    
    # 将string类型的tags转化为数组
    data['tags'] = data['tags'].astype('object')
    data['tags'] = data['tags'].apply(lambda tgs:  [int(x) for x in str(tgs).split()])
    data['lec_tag'] = data['lec_tag'].apply(lambda tgs: [-1] if type(tgs) == float else tgs )

    # 记录各标签学过的占比
    data['learned_rate'] = 0
    data['learned_rate'] = data.apply(lambda x: csnm(x['tags'],x['lec_tag'])/(len(x['tags'])+1e-5), axis = 1)
    
    # 去掉非问题行(tags行)
    data.dropna()
    return data

tdf = trans_tags(tdf)

# 训练集
cols = ['timestamp', 'task_container_id', 'pe_time', 'explanation', 'part', 'learned_rate']
train_x = tdf[cols]
train_y = tdf['answered_correctly']

def enc_scl(data):
    # 类别编码
    le = LabelEncoder()
    le = le.fit([True, False])
    data['explanation'] = le.transform(data['explanation'])

    # 数据scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    Scl = scaler.fit(data)
    data = Scl.transform(data)
    return data

train_x = enc_scl(train_x)
print(train_x)

# AUC函数来源：AI_盲（CSDN）
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N


# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[auc])

keras.initializers.he_normal(seed=None)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=[auc])
model.fit(train_x, train_y, batch_size=4000, epochs=35, verbose=1,validation_split=0.2,
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

sample_test = pd.read_csv('./data/example_test.csv')
test = sample_test.drop(['prior_group_answers_correct','prior_group_responses','group_num','row_id'],1)
test = trans_tags(test)

res = model.predict_proba(test)
print(res)
