import numpy as np
import pandas as pd
import graphviz
from scipy.stats import rankdata
import pickle
import scipy.stats as stats
from sklearn import metrics as me
import scipy.stats as stats
from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
pd.set_option('mode.use_inf_as_na', True)
'''
fields = ['Amount','UVolume','UVolumeFB','HighPrice','HInc','FirstHitTime','LastHitTime','BSV','BSN','PreInc1','PreInc5',
          'PreInc10','UVOP','UVOPFB','Inc','level','SHIndexPre','SHIndexND','SHIndexPre','SHIndexND','SZIndexPre','UplimitCount',
          'RealUplimitCount','FirstBidVol','RateLow','RateClose','FirstBidVol2','ABRate','FirstTickHitTime']
'''#未来因子弃用了

fields = ['UVolumeFB','HighPrice','HInc','FirstHitTime','BSV','BSN','PreInc1','PreInc5',
'PreInc10','UVOP','UVOPFB','SHIndexPre','SZIndexPre','FirstBidVol','FirstBidVol2','ABRate']

df= pd.read_csv('C:/Users/uplimitdata_all.csv')


df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(axis=0, how='any')

df['Uplimit'].shift(1)
df['Uplimit'][0]=True
t=list(df['Uplimit'])
a=[]
'''
for v in t:
    if v==True:
        a.append(1)
    else:
        a.append(0)
target=a
'''
#target.shift(1)
#target = np.append(0, target)
target =df['Inc'].values


df.index = df.Date
#df= df.iloc[:,1:]


#这里的csv是很多只股票的数据整合到一个表内，dataframe内有一列表示股票代码
#total_data = total_data.iloc[:80000,:]

stock_list = df.UCodes.values.tolist() 

length =  []
num = 1
for i in range(len(stock_list)-1):
    if stock_list[i+1] == stock_list[i]:
        if i == len(stock_list)-2:
            length.append(num)
            break
        num+=1
    else:
        length.append(num)
        num = 1
        
# ------------------------------       






   
#target=np.random.randint(0,2,len(total_data['pct10'].values))
data = df[fields]
data['3'] = 3
data['5'] = 5
data['6']=6
data['8'] = 8
data['10'] = 10
data['12'] = 12
data['15'] = 15
data['20']=20
fields = fields + ['3','5','6','8','10','12','15','20']

X_train = np.nan_to_num(data[:].values)


#X_test = data[-test_num:]
y_train = np.nan_to_num(target)
#del data
#del target

init_function = ['add', 'sub', 'mul', 'div','sqrt', 'log','inv','sin','max','min']


def _my_metric(y, y_pred, w):
    return me.normalized_mutual_info_score(y, y_pred)

my_metric = make_fitness(function=_my_metric, greater_is_better=True)

def _rolling_rank(data):
    value = rankdata(data)[-1]    
    return value
def rank(data):
    return rankdata(data)


def _rolling_prod(data):
    return np.prod(data)

def _cube(data):
    return np.square(data)*data

def _square(data):
    return np.square(data)

def _delta(data):
    value = np.diff(data.flatten())
    value = np.append(0, value)   #将0插入第一个元素，保证维度正确
    return value



def _delay(data):
    period=1
    value = pd.Series(data.flatten()).shift(1)
    value = np.nan_to_num(value)

    k=1
    data = pd.Series(data.flatten())
    value = data.mul(1).div(np.abs(data).sum())
    value = np.nan_to_num(value)
    
    return value

#归一化函数
        
def _corr(data1,data2,n):
    
    with np.errstate(divide='ignore', invalid='ignore'):

            
        try:
            if n[0] == n[1] and n[1] ==n[2]:
                window  = n[0]

                x1 = pd.Series(data1.flatten())
                x2 = pd.Series(data2.flatten())

                df = pd.concat([x1,x2],axis=1)
                temp = pd.Series()
                for i in range(len(df)):
                    if i<=window-2:
                        temp[str(i)] = np.nan
                    else:
                        df2 = df.iloc[i-window+1:i,:]
                        temp[str(i)] = df2.corr('spearman').iloc[1,0]
                return np.nan_to_num(temp)
            else:
                return np.zeros(data1.shape[0])
            
        except:
            return np.zeros(data1.shape[0])

def DFT(sig):
    N = sig.size
    V = np.array([[np.exp(-1j*2*np.pi*v*y/N) for v in range(N)] for y in range(N)])
    return sig.dot(V)
#傅里叶变换函数

def _ts_sum(data,n):

    with np.errstate(divide='ignore', invalid='ignore'):

        try:
            if n[0] == n[1] and n[1] ==n[2]:
                window  = n[0]
    
                value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())
                value = np.nan_to_num(value)
    
                return value
            else:
                return np.zeros(data.shape[0])

        except:
            return np.zeros(data.shape[0])
        


def _sma(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):

        try:    
            if n[0] == n[1] and n[1] ==n[2]:
                window  = n[0]
                
                value = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
                value = np.nan_to_num(value)
    
                return value
            else:
                return np.zeros(data.shape[0])
              
        except:
            return np.zeros(data.shape[0])

def _stddev(data,n):   
    with np.errstate(divide='ignore', invalid='ignore'):
        try:    
            if n[0] == n[1] and n[1] ==n[2]:
                window  = int(np.mean(n))
                value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])

def _ts_rank(data,n):
    
    with np.errstate(divide='ignore', invalid='ignore'):

        try:
            if n[0] == n[1] and n[1] ==n[2]:        
                value = np.array(pd.Series(data.flatten()).rolling(window).apply(_rolling_rank).tolist())
                value = np.nan_to_num(value)

                return value
            else:
                return np.zeros(data.shape[0])    
        except:
            return np.zeros(data.shape[0])


ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=2)

def _ts_argmin(data,n):

    try:
        if n[0] == n[1] and n[1] ==n[2]:
            window=n[0]
            value = pd.Series(data.flatten()).rolling(window).apply(np.argmin) + 1 
            value = np.nan_to_num(value)
            return value
        else:
            return np.zeros(data.shape[0])  
    except:
        return np.zeros(data.shape[0])


def _ts_argmax(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] ==n[2]:
                window=n[0]
                value = pd.Series(data.flatten()).rolling(window).apply(np.argmax,raw=False) + 1 
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])
        
def _ts_min(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        
        try:
            if n[0] == n[1] and n[1] ==n[2]:
                window  = n[0]
                #window  = int(np.mean(n))
                value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
                value = np.nan_to_num(value)
    
                return value
            else:
                return np.zeros(data.shape[0])
                
        except:
            return np.zeros(data.shape[0])  
        
def _ts_max(data,n):
    with np.errstate(divide='ignore', invalid='ignore'):
        
        try:
            if n[0] == n[1] and n[1] ==n[2]:
                window  = n[0]
    
                value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
                value = np.nan_to_num(value)
    
                return value
            else:
                return np.zeros(data.shape[0])
         
                
        except:
            return np.zeros(data.shape[0])  
        
def _ts_argmaxmin(data,n):
    return _ts_argmax(data,n) - _ts_argmin(data,n)
def _sigmoid(data):
    return np.array([1/(1+np.exp(-i)) for i in data])


rank = make_function(function=rank, name='rank', arity=1)
sigmoid = make_function(function=_sigmoid, name='sigmoid', arity=1)
stddev = make_function(function=_stddev, name='stddev', arity=2)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2)
stddev = make_function(function=_stddev, name='stddev', arity=2)
corr = make_function(function=_corr, name='corr', arity=3)# 

ts_min = make_function(function=_ts_min, name='ts_min', arity=2)

delta = make_function(function=_delta, name='delta', arity=1)
delay = make_function(function=_delay, name='delay', arity=1)
sma = make_function(function=_sma, name='sma', arity=2)
cube = make_function(function=_cube, name='cube', arity=1)
square = make_function(function=_square, name='square', arity=1)

ts_argmaxmin = make_function(function=_ts_argmaxmin, name='ts_argmaxmin', arity=2)

ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=2)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=2)

ts_min = make_function(function=_ts_min, name='ts_min', arity=2)

ts_max = make_function(function=_ts_max, name='ts_max', arity=2)
user_function = [sigmoid,ts_min,square,cube,delta, delay, ts_argmax ,sma,stddev, ts_argmin, ts_max,ts_min,ts_sum,ts_rank,ts_argmaxmin,corr]
function_set = init_function + user_function
metric = my_metric
population_size = 1000
generations = 20
random_state=0
est_gp = SymbolicTransformer(
                            feature_names=fields, 
                            function_set=function_set,
                            generations=generations,
                            stopping_criteria=0.9,
                            metric=metric,#'spearman'秩相关系数
                            population_size=population_size,
                            tournament_size=30, 
                            random_state=random_state,
                            verbose=2,
                            parsimony_coefficient=0.001,
                            p_crossover = 0.4,
                            p_subtree_mutation = 0.01,
                            p_hoist_mutation = 0.01,
                            p_point_mutation = 0.01,
                            p_point_replace = 0.4,
                            n_jobs = 6
                            )
###------------set 训练集的为空值---------
num_set_y_nan = 0
for i in length:
    num_set_y_nan = i + num_set_y_nan
    try:
        y_train[num_set_y_nan] = np.nan
    except:
        break

X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)  #这里是一个待改进的缺陷，用0代替了缺失元素（可以尝试用均值代替？）
est_gp.fit(X_train, y_train)

best_programs = est_gp._best_programs
best_programs_dict = {}

for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness':p.fitness_, 'expression':str(p), 'depth':p.depth_, 'length':p.length_}
     
best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')
best_programs_dict 



def alpha_factor_graph(num):
    # 打印指定num的表达式图

    factor = best_programs[num-1]
    print(factor)
    print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('images/alpha_factor_graph', format='png', cleanup=True)
    
    return graph

graph1 = alpha_factor_graph(1)
graph2 = alpha_factor_graph(2)
graph3 = alpha_factor_graph(3)
graph4 = alpha_factor_graph(4)
graph5 = alpha_factor_graph(5)



