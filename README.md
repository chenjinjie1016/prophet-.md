# prophet-.md
```python
import pandas as pd
import numpy as np
from fbprophet import Prophet
from pandas import to_datetime
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
```


```python
df=pd.read_excel('F:\刘新建学长\wind.xlsx',usecols=[0,2])#只取数值不取时间
df.columns = ['ds','y']
df=df.dropna()
df["ds"] = to_datetime(df.ds, format="%d/%m/%Y %H:%M:%S")
df['y'] = df['y'].apply(lambda x:0 if (str(x).isspace() or str(x)=='') else x)
```


```python
trainsize = int(df.shape[0]*0.7)
train = df[0:trainsize]
test = df[trainsize:df.shape[0]]
```


```python
model = Prophet(changepoint_prior_scale=0.5)
model.fit(train);
```

    INFO:numexpr.utils:NumExpr defaulting to 4 threads.
    INFO:fbprophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    


```python
future = model.make_future_dataframe(periods=df.shape[0]-trainsize,freq="min")
print("-----------------future.tail-----------------")
print(future.tail())
```

    -----------------future.tail-----------------
                           ds
    51471 2017-03-26 20:14:00
    51472 2017-03-26 20:15:00
    51473 2017-03-26 20:16:00
    51474 2017-03-26 20:17:00
    51475 2017-03-26 20:18:00
    


```python
forecast = model.predict(future)
print("-----------------forcast tail-----------------")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
```

    -----------------forcast tail-----------------
                           ds      yhat  yhat_lower  yhat_upper
    51471 2017-03-26 20:14:00  8.485553   -4.346276   22.125345
    51472 2017-03-26 20:15:00  8.482845   -4.655776   21.567263
    51473 2017-03-26 20:16:00  8.480076   -4.656222   21.650078
    51474 2017-03-26 20:17:00  8.477246   -4.673884   21.941373
    51475 2017-03-26 20:18:00  8.474355   -3.994384   22.138684
    


```python
model.plot(forecast);
```


![png](output_6_0.png)



```python
model.plot_components(forecast);
print("-----------------forcast.columns-----------------")
print(forecast.columns)
```

    -----------------forcast.columns-----------------
    Index(['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
           'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
           'daily', 'daily_lower', 'daily_upper', 'weekly', 'weekly_lower',
           'weekly_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
           'multiplicative_terms_upper', 'yhat'],
          dtype='object')
    


![png](output_7_1.png)



```python
print("mse is",mean_squared_error(test['y'].values,forecast['yhat'].values[trainsize:df.shape[0]]))
```

    mse is 16.756599480750026
    


```python

```

