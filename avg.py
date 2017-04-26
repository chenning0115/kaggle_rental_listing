import numpy as np
import pandas as pd

# pair =[('gdy5.csv',1),
# ('r.csv',1),
# ('sigma_stack_pred.csv',1),
# ('stacknet_sigle_model.csv',1),
# ('stacknet_sigle_model.csv',1),
# ('zhuangyulong_add_manager_id_groupby.csv',1),

# ]

pair =[
('charnix_best.csv',0.3),
('0.49440.csv',0.7),
]

df_list = []
for path,weight in pair:
    df_temp = pd.read_csv(path)
    df_list.append((df_temp,weight))
    
listing_id = df_list[0][0]['listing_id']
    
df_final = pd.DataFrame({'listing_id':listing_id})
df_final['high'] = 0
df_final['medium'] = 0
df_final['low'] = 0
sum_wei = 0
df_final = df_final.sort_values('listing_id')
l = 7191391
i = 0
for df,weight in df_list:
	df = df.sort_values('listing_id')
	print(df[df.listing_id==l])
	df_final['high'] = df_final['high'] + df['high'] * weight

	# print(df_final.ix[df.listing_id==7142618,'high'])

	df_final['medium'] = df_final['medium'] +  df['medium']*weight
	df_final['low'] = df_final['low'] + df['low']*weight
	sum_wei += weight
	i+=1
    

for i in ['high','medium','low']:
	df_final[i] /= sum_wei
print(df_final[df_final.listing_id==l])
df_final.to_csv('charnix_best.csv',index=False)