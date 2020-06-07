# импортируем библиотеки
from ludwig.api import LudwigModel
import pandas as pd
# считываем файл
df = pd.read_csv('datasets_2015.csv')
print(df)
# указываем на каких данных тренируем модель
model_definition = {
   'input_features':[
       {'name': 'Country', 'type': 'text'},
       {'name': 'Freedom', 'type': 'text'}
   ],
   'output_features':[
       {'name': 'Generosity', 'type': 'numerical'}
   ]
}
# непосредственно тренировка
model = LudwigModel(model_definition)
train_stats = model.train(df)
