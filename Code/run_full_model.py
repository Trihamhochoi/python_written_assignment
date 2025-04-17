import os
from model import LinearRegressionModel

# Initialization
data_dict = {'train': os.path.join('Dataset', 'train.csv'),
             'test': os.path.join('Dataset', 'test.csv'),
             'ideal': os.path.join('Dataset', 'ideal.csv'), }
set_id = 1

# Build Model along with Loader
reg_model = LinearRegressionModel(db_name="data.db",data_dict=data_dict)

# 1. select best ideal function of specific set
best_ideal_dict = reg_model.choose_y_ideal(set_id=set_id)
print(best_ideal_dict)

# 3. Visualize plot of ideal function
# Extract X y
X_train, y_train = reg_model.extract_x_y(is_train=True, set_id=set_id)


reg_model.visual_ideal_plot(X_train, y_train, set_id=set_id)
reg_model.visual_4_ideal_func_test_set(X_train,is_save_plot=True,re_run=True)
# ideal_dict = reg_model.load_table_as_df(tb_name='four_best_ideal_tb').to_dict(orient='records')
# print(ideal_dict)
