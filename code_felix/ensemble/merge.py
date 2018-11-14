from code_felix.feature.read_file import *
from code_felix.feature.config import *


def get_result_list():
    result = {}
    for label in label_name_list:
        file_list = get_all_file(f'./output/sub/{label}')
        result[label] = [f'./output/sub/{label}/{file}' for file in file_list]
    return result


def get_top_n(top=1):
    result = get_result_list()
    new_result = {}
    for feaute_name, file_list in result.items():
        file_list = sorted(file_list, key=lambda val: val.split('.')[-2])
        new_result[feaute_name] = file_list[0]
    logger.debug(f'Top file list:{new_result}')
    return new_result


report = get_report()
top_file = get_top_n()

for feature_name, file in top_file.items():
    sub = pd.read_hdf(file, 'test')
    #logger.debug(sub.columns)
    report.loc[sub.index, feature_name] = round(sub[feature_name], 7)

    report

report_col = ['id','phosphorus_content','nitrogen_content','total_nutrient','water_content','particle_size']

report[report.month==5][report_col].to_csv('./output/sub.csv', index=None)