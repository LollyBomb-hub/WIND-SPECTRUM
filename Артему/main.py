import pandas as pd
import numpy as np
import csv


# Тут считываются данные(время и имена датчиков)
def reader(file_name):
    global data, name_func
    if 'xlsx' in file_name:
        data = pd.read_excel(file_name, skiprows=4)
    elif 'csv' in file_name:
        # Строки с 12 по 14 определяют разделитель файла csv
        sniffer = csv.Sniffer()
        fp = open(file_name)
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
        data = pd.read_csv(file_name, skiprows=4, sep=delimiter)
        fp.close()
    name_func = [i[15:] for i in data.columns[1:]]
    return name_func


# тут в словарь где ключ имя датчика, вписываются значения
# с 31 по 41 определяется дискретизация(кол-во отсчетов в секунду)
def preparation():
    name_sensor_and_value = dict()
    time = np.array(list(map(lambda x: float(x), data[data.columns[0]][4:])))
    N = len(time)
    for i in range(len(name_func)):
        name_sensor_and_value[name_func[i]] = list(map(lambda x: float(x),
                                                       data[data.columns[i + 1]][4:]))
    pr = [time[0], 0]
    k = []
    for i in range(len(time)):
        cor = [time[i], i]
        if str(int(cor[0]))[-1] != str(int(pr[0]))[-1]:
            k.append(cor[1] - pr[1])
            pr = cor
    max_FD = [k[0], 0]
    for i in set(k):
        if k.count(i) > max_FD[1]:
            max_FD = [i, k.count(i)]
    return time, name_sensor_and_value, N, max_FD[1]
