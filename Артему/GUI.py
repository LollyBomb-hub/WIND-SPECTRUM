import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

from main import reader, preparation

from scipy.fft import fft, rfftfreq
from scipy.signal import argrelextrema, welch

from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivy.properties import ObjectProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas, NavigationToolbar2Kivy

from kivymd.app import MDApp
from kivymd.toast import toast
from kivymd.uix.dialog import MDDialog
from kivymd.uix.tooltip import MDTooltip
from kivymd.uix.button import MDIconButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.filemanager import MDFileManager


class Content(MDBoxLayout):
    pass


class MyButton(MDGridLayout):
    sensor_name = ObjectProperty(None)


class TitleTable(MDGridLayout):
    pass


class TooltipMDIconButton(MDIconButton, MDTooltip):
    pass


class WIND_SPECTRUM(MDApp):
    dialog = None

    def __init__(self, **kwargs):
        global T, L, V, name_sensors
        super().__init__(**kwargs)
        self.restart()
        self.L = 1
        self.V = 1
        self.N = 0
        self.FD = 1
        self.sensor_name_cut = ''
        Window.bind(on_keyboard=self.events)
        self.manager_open = False
        self.file_manager = MDFileManager(
            exit_manager=self.exit_manager,
            select_path=self.select_path
        )

    def build(self):
        return Builder.load_file("KV.kv")

    def add_datatable(self):
        self.fl = MDBoxLayout(orientation="vertical",
                              pos_hint={"top": 1, "center_x": .29},
                              size_hint=(0.7, 0.7))
        self.root.ids.data.add_widget(TitleTable())
        for i in range(len(self.name_sensors)):
            btw = MyButton()
            self.root.ids.data.add_widget(btw)
            btw.ids.sensor_name.text = self.name_sensors[i]
            # selected_sensors это слоаврь где ключ имя сенсора,
            # а значения представлены массивом длиной 8
            # где значение под индексом 0 отвечает за вкл\выкл
            # отображения графика
            # где значение под индексом 1 отвечает за вкл\выкл
            # отображения линии минимума
            # где значение под индексом 2 отвечает за вкл\выкл
            # отображения линии среднего
            # где значение под индексом 3 отвечает за вкл\выкл
            # отображения линии максимума
            # где значение под индексом 4 отвечает за количество
            # отображаеммых точек максимума
            # где в значение под индексом 5 записан период данных
            # значения под индексом 6 и 7 отвечают за обрезку данных
            # число под индекос 6 начало, под индексом 7 конец
            self.selected_sensors[self.name_sensors[i]] = [0, 0, 0, 0, 1, self.identification_period(
                self.name_sensor_and_value[self.name_sensors[i]]), 0, self.N]

    # метод открывает диалоговое окно для обрези данных

    def cut_dialog(self, sensor_name):
        if not self.dialog:
            self.dialog = MDDialog(
                title=sensor_name,
                type="custom",
                content_cls=Content(),
            )
        self.sensor_name_cut = sensor_name
        self.dialog.content_cls.ids.data_start.text = str(self.selected_sensors[sensor_name][6])
        self.dialog.content_cls.ids.data_finish.text = str(self.selected_sensors[sensor_name][7])
        self.dialog.open()

    # методы принимают значения из диалогового окна

    def get_cut_start(self, value):
        if value.isnumeric():
            self.selected_sensors[self.sensor_name_cut][6] = int(value)
            self.reload_chosen_mode()
        else:
            pass

    def get_cut_finish(self, value):
        if value.isnumeric():
            self.selected_sensors[self.sensor_name_cut][7] = int(value)
            self.reload_chosen_mode()
        else:
            pass

    # Следующий метод определяет период

    def identification_period(self, x):
        arr = np.array(x)
        max_ind = argrelextrema(arr, np.greater)
        maxima = [self.time[i] for i in max_ind]
        raz = 0
        kl = len(maxima[0])
        for i in range(kl - 1):
            raz += maxima[0][i + 1] - maxima[0][i]
        return raz / kl

    # Следующий метод перезагружает график

    def reload_function(self):
        fig = plt.figure()
        for key, value in self.selected_sensors.items():
            if value[0]:
                art = plt.plot(self.time[value[6]:value[7]], self.name_sensor_and_value[key][value[6]:value[7]],
                               linewidth=2, antialiased=True, label=key)
                if value[1]:
                    minv = np.min(self.name_sensor_and_value[key][value[6]:value[7]])
                    plt.plot((np.min(self.time[value[6]:value[7]]), np.max(self.time[value[6]:value[7]])), (minv, minv),
                             linestyle='--',
                             color=art[-1].get_color())
                    plt.annotate('Min = %.3f' % minv, xycoords='data', xy=(np.mean(self.time[value[6]:value[7]]), minv),
                                 horizontalalignment='center',
                                 verticalalignment='bottom', fontsize=12, weight='bold', backgroundcolor='w')
                if value[2]:
                    avg = np.mean(self.name_sensor_and_value[key][value[6]:value[7]])
                    plt.plot((np.min(self.time[value[6]:value[7]]), np.max(self.time[value[6]:value[7]])), (avg, avg),
                             linestyle='--',
                             color=art[-1].get_color())
                    plt.annotate('Avg = %.3f' % avg, xycoords='data', xy=(np.mean(self.time[value[6]:value[7]]), avg),
                                 horizontalalignment='center', verticalalignment='bottom', fontsize=12,
                                 weight='bold', backgroundcolor='w')
                if value[3]:
                    # определяет максимальны значения с 155 по 157 аналогично в других режимах
                    arr = np.array(self.name_sensor_and_value[key][value[6]:value[7]])
                    max_ind = argrelextrema(arr, np.greater)
                    maxima = [arr[ind] for ind in max_ind]
                    maxima = sorted(maxima[0], reverse=True)[:value[4]]
                    for maxv in maxima:
                        plt.plot((np.min(self.time[value[6]:value[7]]), np.max(self.time[value[6]:value[7]])),
                                 (maxv, maxv), linestyle='--',
                                 color=art[-1].get_color())
                        plt.annotate('Max = %.3f' % maxv, xycoords='data',
                                     xy=(self.time[value[6]:value[7]][np.where(maxv == arr)[0][0]], maxv),
                                     horizontalalignment='center', verticalalignment='top', fontsize=12,
                                     weight='bold', backgroundcolor='w')
        plt.legend()
        plt.grid(True)
        plt.xlabel('Time, sec', fontsize=15)
        plt.ylabel('Аэро коэф', fontsize=15)
        canvas = fig.canvas
        plt.close()
        self.nav1 = NavigationToolbar2Kivy(canvas)
        self.root.ids.graph.clear_widgets()
        self.root.ids.graph.add_widget(canvas)
        self.root.ids.graph.add_widget(self.nav1.actionbar)

    def reload_spectrum(self):
        fig = plt.figure()
        for key, value in self.selected_sensors.items():
            if value[0]:
                yf = (1 / self.N) * (np.abs(fft(self.name_sensor_and_value[key][value[6]:value[7]])))[1:self.N // 2]
                xf = rfftfreq(self.N, 1 / self.FD)[1:self.N // 2]
                plt.plot(xf, yf, linewidth=2, antialiased=True, label=key)
                if value[1]:
                    minv = np.min(yf)
                    plt.annotate('Min = %.3f' % minv, xycoords='data', xy=(np.mean(xf), minv),
                                 horizontalalignment='center',
                                 verticalalignment='bottom', fontsize=12, weight='bold', backgroundcolor='w')
                if value[2]:
                    avg = np.mean(yf)
                    plt.annotate('Avg = %.6f' % avg, xycoords='data', xy=(np.mean(xf), avg),
                                 horizontalalignment='center', verticalalignment='bottom', fontsize=12,
                                 weight='bold', backgroundcolor='w')
                if value[3]:
                    arr = np.array(yf)
                    max_ind = argrelextrema(arr, np.greater)
                    maxima = [arr[ind] for ind in max_ind]
                    maxima = sorted(maxima[0], reverse=True)[:value[4]]
                    for maxv in maxima:
                        plt.annotate('%.3f' % xf[np.where(maxv == arr)[0][0]], xycoords='data',
                                     xy=(xf[np.where(maxv == arr)[0][0]], maxv),
                                     horizontalalignment='center', verticalalignment='top', fontsize=12,
                                     weight='bold', backgroundcolor='w')
        plt.legend()
        plt.grid()
        plt.xlabel('Frequency', fontsize=15)
        plt.ylabel('Amplitude, kH', fontsize=15)
        canvas = fig.canvas
        plt.close()
        self.nav1 = NavigationToolbar2Kivy(canvas)
        self.root.ids.graph.clear_widgets()
        self.root.ids.graph.add_widget(canvas)
        self.root.ids.graph.add_widget(self.nav1.actionbar)

    def reload_Power_Spectrum(self):
        fig = plt.figure()
        for key, value in self.selected_sensors.items():
            if value[0]:
                self.Sh = self.L / ((1 / value[5]) * self.V)
                temp, psd = welch(self.name_sensor_and_value[key][value[6]:value[7]])
                xf = np.linspace(0, self.Sh, self.N // 2)[1:len(psd) + 1]
                plt.plot(xf, psd, linewidth=2, antialiased=True, label=key)
                if value[1]:
                    minv = np.min(psd)
                    plt.annotate('Min = %.3f' % minv, xycoords='data', xy=(np.mean(xf), minv),
                                 horizontalalignment='center',
                                 verticalalignment='bottom', fontsize=12, weight='bold', backgroundcolor='w')
                if value[2]:
                    avg = np.mean(psd)
                    plt.annotate('Avg = %.6f' % avg, xycoords='data', xy=(np.mean(xf), avg),
                                 horizontalalignment='center', verticalalignment='bottom', fontsize=12,
                                 weight='bold', backgroundcolor='w')
                if value[3]:
                    arr = np.array(psd)
                    max_ind = argrelextrema(arr, np.greater)
                    maxima = []
                    for ind in max_ind:
                        maxima.append(arr[ind])
                    maxima = sorted(maxima[0], reverse=True)[:value[4]]
                    for maxv in maxima:
                        plt.annotate('%.3f' % xf[np.where(maxv == arr)[0][0]], xycoords='data',
                                     xy=(xf[np.where(maxv == arr)[0][0]], maxv),
                                     horizontalalignment='center', verticalalignment='top', fontsize=12,
                                     weight='bold', backgroundcolor='w')
        plt.legend()
        plt.grid()
        plt.xlabel('Sh', fontsize=15)
        plt.ylabel('kH^2', fontsize=15)
        canvas = fig.canvas
        plt.close()
        self.nav1 = NavigationToolbar2Kivy(canvas)
        self.root.ids.graph.clear_widgets()
        self.root.ids.graph.add_widget(canvas)
        self.root.ids.graph.add_widget(self.nav1.actionbar)

    # Следующие метод перезагружает(заново отрисовывает)
    # выбранный режим(график, спектр, спектр к числам струхаля)

    def reload_chosen_mode(self):
        if self.chosen_mode == "График":
            self.reload_function()
        elif self.chosen_mode == "АЧХ":
            self.reload_spectrum()
        else:
            self.reload_Power_Spectrum()

    # Следующие метод переключает режим
    # (график, спектр, спектр к числам струхаля)

    def change_chosen_mode(self, mode):
        self.chosen_mode = mode

    # Следующие 4 метода отвечают за чекбоксы

    def on_checkbox_sensor_name(self, checkbox, value, sensor_name):
        if value:
            self.selected_sensors[sensor_name][0] = 1
        else:
            self.selected_sensors[sensor_name][0] = 0
        self.reload_chosen_mode()

    def on_checkbox_minimum(self, checkbox, value, sensor_name):
        if value:
            self.selected_sensors[sensor_name][1] = 1
        else:
            self.selected_sensors[sensor_name][1] = 0
        self.reload_chosen_mode()

    def on_checkbox_average(self, checkbox, value, sensor_name):
        if value:
            self.selected_sensors[sensor_name][2] = 1
        else:
            self.selected_sensors[sensor_name][2] = 0
        self.reload_chosen_mode()

    def on_checkbox_maximum(self, checkbox, value, sensor_name):
        if value:
            self.selected_sensors[sensor_name][3] = 1
        else:
            self.selected_sensors[sensor_name][3] = 0
        self.reload_chosen_mode()

    # Следующие метод берет значение из поля с данными
    # (количество максимальных значений)

    def get_num_max(self, sensor_name, value):
        if value.isnumeric():
            self.num_max = int(value)
            self.selected_sensors[sensor_name][4] = self.num_max
        else:
            pass
        self.reload_chosen_mode()

    # Следующие 2 метода берут значения из 2 полей
    # (Скорость и Размер)

    def get_L(self, value):
        if value.isnumeric():
            self.L = float(value)
            self.reload_Power_Spectrum()
        else:
            pass

    def get_V(self, value):
        if value.isnumeric():
            self.V = float(value)
            self.reload_Power_Spectrum()
        else:
            pass

    # Следующие 4 метода отвечают за работу проводника
    # (взято с официальных доков)

    def file_manager_open(self):
        self.file_manager.show('/')
        self.manager_open = True

    def select_path(self, path):
        self.restart()
        self.exit_manager()
        toast(path)
        self.name_sensors = reader(os.path.abspath(path))
        self.time, self.name_sensor_and_value, self.N, self.FD = preparation()
        self.root.ids.data.clear_widgets()
        self.add_datatable()

    def exit_manager(self, *args):
        self.manager_open = False
        self.file_manager.close()

    def events(self, instance, keyboard, keycode, text, modifiers):
        if keyboard in (1001, 27):
            if self.manager_open:
                self.file_manager.back()
        return True

    # Данный метод сбрасывает все переменные

    def restart(self):
        self.selected_sensors = dict()
        self.time = None
        self.keys = set()
        self.name_sensor_and_value = None
        self.name_sensors = []
        self.data_tables = None
        self.chosen_mode = "График"


if __name__ == '__main__':
    WIND_SPECTRUM().run()
