import sys  # sys нужен для передачи argv в QApplication
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QApplication, QPushButton, QComboBox, QProgressBar)
from PyQt5.QtGui import QPixmap
import random
import pyqtgraph as pg
import numpy as np
import cmath
from PIL import Image


class ExampleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.wh = 0
        self.weights_Layer_1 = np.random.normal(0, 0.3, (self.wh, 6))
        self.weights_Layer_2 = np.random.normal(0, 0.3, (6, 10))
        self.weights_Layer_output = np.random.normal(0, 0.3, (10, 4))
        self.inputs_x = []
        self.er_n = []
        self.sa = []
        self.ra = []
        self.g1a = []
        self.g2a = []
        self.ras = []
        self.test_error = []
        self.er_n2 = []
        self.count = 0
        self.iter = 500
        self.init_ui()
        self.combo()

    def init_ui(self):

        layout2 = QHBoxLayout()
        layout3 = QHBoxLayout()
        layout4 = QVBoxLayout()
        layout5 = QVBoxLayout()
        self.btn = QPushButton("Выйти")
        self.comboBox = QComboBox()
        self.comboBox.addItems(["Бег", "Атлетика",
                                "Спортивная гимнастика", "Художественная гимнастика"])
        self.progress = QProgressBar()
        self.progress.setMaximum(self.iter * 4)
        self.label_pic = QLabel("")
        self.label_pic.setPixmap(QPixmap("Run.jpg"))
        self.btn_learn = QPushButton("Провести обучение")
        self.view = pg.PlotWidget()
        self.view.addLegend()
        self.curve2 = self.view.plot(pen='y', name="Ошибка зашумленных картинок")
        self.curve = self.view.plot(pen='r', name="Ошибка при обучении")
        self.view.showGrid(True, True)
        self.label_teor = QLabel("Теоритическое занчение:")
        self.label_pract = QLabel("Полученное значение:")
        self.ra = self.get_pixel(r'pictures/run.bmp')
        self.sa = self.get_pixel(r'pictures/sprint.bmp')
        self.g1a = self.get_pixel(r'pictures/gym1.bmp')
        self.g2a = self.get_pixel(r'pictures/gym2.bmp')
        self.ras = self.get_pixel(r'pictures/run2.bmp')

        self.comboBox.addItems(["Сломанный бег", "Сломанная атлетика", "Сломанная сп. гим", "Сломанная худ. гим"])

        layout2.addWidget(self.comboBox)
        layout2.addWidget(self.label_pic)

        layout5.addWidget(self.btn_learn)
        layout5.addWidget(self.progress)
        layout5.addLayout(layout2)
        layout5.addWidget(self.label_teor)
        layout5.addWidget(self.label_pract)

        layout3.addLayout(layout5)
        layout3.addWidget(self.view)

        layout4.addLayout(layout3)
        layout4.addWidget(self.btn)
        self.setLayout(layout4)

        self.setGeometry(100, 50, 1200, 600)
        self.show()

    def combo(self):
        self.comboBox.activated[str].connect(self.on_activated)
        self.btn.clicked.connect(self.quit)
        self.btn_learn.clicked.connect(self.check)

    def check(self):
        self.inputs_x = []
        self.broke_run = self.broke(self.ra)
        self.broke_sprint = self.broke(self.sa)
        self.broke_gym1 = self.broke(self.g1a)
        self.broke_gym2 = self.broke(self.g2a)
        self.create_pic(self.broke_run, r'pictures/br_run.jpg')
        self.create_pic(self.broke_sprint, r'pictures/br_sprint.jpg')
        self.create_pic(self.broke_gym1, r'pictures/br_gym1.jpg')
        self.create_pic(self.broke_gym2, r'pictures/br_gym2.jpg')
        # self.weights_Layer_2 = ((np.random.normal(0, 0.1) for x in range(6-1) for i in range(40-1)))
        self.weights_Layer_2 = np.random.normal(0, 0.1, (6, 40))
        self.weights_Layer_output = np.random.normal(0, 0.1, (40, 4))
        self.inputs_x.append(self.ra)
        self.inputs_x.append(self.sa)
        self.inputs_x.append(self.g1a)
        self.inputs_x.append(self.g2a)
        self.weights_Layer_1 = np.random.normal(0, 0.1, (self.wh, 6))
        self.predict(self.inputs_x)
        # self.label_predict.text("Обучение закончено")
        self.random_plot()
        self.er_n2 = []

    def random_plot(self):
        # random_array = np.array([self.activation(x) for x in range(-10, 10, 1)])
        self.curve.setData(self.er_n)
        self.curve2.setData(self.er_n2)

    def create_pic(self, data, name):
        img = Image.new('RGB', (50, 50), color=0)
        pix_tmp = data
        k = 0
        for i in range(int(self.wh ** (0.5))):
            for j in range(int(self.wh ** (0.5))):
                if pix_tmp[k] == 1:
                    img.putpixel((i, j), (255, 255, 255))
                else:
                    img.putpixel((i, j), (0, 0, 0))
                k = k + 1
        img = img.resize((250, 250), Image.ANTIALIAS)
        img.save(str(name))

    def on_activated(self, file):
        if self.comboBox.currentIndex() == 0:
            self.label_pic.setPixmap(QPixmap(r'pictures/Run.jpg'))
        if self.comboBox.currentIndex() == 1:
            self.label_pic.setPixmap(QPixmap(r'pictures/Sprint.jpg'))
        if self.comboBox.currentIndex() == 2:
            self.label_pic.setPixmap(QPixmap(r'pictures/idk.jpg'))
        if self.comboBox.currentIndex() == 3:
            self.label_pic.setPixmap(QPixmap(r'pictures/idk2.jpg'))
        if self.comboBox.currentIndex() == 4:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_run.jpg'))
        if self.comboBox.currentIndex() == 5:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_sprint.jpg'))
        if self.comboBox.currentIndex() == 6:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_gym1.jpg'))
        if self.comboBox.currentIndex() == 7:
            self.label_pic.setPixmap(QPixmap(r'pictures/br_gym2.jpg'))
        self.predict_pic()

    def predict_pic(self):
        if self.comboBox.currentIndex() == 0:
            out = self.fow(self.ra, pr=True)
            y = [1, 0, 0, 0]
        if self.comboBox.currentIndex() == 1:
            out = self.fow(self.sa, pr=True)
            y = [0, 1, 0, 0]
        if self.comboBox.currentIndex() == 2:
            out = self.fow(self.g1a, pr=True)
            y = [0, 0, 1, 0]
        if self.comboBox.currentIndex() == 3:
            out = self.fow(self.g2a, pr=True)
            y = [0, 0, 0, 1]
        if self.comboBox.currentIndex() == 4:
            out = self.fow(self.broke_run, pr=True)
            y = [1, 0, 0, 0]
        if self.comboBox.currentIndex() == 5:
            out = self.fow(self.broke_sprint, pr=True)
            y = [0, 1, 0, 0]
        if self.comboBox.currentIndex() == 6:
            out = self.fow(self.broke_gym1, pr=True)
            y = [0, 0, 1, 0]
        if self.comboBox.currentIndex() == 7:
            out = self.fow(self.broke_gym2, pr=True)
            y = [0, 0, 0, 1]
        if out.index(max(out)) == 0:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        if out.index(max(out)) == 1:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        if out.index(max(out)) == 2:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        if out.index(max(out)) == 3:
            self.label_pract.setText("Полученное значение на выходе:" + str(out))
        self.label_teor.setText("Теоритическое значение на выходе:" + str(y))

    def quit(self):
        exit()

    def activation(self, x, derive=False):
        if derive == True:
            i = 1 / (cmath.cosh(x)) ** 2
            return i.real
        return cmath.tanh(x).real

    def broke(self, data):
        temp = data.copy()
        changed = []
        chaff = 0.2
        for element in range(0, int(self.wh * chaff)):
            index = random.randint(0, self.wh - 1)
            if index not in changed:
                changed.append(index)
                if temp[index] == 1:
                    temp[index] = 0
                else:
                    temp[index] = 1
            else:
                element = element - 1
        return temp

    def get_pixel(self, pic):
        inputs_data = []
        picture = Image.open(pic)
        picture = picture.convert('RGB')
        (width, height) = picture.size
        self.wh = width * height
        for x in range(width):
            for y in range(height):
                if picture.getpixel((x, y)) == (255, 255, 255):
                    inputs_data.append(1)
                else:
                    inputs_data.append(0)
        return inputs_data

    def predict(self, X):
        func_error = []
        self.test_error = []
        s = 1
        for g in range(self.iter):
            for ele in X:
                tmp = 0
                y2 = [1, 0, 0, 0]
                if s == 1:
                    y1 = np.array([1, 0, 0, 0])
                elif s == 2:
                    y1 = np.array([0, 1, 0, 0])
                elif s == 3:
                    y1 = np.array([0, 0, 1, 0])
                elif s == 4:
                    y1 = np.array([0, 0, 0, 1])
                    s = 0

                layer_1_output, layer_2_output, output2 = self.fow(ele)
                br_er = self.fow(self.broke_run, pr=True)
                for i in range(len(br_er)):
                    tmp += (y2[i] - br_er[i]) ** 2

                self.test_error.append(tmp)
                temp = self.learn(ele, layer_1_output, layer_2_output, output2, y1)
                func_error.append(temp)
                s += 1
                self.count += 1
                self.progress.setValue(self.count)
        self.er_n = []
        tmp_error = 0
        tmp_error2 = 0
        for i in range(len(func_error)):
            tmp_error += func_error[i]
            tmp_error2 += self.test_error[i]
            if ((i + 1) % 4) == 0:
                self.er_n.append(tmp_error)
                self.er_n2.append(tmp_error2)
                tmp_error = 0
                tmp_error2 = 0

    def fow(self, inputs, pr=False):
        layer_1_input = []
        layer_2_input = []
        output = []

        # Первый слой
        """
        for i in range(self.weights_Layer_1.shape[1]):
            Layer_1_input.append(inputs[0] * self.weights_Layer_1[0][i])
            for j in range(len(inputs) - 1):
                Layer_1_input[i] += inputs[j + 1] * self.weights_Layer_1[j + 1][i]
        """
        layer_1_input = np.dot(inputs, self.weights_Layer_1)
        layer_1_output = np.array([self.activation(x) for x in layer_1_input])

        # Второй слой
        """
        for i in range(self.weights_Layer_2.shape[1]):
            Layer_2_input.append(layer_1_output[0] * self.weights_Layer_2[0][i])
            for j in range(len(layer_1_output) - 1):
                Layer_2_input[i] += layer_1_output[j + 1] * self.weights_Layer_2[j + 1][i]
        """
        layer_2_input = np.dot(layer_1_output, self.weights_Layer_2)
        layer_2_output = np.array([self.activation(x) for x in layer_2_input])

        # Третий слой
        """
        for i in range(self.weights_Layer_output.shape[1]):
            output.append(layer_2_output[0] * self.weights_Layer_output[0][i])
            for j in range(len(layer_2_output) - 1):
                output[i] += layer_2_output[j + 1] * self.weights_Layer_output[j + 1][i]
        """
        output = np.dot(layer_2_output, self.weights_Layer_output)
        output2 = np.array([self.activation(x) for x in output])
        if pr == True:
            out = [round(el, 2) for el in output2.real]
            return out
        else:
            return layer_1_output, layer_2_output, output2

    def learn(self, inputs, layer_1_output, layer_2_output, output, y):

        error = []
        delta = []
        delta_1 = []
        delta_2 = []
        learn_rate = 0.02
        ## --------------------------------  Обучение методом !"№"!; --------------------------------- ##

        ##----------------- Вычисление ошибки внешнего слоя -------------------##
        for n in range(len(output)):
            error.append(y[n] - output[n])
            delta.append(error[n] * self.activation(output[n], derive=True))

            #  ----------------- Вычисление ошибки скрытого слоя ---------------  #
        for k in range(len(layer_2_output)):
            tmp = 0
            for i in range(len(error)):
                tmp += delta[i] * self.weights_Layer_output[k][i]
            delta_1.append(tmp * self.activation(layer_2_output[k], derive=True))

        for k in range(len(layer_1_output)):
            tmp = 0
            for i in range(len(delta_1)):
                tmp += delta_1[i] * self.weights_Layer_2[k][i]
            delta_2.append(tmp * self.activation(layer_1_output[k], derive=True))

            ##------------------ Корректировка весов --------------------- ##
        t = 0
        for el in self.weights_Layer_output:
            for j in range(len(output)):
                el[j] += layer_2_output[t] * delta[j] * learn_rate
            t += 1

        k = 0
        for el in self.weights_Layer_2:
            for i in range(len(layer_2_output)):
                el[i] += layer_1_output[k] * delta_1[i] * learn_rate
            k += 1

        k = 0
        for el1 in self.weights_Layer_1:
            for o in range(len(layer_1_output)):
                el1[o] += inputs[k] * delta_2[o] * learn_rate
            k += 1

        s_er = 0
        for i in range(len(y)):
            s_er += (y[i] - output[i]) ** 2

        return s_er


def main():
    app = QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
