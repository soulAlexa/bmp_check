
import copy
import math
from matplotlib import pyplot as plt
from matl import *
import numpy as np

def clamp(value, min1, max1):
    return max(min1, min(value, max1))

class bit():
    def __init__(self, widths = None, heidths = None, header = None, image = None, name = None):
        self.compresslvl = 1
        if name == None:
            self.header = header
            self.widths = widths
            self.heidths = heidths
            self.image = image
        else:
            self.header = None
            self.image = None
            self.widths = 0
            self.heidths = 0
            self.load(name)


    def load(self, name):
        f = open(name, mode='rb')
        header = []
        for val in f.read(54):
            header.append(val)
        res = []
        for val in f.read():
            res.append(val)
        self.header = header
        self.image = res
        self.weihgtsandhai()


    def weihgtsandhai(self):
        kk = 0
        for i in reversed(range(18, 21)):
            self.widths = (self.widths << 8) | self.header[i]
        for i in reversed(range(22, 25)):
            self.heidths = (self.heidths << 8) | self.header[i]
        # for i in reversed(range(10, 14)):
        #     kk = (kk << 8) | self.header[i]
        # print(kk)

    def rever(self):
        o = open("Dop.bmp", mode="wb")
        o.write(bytes(self.header))
        new = []
        for i in range(0, len(self.image), 3):
            for ii in reversed(range(0, 3, 1)):
                new.append(self.image[len(self.image) - 1 - i - ii])
        o.write(bytes(new))
        o.close()




    def printbmp(self, nam = None ,arr = None):
        if nam == None:
            o = open("nx.bmp", mode="wb")
        else:
            o = open(nam, mode="wb")
        o.write(bytes(self.header))

        if arr == None:
            o.write(bytes(self.image))
        else:
            o.write(bytes(arr))
        o.close()

    def splitrgb(self):
        red = []
        blue = []
        green = []
        for i in range(len(self.image)):
            if i % 3 == 0:
                blue.append(self.image[i])
                red.append(0)
                green.append(0)
            elif i % 3 == 1:
                green.append(self.image[i])
                red.append(0)
                blue.append(0)

            elif i % 3 == 2:
                red.append(self.image[i])
                blue.append(0)
                green.append(0)
        self.printbmp('blue.bmp', blue)
        self.printbmp('green.bmp', green)
        self.printbmp('red.bmp', red)

    def splitYCC(self):
        red = []
        blue = []
        green = []
        for i in range(len(self.image)):
            if i % 3 == 0:
                blue.append(self.image[i])
                blue.append(self.image[i])
                blue.append(self.image[i])

            elif i % 3 == 1:
                green.append(self.image[i])
                green.append(self.image[i])
                green.append(self.image[i])

            elif i % 3 == 2:
                red.append(self.image[i])
                red.append(self.image[i])
                red.append(self.image[i])
        self.printbmp('blueYCC.bmp', blue)
        self.printbmp('greenYCC.bmp', green)
        self.printbmp('redYCC.bmp', red)

    def math_expec(self, start):
        out  = 0.0
        i = start
        while i < len(self.image):
            out += self.image[i]
            i += 3
        return out/(self.widths * self.heidths)

    def dispersion(self, start):
        out = 0.0
        i = start
        expec = self.math_expec(start)
        while i < len(self.image):
            out += pow(self.image[i] - expec, 2)
            i += 3
        return out / (self.widths * self.heidths - 1)

    def std_deviation(self, start):
        return math.sqrt(self.dispersion(start))

    def correlation(self, start1, start2):
        m1, m2 = self.math_expec(start1), self.math_expec(start2)
        out = 0.0
        i1 = start1
        i2 = start2
        while i1 < len(self.image):
            out += (self.image[i1] - m1) * (self.image[i2] - m2)
            i1 += 3
            i2 += 3
        return out/(self.widths * self.heidths * self.std_deviation(start1) * self.std_deviation(start2))

    def PSNR(self, any, start):
        out = 0.0
        i = start
        while i < len(self.image):
            out += math.pow((self.image[i] - any.image[i]), 2)
            i += 3
        return math.log10((math.pow((1 << 8) - 1, 2) * self.heidths * self.widths) / out) * 10

    def YCbCr(self):

        mass = []
        i = 0
        while i < len(self.image):
            mass.append(int(clamp(0.299 * self.image[i + 2] + 0.587 * self.image[i + 1] + 0.114 * self.image[i], 0, 255)))
            mass.append(int(clamp(0.5643 * (self.image[i] - int(mass[i])) + 128, 0, 255)))
            mass.append(int(clamp(0.7132 * (self.image[i + 2] - int(mass[i])) + 128, 0, 255)))
            i += 3

        cop = bit(self.widths, self.heidths, self.header, mass)
        return cop

    def to_RGB(self):
        mass = []
        i = 0
        while i < len(self.image):
            cr, cb = self.image[i + 2] - 128, self.image[i + 1] - 128
            mass.append(int(clamp(int(self.image[i]) + 1.772 * cb, 0, 255)))
            mass.append(int(clamp(int(self.image[i]) - 0.714 * cr - 0.334 * cb, 0, 255)))
            mass.append(int(clamp(int(self.image[i]) + 1.402 * cr, 0, 255)))
            i += 3

        cop = bit(self.widths, self.heidths, self.header, mass)
        return cop

    def set_pixel(self, x, y, b = -1, g = -1, r = -1):
        i = (y + x * self.widths) * 3
        if b != -1:
            self.image[i] = b
        if g != -1:
            self.image[i + 1] = g
        if r != -1:
            self.image[i + 2] = r

    def get_pixel(self, x, y):
        i = (x * self.widths + y) * 3
        return self.image[i], self.image[i + 1], self.image[i + 2]

    def decimation_a(self, n):
        cop = bit(self.widths, self.heidths, self.header, self.image)
        n2 = int(n / 2)
        cop.compresslvl = n

        for i in range(0, self.heidths, n):
            for j in range(0, n2):
                for ii in range(n2, self.widths, n):
                    for it in range(0, n2):
                        cop.set_pixel(i + j, ii + it, g=0, r=0)
            for j in range(n2, n):
                for it in range(0, self.widths):
                    cop.set_pixel(i + j, it, g=0, r=0)

        return cop

    def decimation_b(self,  n):
        cop = bit(self.widths, self.heidths, self.header, self.image)
        n2 = n >> 1
        cop.compresslvl = n
        row = 0
        while row < self.heidths:
            col = 0
            while col < self.widths:
                for i in range(n2):
                    for ii in range(n2):
                        b_out, g_out, r_out = 0, 0, 0
                        i_1 = 0
                        while i_1 <= n2:

                            ii_1 = 0
                            while ii_1 <= n2:
                                b, g, r = self.get_pixel(row + i + i_1, col + ii + ii_1)
                                b_out += b
                                g_out += g
                                r_out += r
                                ii_1 += n2
                            i_1 += n2
                        b_out >>= 2
                        g_out >>= 2
                        r_out >>= 2
                        i_1 = 0
                        while i_1 <= n2:
                            ii_1 = 0
                            while ii_1 <= n2:
                                if not i_1 and not ii_1:
                                    g1 = int(g_out)
                                else:
                                    g1 = 0
                                if not i_1 and not ii_1:
                                    r1 = int(r_out)
                                else:
                                    r1 = 0
                                cop.set_pixel(row + i + i_1, col + ii + ii_1, g=g1, r=r1)
                                ii_1 += n2
                            i_1 += n2
                col += n
            row += n
        return cop

    def restore(self):
        n = self.compresslvl
        n2 = n >> 1
        row = 0
        while row < self.heidths:
            col = 0
            while col < self.widths:
                for i in range(n2):
                    for ii in range(n2):
                        b, g, r = self.get_pixel(row + i, col + ii)
                        i_1 = 0
                        while i_1 <= n2:
                            ii_1 = 0
                            while ii_1 <= n2:
                                self.set_pixel(row + i + i_1, col + ii + ii_1, g=int(g), r=int(r))
                                ii_1 += n2
                            i_1 += n2
                col += n
            row += n

    def get_freqs(self):
        b = [0]*256
        g = [0]*256
        r = [0]*256
        i = 0
        while i < len(self.image):
            b[self.image[i]] += 1
            g[self.image[i + 1]] += 1
            r[self.image[i + 2]] += 1
            i += 3
        return b, r, g

    def get_Bpp(self):
        b_out, g_out, r_out = 0.0, 0.0, 0.0
        b, r, g = self.get_freqs()
        wh = 1.0 * self.widths * self.heidths
        for i in range(0, 256):
            if b[i] > 0:
                b_out -= b[i] / wh * math.log2(b[i] / wh)
            if g[i] > 0:
                g_out -= g[i] / wh * math.log2(g[i] / wh)
            if r[i] > 0:
                r_out -= r[i] / wh * math.log2(r[i] / wh)
        return b_out, g_out, r_out

    def set_pixel1(self, mass, width, x, y, b=-1, g=-1, r=-1):
        i = (y + x * width) * 3
        if b != -1:
            mass[i] = b
        if g != -1:
            mass[i + 1] = g
        if r != -1:
            mass[i + 2] = r


    def get_DPCM(self):

        r1 = [0] * len(self.image)
        r2 = [0] * len(self.image)
        r3 = [0] * len(self.image)
        r4 = [0] * len(self.image)
        # row = 0
        for row in range(0, self.heidths, 1):
            # col = 0
            for col in range(0, self.widths, 1):
                b, g, r = self.get_pixel(row, col)
                if not row or not col:
                    self.set_pixel1(r1, self.widths, row, col, b, g, r)
                    self.set_pixel1(r2, self.widths, row, col, b, g, r)
                    self.set_pixel1(r3, self.widths, row, col, b, g, r)
                    self.set_pixel1(r4, self.widths, row, col, b, g, r)
                else:

                    fb_1, fg_1, fr_1 = self.get_pixel(row, col - 1)
                    fb_2, fg_2, fr_2 = self.get_pixel(row - 1, col)
                    fb_3, fg_3, fr_3 = self.get_pixel(row - 1, col - 1)
                    fb_4, fg_4, fr_4 = int((fb_1 + fb_2 + fb_3) / 3), int((fg_1 + fg_2 + fg_3) / 3), int((fr_1 + fr_2 + fr_3) / 3)

                    self.set_pixel1(r1, self.widths, row, col, b - fb_1, g - fg_1, g - fg_1)
                    self.set_pixel1(r2, self.widths, row, col, b - fb_2, g - fg_2, g - fg_2)
                    self.set_pixel1(r3, self.widths, row, col, b - fb_3, g - fg_3, g - fg_3)
                    self.set_pixel1(r4, self.widths, row, col, b - fb_4, g - fg_4, g - fg_4)

        return r1, r2, r3, r4

def dop():
    h = bit(name="kodim05.bmp")
    h.rever()

def p3():
    h = bit(name="kodim05.bmp")
    h.splitrgb()

def p4_a():
    h = bit(name="kodim05.bmp")
    print("corelarion red to green: " + str(h.correlation(2, 1)))
    print("corelarion red to blue: " + str(h.correlation(2, 0)))
    print("corelarion blue to green: " + str(h.correlation(0, 1)))


def p5():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    new.printbmp("Ycc.bmp")
    print("corelarion Cr to Cb: " + str(new.correlation(2, 1)))
    print("corelarion Cr to Y: " + str(new.correlation(2, 0)))
    print("corelarion Y to Cb: " + str(new.correlation(0, 1)))

def p6():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    new.splitYCC()

def p7():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    YCC_to_RGB = new.to_RGB()
    YCC_to_RGB.printbmp("YCC_to_RGB.bmp")
    print("PSNR(Red)   =  " + str(YCC_to_RGB.PSNR(h, 2)))
    print("PSNR(Green) =  " + str(YCC_to_RGB.PSNR(h, 1)))
    print("PSNR(Blue)  =  " + str(YCC_to_RGB.PSNR(h, 0)))

def p8_a():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    a = new.decimation_a(4)
    a.printbmp("decimation_a.bmp")


def p8_b():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    a = new.decimation_b(4)
    a.printbmp("decimation_b.bmp")

def p9_p10():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    a = new.decimation_a(4)
    a.restore()
    a.printbmp("dec_to_restore.bmp")
    new1 = a.to_RGB()
    new1.printbmp("dec_to_restore_to_RGB.bmp")
    new = h.YCbCr()
    print("For decimation_a: ")
    print("PSNR(Red)(restore vs original)   =  " + str(new1.PSNR(h, 2)))
    print("PSNR(Green)(restore vs original) =  " + str(new1.PSNR(h, 1)))
    print("PSNR(Blue)(restore vs original)  =  " + str(new1.PSNR(h, 0)))
    print("PSNR(Cr)(restore vs original)  =  " + str(a.PSNR(new, 2)))
    print("PSNR(Cb)(restore vs original)  =  " + str(a.PSNR(new, 1)))

    new = h.YCbCr()
    a = new.decimation_b(4)
    a.restore()
    a.printbmp("dec_to_restore2.bmp")
    new1 = a.to_RGB()
    new1.printbmp("dec_to_restore_to_RGB2.bmp")
    new = h.YCbCr()
    print("For decimation_b: ")
    print("PSNR(Red)(restore vs original)b   =  " + str(new1.PSNR(h, 2)))
    print("PSNR(Green)(restore vs original)b =  " + str(new1.PSNR(h, 1)))
    print("PSNR(Blue)(restore vs original)b  =  " + str(new1.PSNR(h, 0)))
    print("PSNR(Cr)(restore vs original)b  =  " + str(a.PSNR(new, 2)))
    print("PSNR(Cb)(restore vs original)b  =  " + str(a.PSNR(new, 1)))

def p11():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    a = new.decimation_a(16)
    a.printbmp("decimation_a(4x4).bmp")
    new = h.YCbCr()
    a1 = new.decimation_b(16)
    a1.printbmp("decimation_b(4x4).bmp")

    new = h.YCbCr()
    a = new.decimation_a(16)
    a.restore()
    a.printbmp("dec_to_restore3.bmp")
    new1 = a.to_RGB()
    new1.printbmp("dec_to_restore_to_RGB3.bmp")
    new = h.YCbCr()

    print("For decimation_a: ")
    print("PSNR(Red)(restore vs original)   =  " + str(new1.PSNR(h, 2)))
    print("PSNR(Green)(restore vs original) =  " + str(new1.PSNR(h, 1)))
    print("PSNR(Blue)(restore vs original)  =  " + str(new1.PSNR(h, 0)))
    print("PSNR(Cr)(restore vs original)  =  " + str(a.PSNR(new, 2)))
    print("PSNR(Cb)(restore vs original)  =  " + str(a.PSNR(new, 1)))

    new = h.YCbCr()
    a = new.decimation_b(16)
    a.restore()
    a.printbmp("dec_to_restore4.bmp")
    new1 = a.to_RGB()
    new1.printbmp("dec_to_restore_to_RGB4.bmp")
    new = h.YCbCr()
    print("For decimation_b: ")
    print("PSNR(Red)(restore vs original)b   =  " + str(new1.PSNR(h, 2)))
    print("PSNR(Green)(restore vs original)b =  " + str(new1.PSNR(h, 1)))
    print("PSNR(Blue)(restore vs original)b  =  " + str(new1.PSNR(h, 0)))
    print("PSNR(Cr)(restore vs original)b  =  " + str(a.PSNR(new, 2)))
    print("PSNR(Cb)(restore vs original)b  =  " + str(a.PSNR(new, 1)))

def p12():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    b, g, r = h.get_freqs()
    y, cb, cr = new.get_freqs()
    x = np.arange(0, 256)
    drawHist(x, r, color="red", sub=231, linewidth=0.)
    drawHist(x, g, color="green", sub=232)
    drawHist(x, b, color="blue", sub=233)
    drawHist(x, y, color="#aaaaaa", sub=234)
    drawHist(x, cb, color="#0000aa", sub=235)
    drawHist(x, cr, color="#aa0000", sub=236)
    show()

def p13():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    b, g, r = h.get_Bpp()
    y, cb, cr = new.get_Bpp()
    print("BPP(Red) =  " + str(b))
    print("BPP(Green) = " + str(g))
    print("BPP(Blue) = " + str(r))
    print("BPP(Y) = " + str(y))
    print("BPP(Cb) = " + str(cb))
    print("BPP(Cr) = " + str(cr))

def p14():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    rgb_r1, rgb_r2, rgb_r3, rgb_r4 = h.get_DPCM()
    ycc_r1, ycc_r2, ycc_r3, ycc_r4 = new.get_DPCM()

    p15(rgb_r1, ycc_r1)
    p15(rgb_r2, ycc_r2)
    p15(rgb_r3, ycc_r3)
    p15(rgb_r4, ycc_r4)


def p15(RGB,Ycc):
    b, g, r = split5(RGB)
    y, cb, cr = split5(Ycc)
    bins = 100
    drawMatlabHist(r, bins=bins, color="red", sub=231)
    drawMatlabHist(g, bins=bins, color="green", sub=232)
    drawMatlabHist(b, bins=bins, color="blue", sub=233)
    drawMatlabHist(y, bins=bins, color="#aaaaaa", sub=234)
    drawMatlabHist(cb, bins=bins, color="#0000aa", sub=235)
    drawMatlabHist(cr, bins=bins, color="#aa0000", sub=236)

    show()

def p16():
    h = bit(name="kodim05.bmp")
    new = h.YCbCr()
    rgb_r1, rgb_r2, rgb_r3, rgb_r4 = h.get_DPCM()
    ycc_r1, ycc_r2, ycc_r3, ycc_r4 = new.get_DPCM()
    wh = (h.widths * h.heidths)
    do_16(1, rgb_r1, ycc_r1, wh)
    do_16(2, rgb_r2, ycc_r2, wh)
    do_16(3, rgb_r3, ycc_r3, wh)
    do_16(4, rgb_r4, ycc_r4, wh)

def do_16(n, RGB, YCC, wh):
    b, g, r = split5(RGB)
    y, cb, cr = split5(YCC)
    print("NRule = " + str(n))
    print("BPP(Red) = " + str(get_DPCM_BPP(b, wh)))
    print("BPP(Green) = " + str(get_DPCM_BPP(g, wh)))
    print("BPP(Blue) = " + str(get_DPCM_BPP(r, wh)))
    print("BPP(Y) = " + str(get_DPCM_BPP(y, wh)))
    print("BPP(Cb) = " + str(get_DPCM_BPP(cb, wh)))
    print("BPP(Cr) = " + str(get_DPCM_BPP(cr, wh)))


def get_DPCM_BPP(inn, wh):
    y = DPMC_to_bars(inn)
    out = .0
    for it in y:
        if it > 0:
            out -= it / wh * math.log2(it/wh)
    return out

def split5(inn):
    b = [0] * int((len(inn) / 3))
    g = [0] * int(len(inn) / 3)
    r = [0] * int(len(inn) / 3)
    ii = 0
    for i in inn:
        if ii % 3 == 0:
            b[int(ii / 3)] = i
        elif ii % 3 == 1:
            g[int(ii / 3)] = i
        elif ii % 3 == 2:
            r[int(ii / 3)] = i
        ii += 1
    return b, g, r

def DPMC_to_bars(inn):
    y = [0] * 1024
    for it in inn:
        y[it + 512] += 1

    return y

if __name__ == "__main__":
    p16()
    print("\n\n\n\n\n")
    # h = bit(name="kodim05.bmp")
    # dop()
    # p13()

