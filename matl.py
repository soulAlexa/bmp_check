from matplotlib import pyplot as plt

def drawArrays(x, ys, lineType ='-', xL ='', yL ='', linesLabel=None, fillIt = False, sub = 0):
    if sub != 0:
        plt.subplot(sub)

    if linesLabel is None:
        for y in ys:
            plt.plot(x, y, lineType)
            if fillIt:
                plt.fill_between(x, y)
    else:
        i = 0
        for y in ys:
            plt.plot(x, y, lineType, label=linesLabel[i])
            if fillIt:
                plt.fill_between(x, y)
            i += 1
        plt.legend()

    plt.ylabel(yL)
    plt.xlabel(xL)
    plt.grid(visible=True)

def drawArraysWithoutX(ys, lineType ='-', xL ='', yL ='', linesLabel=None, sub = 0):
    if sub != 0:
        plt.subplot(sub)

    if linesLabel is None:
        for y in ys:
            plt.plot(y, lineType)

    else:
        i = 0
        for y in ys:
            plt.plot(y, lineType, label=linesLabel[i])
            i += 1
        plt.legend()

    plt.ylabel(yL)
    plt.xlabel(xL)
    plt.grid(visible=True)

def drawHist(x, y, color='blue', xL='', yL='', sub=0, xLim=-1, yLim=-1, linewidth=0.2):
    if sub != 0:
        plt.subplot(sub)

    if xLim > 0:
        plt.xlim(0, xLim)

    if yLim > 0:
        plt.ylim(0, yLim)

    plt.bar(x, y, color=color, width=1, linewidth=linewidth, edgecolor="white")

    plt.ylabel(yL)
    plt.xlabel(xL)
    plt.grid(visible=False)

def drawMatlabHist(data, bins=0, color='#000099', xL='', yL='', sub=0, xLim=-1, yLim=-1):
    if sub != 0:
        plt.subplot(sub)

    if xLim > 0:
        plt.xlim(0, xLim)

    if yLim > 0:
        plt.ylim(0, yLim)

    if bins != 0:
        plt.hist(data, bins=bins, color=color)
    else:
        plt.hist(data, color=color)

    plt.ylabel(yL)
    plt.xlabel(xL)
    plt.grid(visible=False)

def newFig(n):
    plt.figure(n)

def show():
    plt.show()

def save(filename):
    plt.savefig(filename)
