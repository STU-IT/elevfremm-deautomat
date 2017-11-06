import os
import time
import threading
from tkinter import *

def start(parent):
    show_mainview(parent)
    parent.mainloop()
    return True

def show_mainview(parent):
    mainview_frame = Frame(parent, height=200, width=300)
    mainLabel = Label(mainview_frame, text='Main View')
    goToSubButton = Button(mainview_frame, text="go to subview")

    mainLabel.pack()
    goToSubButton.pack()

    mainview_frame.pack()

    parent.update()

    #return True

def show_subview(parent):
    return True

if __name__ == '__main__':
    root = Tk()
    start(root)