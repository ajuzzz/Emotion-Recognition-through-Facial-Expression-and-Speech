from tkinter import *

'''
def raise_frame(frame):
    frame.tkraise()

root = Tk()

f1 = Frame(root)
f2 = Frame(root)

for frame in (f1, f2):
    frame.grid(row=0, column=0, sticky='news')

Button(f1, text='Go to frame 2', command=lambda:raise_frame(f2)).pack()
Label(f1, text='FRAME 1').pack()

Label(f2, text='FRAME 2').pack()
Button(f2, text='Go to frame 3', command=lambda:raise_frame(f1)).pack()

raise_frame(f1)
root.mainloop()

'''
class App():
    def __init__(self):
        self.root = Tk()
        self.root.title("Emotion Recognition Project")
        self.root.geometry('1000x500')
        self.root.resizable(width=False, height=False)
        self.f1 = Frame(self.root)
        self.f2 = Frame(self.root)
        
        for frame in (self.f1, self.f2):
            frame.grid(row=0, column=0, sticky='news')
        #MAIN HEADING
        self.mainHeading = Label(self.f1 , text = "Emotion Recognition" , font = ('arial' , 50 , 'bold'))
        self.mainHeading.grid(row = 0 , padx = )
        
        self.mainHeading.pack()
        
        raise_frame(self.f1)
        self.root.mainloop()
        
    def raise_frame(self , frame):
        frame.tkraise()
        

App()