from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from n_detector import detector_op


def openfile():
    global img, label1, text
    file_name = window.filename = filedialog.askopenfilename(initialdir = "d:", title ="Select A Image", filetypes=(("PNG files","*.png"),("JPEG files","*.jpeg"),("JPG files","*.jpg")))
    if (window.filename != ""):
        image = Image.open(window.filename)
        asd = detector_op(file_name)
        text = Text(text_window)
        text . insert(INSERT, asd)
        text.place(x = 385,y =115,h = 180, w = 350)
        image = image.resize((300,300))
        img = ImageTk.PhotoImage(image)
        label1 = Label(image=img)
        label1.place(x = 40, y=120)

def deletedata():
    try:
        label1.destroy()
        text.destroy()
    except:
        pass

def exitwindow():
    window.destroy()

def about():
    root = Tk()
    root.title("About Us")
    Label(root,text = "This software was created and developed by \n - Hari Om Swarup S A \n -Venilla V \n of PES University" ).pack()
    root.mainloop()

def main_window():
    global window, text_window
    window = Tk()
    window.title("Handwriting Text Detection")
    window.geometry("760x530")
    window.configure(bg="#242333")
    image = Image.open("Data\\Gallery\\opencv.png")
    image = image.resize((60,60))
    img = ImageTk.PhotoImage(image)
    label1 = Label(image=img,bg="#242333")
    label1.place(x = 110, y=20)
    Label(window, text = "Handwriting Text Detection", font =("TimesNewRoman",28,"bold"),fg = "white",bg="#242333").place(x = 180, y = 30)
    image_window = LabelFrame(window, text="Image", width=340, height=430, font="TimesNewRoman", fg="Green",bg="#242333").place(x=20, y=90)
    text_window = LabelFrame(window, text = "Text", width=360, height=210, font="TimesNewRoman", fg="Red",bg="#242333").place(x=380, y=90)
    button_window = LabelFrame(window, text="Buttons", width=360, height=210, font="TimesNeWRoman", fg="Blue",bg="#242333").place(x=380, y=310)

    Label(button_window, fg="white", bg="#242333", text="Open file : ", font=("calibri", 16, "bold")).place(x= 450, y=360)
    Label(button_window, fg="white", bg="#242333", text="Clear\t: ", font=("calibri", 16, "bold")).place(x= 450, y=390)
    Label(button_window, fg="white", bg="#242333", text="Exit\t: ", font=("calibri", 16, "bold")).place(x= 450, y=420)
    Label(button_window, fg="white", bg="#242333", text="About Us\t: ", font=("calibri", 16, "bold")).place(x= 450, y=450)
    Button(button_window,command =openfile, text = "Open Image", fg ="#ffffff", bg = "#01b0d3", borderwidth=2, font=("calibri", 10, "bold")).place(x = 580, y = 360, w = 80)
    Button(button_window,command =deletedata, text = "Clear", fg ="#ffffff", bg = "#01b0d3", borderwidth=2, font=("calibri", 10, "bold")).place(x = 580, y = 390, w = 80)
    Button(button_window,command =exitwindow, text = "Exit", fg ="#ffffff", bg = "#01b0d3", borderwidth=2, font=("calibri", 10, "bold")).place(x = 580, y = 420, w = 80)
    Button(button_window,command =about, text = "About Us", fg ="#ffffff", bg = "#01b0d3", borderwidth=2, font=("calibri", 10, "bold")).place(x = 580, y = 450, w = 80)


    window.mainloop()


main_window()
