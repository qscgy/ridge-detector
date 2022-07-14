import tkinter as tk
from PIL import ImageTk, Image
import os
from natsort import natsorted

root = tk.Tk()
root.geometry("600x500")
root.configure(background='grey')

text_widget = tk.Entry(root)
text_widget.pack(anchor = "w", padx = 20, pady = 20)
text_widget.focus_set()

base_dir = '/bigpen/Datasets/jhu-released'
im_dir = 'seq1'
image_names = natsorted(os.listdir(os.path.join(base_dir, im_dir)))
image_names = image_names[::20]

paths = {}

paths['idx'] = 0
paths['impath'] = os.path.join(base_dir, im_dir, image_names[paths['idx']])
img = ImageTk.PhotoImage(Image.open(paths['impath']).resize((500,400)))

panel = tk.Label(root, image=img)

panel.pack()

fname = f'label_counts_{im_dir}.txt'
label_file = open(fname, 'a')

def key_return(e, paths):
    if e.keycode==36 or e.keycode==104:
        label_file.write(paths['impath'] + ',' + text_widget.get() + '\n')
        paths['idx'] = paths['idx']+1
        paths['impath'] = os.path.join(base_dir, im_dir, image_names[paths['idx']])
        text_widget.delete(0,tk.END)
        img = ImageTk.PhotoImage(Image.open(paths['impath']).resize((500,400)))
        panel.configure(image=img)
        panel.image = img


root.bind('<KeyPress>', lambda e: key_return(e, paths))
root.mainloop()
label_file.close()
