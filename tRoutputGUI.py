import tkinter as tk

#Purpose: Create a Ouput GUI for output of Translation and Rotation Vectors

#def disp_PoseVal(ids, combpairs, id_rvec, id_tvec, r_rel, t_rel)
#marker id 
a=[3.5, 2.5, 8]
a = " ".join(str(elem) for elem in a)
#a_str=''.join(a)
print(a)
#tranlation
#need to add an asterix to unpack range!
b = [[*range(3)], [*range(2,5)], [*range(8,11)]] 
print(b)   
#rotation                   
c=b

rootwindow = tk.Tk()
rootwindow.title("ARUCO Marker Pose information")
rootwindow.geometry('{}x{}'.format(800, 500))

#Frame 0: IDs
frame_0 = tk.LabelFrame(rootwindow,text="Detected Marker IDs")
frame_0.pack(side="top")

#Read IDs
rids = tk.Text(frame_0)
rids.insert("end",a)
rids.grid(row=0,columnspan=2)
#rids.pack()


#Frame 1: WRT to camera
frame_1 = tk.LabelFrame(rootwindow, height=100, text="Pose Vectors: Tranlation+ Rotation")
frame_1.pack()

#Translation Vector
Trans=tk.Text(frame_1)
for x in b:
    ttemp = " ".join(str(x))
    print(ttemp)
    print('type',type(ttemp))
    Trans.insert("end", ttemp + '\n')
Trans.grid(row=1,column=0)


#Rotation Vector
Rot=tk.Text(frame_1)
for y in c:
    rtemp = " ".join(str(y))
    print(rtemp)
    print('type', type(rtemp))
    Rot.insert("end", rtemp + '\n')
Rot.grid(row=1, column=1)


rootwindow.mainloop()
#use rootwindow.after to update the display with every frame
