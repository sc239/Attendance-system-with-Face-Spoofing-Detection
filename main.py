import datetime
import os.path
import pickle

import tkinter as tk

import face_recognition

from test import test
from PIL import Image, ImageTk
import cv2

import util


class App:
    def __init__(self):
        self.main_window = tk.Tk(className='Hello')
        self.main_window.geometry("1200x520+350+100")

        self.login_main_win = util.get_button(
            window=self.main_window,
            text='login',
            color="green",
            fg="white",
            command=self.login
        )
        self.login_main_win.place(x=750, y=200)

        self.register_main_win = util.get_button(
            window=self.main_window,
            text='register',
            color="yellow",
            fg="black",
            command=self.register
        )
        self.register_main_win.place(x=750, y=350)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.addWebcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_dir = './logs.txt'
    def addWebcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.processWebcam()

    def processWebcam(self):
        ret, frame = self.cap.read()

        self.most_recent_frame = frame
        img_ = cv2.cvtColor(self.most_recent_frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(
            image=self.most_recent_capture_pil
        )
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.processWebcam)

    def login(self):
        label = test(
            image = self.most_recent_frame,
            model_dir = 'C:/Users/Chintu/PycharmProjects/Computer Vision/Face Attendance System/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
            device_id = 0
        )


        # unknown_img_path = './.tmp.jpg'
        # cv2.imwrite(unknown_img_path, self.most_recent_frame)
        # output = subprocess.check_output(["face_recognition", self.db_dir, unknown_img_path])
        # name = output.decode('utf-8').split(',')[1][:-2]
        if label == 1:  # actual person
            name = util.recognize(self.most_recent_frame, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box("Oops!", 'Unknown User. Please try again!')
            else:
                util.msg_box("Success!", f"Welcome back\n {name}")
#               create log
                with open(self.log_dir, 'a') as f:
                    f.write(f'{name} logged in at {datetime.datetime.now()}\n')
                    f.close()
        else:
            util.msg_box("Spoofer Spoofer!", 'Pants on fire!')

        # os.remove(unknown_img_path)

    def register(self):
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("1200x520+370+120")

        self.accept_register_window = util.get_button(
            window=self.register_window,
            text='accept',
            color="green",
            fg="white",
            command=self.accept_reg_new_user
        )
        self.accept_register_window.place(x=750, y=300)

        self.tryagain_register_window = util.get_button(
            window=self.register_window,
            text='try again',
            color="gray",
            fg="white",
            command=self.tryagain_reg_new_user
        )
        self.tryagain_register_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_window, 'Input username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(
            image=self.most_recent_capture_pil
        )
        label.imgtk = imgtk
        label.configure(image=imgtk)
        # by the time, we are saving, many frames have been captured
        self.register_new_user_capture = self.most_recent_frame.copy()

    def accept_reg_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        # cv2.imwrite(os.path.join(self.db_dir, f'{name}.jpg'), self.register_new_user_capture)
        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]
        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box("Success", "Registered successfully")

        self.register_window.destroy()

    def tryagain_reg_new_user(self):
        self.register_window.destroy()

    def start(self):
        self.main_window.mainloop()


if __name__ == '__main__':
    app = App()
    app.start()
