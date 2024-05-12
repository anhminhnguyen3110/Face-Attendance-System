import os.path
import datetime
import pickle
import threading
from time import sleep
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition

import util


class App:
    def __init__(self):
        # Initialize the main window
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        # Database directory setup
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = './log.txt'

        # Variable to toggle showing the box
        self.show_box = False

        # Setting up buttons
        self.setup_buttons()

        # Webcam display label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        # Initialize webcam
        self.add_webcam(self.webcam_label)

        # Start the Tkinter main loop
        self.main_window.mainloop()

    def setup_buttons(self):
        # Button to check attendance
        self.check_attendance_button_main_window = util.get_button(self.main_window, 'Check attendance', 'green', self.check_attendance)
        self.check_attendance_button_main_window.place(x=750, y=200)

        # Button to identify (color changes based on self.show_box)
        self.identify_button_color = 'green' if self.show_box else 'red'
        self.identify_button_main_window = util.get_button(self.main_window, 'Identify', self.identify_button_color, self.identify)
        self.identify_button_main_window.place(x=750, y=300)

        # Button to register a new user
        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray', self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

    def update_identify_button_color(self):
        # Update the color based on self.show_box
        new_color = 'green' if self.show_box else 'red'
        self.identify_button_main_window.config(bg=new_color)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()
        
    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            self._label.after(1, self.process_webcam)
            return  # Exit the function if no frame is captured
        # Continue with processing if the frame was captured successfully
        self.most_recent_capture_arr = frame
        
        if self.show_box:
            self.show_box_function(frame)
            
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        self._label.after(1, self.process_webcam)

    def show_box_function(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Ensure the frame is in RGB for face_recognition
        face_locations = face_recognition.face_locations(rgb_frame)
        face_name = "No Face"
        if face_locations:
            face_name = util.recognize(rgb_frame, self.db_dir)  # Ensure that the correct frame format is used for recognition
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, face_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                roi_gray = frame[top:bottom, left:right]
                roi_color = frame[top:bottom, left:right]
                # Detect eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(frame, 'Eye', (left + ex, top + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Detect smiles
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(30, 30))
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                    cv2.putText(frame, 'Smile', (left + sx, top + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    def check_attendance(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
            with open(self.log_path, 'a') as f:
                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                f.close()

    def identify(self):
        self.show_box = not self.show_box
        self.update_identify_button_color()


    def register_new_user(self):
        if(self.show_box):
            util.msg_box('Error', 'Please, stop identification before registering new user.')
            return
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)
        
    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)
        
        if not embeddings:
            util.msg_box('Error', 'No face found. Please try again.')
            return

        embeddings = embeddings[0]
        
        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()