import sys
import cv2
import numpy as np
import mediapipe as mp
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        # Load models & Labels
        self.model_dict_42 = pickle.load(open('./model_42.p', 'rb'))
        self.model_42 = self.model_dict_42['model']

        self.label_42 = {}
        for i in range(26):
            self.label_42.update({i:chr(65 + i)})

        # Initialize CV2 + Tools for Handdrawing
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Initialize App's variables
        self.counter = 0
        self.word_input = ""
        self.current_letter_index = 0

        # Initialize Updater
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
    
    def init_ui(self):
        # Basic Initialization of window
        self.setWindowTitle('Sign Language Converter')
        self.setGeometry(100,100,800,600)

        # Create Layout
        layout = QVBoxLayout()

        # Display Current Letter
        self.label_current_letter = QLabel("Current Letter to sign: None", self)
        layout.addWidget(self.label_current_letter)

        # Display Video
        self.label_video = QLabel(self)
        layout.addWidget(self.label_video)

        # Display Text Input
        self.word_text_input = QLineEdit(self)
        self.word_text_input.setPlaceholderText("Enter a word to sign...")
        layout.addWidget(self.word_text_input)

        # Start Button
        self.start_button = QPushButton('Start Signing', self)
        self.start_button.clicked.connect(self.start_signing)
        layout.addWidget(self.start_button)

        # Add layout to screen
        self.setLayout(layout)
    


    def start_signing(self):
        self.word_input = self.word_text_input.text().upper()
        self.current_letter_index = 0
        if self.word_input:
            self.label_current_letter.setText(f"Current Letter to Sign: {self.word_input[self.current_letter_index]}")
            self.timer.start(10)
        else:
            self.label_current_letter.setText("Enter a word")
    

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw handmarks on frame
                    self.draw_hand_landmarks(frame, hand_landmarks)

                    # Check handsign
                    if self.check_sign(hand_landmarks):
                        self.counter += 1
                        if (self.counter == 5):
                            self.change_letter()
            
            # Convert OpenCV frame to QImage for PyQt5 display
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.label_video.setPixmap(QPixmap.fromImage(qimg))
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            frame, # image to draw
            hand_landmarks, # model output
            self.mp_hands.HAND_CONNECTIONS, # hand connections
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style())

    def change_letter(self):
        self.current_letter_index += 1
        self.counter = 0
        if self.current_letter_index == len(self.word_input):
            self.label_current_letter.setText("You have signed the word correctly! Goodjob! Enter another word!")
            self.label_video.clear()
            self.label_video.setVisible(False)
            self.current_letter_index = 0

        else:
            self.label_current_letter.setText(f"Current Letter to Sign: {self.word_input[self.current_letter_index]}")
        


    def check_sign(self, hand_landmarks):
        data_aux = []
        for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
        
        if len(data_aux) == 42:
            prediction = self.model_42.predict([np.asarray(data_aux)])

            predicted_class = int(prediction[0])

            # Check if the predicted class exists in the dictionary
            if predicted_class in self.label_42:
                return self.label_42[predicted_class] == self.word_input[self.current_letter_index]
        
        return False
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())
