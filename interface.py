import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from predict import predict_image
from Transformation import transformation_task
import os
from PIL import Image, ImageTk


class my_app(TkinterDnD.Tk):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = self.search_newest_model(model_path)
        self.title("Leaffliction")
        self.geometry("1550x950")

        lb = tk.Listbox(self, width=50, height=15)
        lb.insert(1, "Drop image here!")

        lb.drop_target_register(DND_FILES)
        lb.dnd_bind('<<Drop>>', self.retrieve_folder)
        lb.pack()

        self.canvas = tk.Canvas(self, width=800, height=400)
        self.canvas.pack()

        button = tk.Button(self, text="Predict Image", command=self.predict)
        button.pack()

        self.attributes('-topmost', True)
        self.focus_force()
        self.after(100, lambda: self.attributes('-topmost', False))
        self.mainloop()

    def put_result(self):
        if self.true_label == self.predicted_labels:
            self.canvas.create_text(400, 320, text=self.predicted_labels,
                                    font=("Arial", 40, "bold"), fill="green")
        else:
            self.canvas.create_text(400, 320, text=self.predicted_labels,
                                    font=("Arial", 40, "bold"), fill="red")

    def predict(self):
        if hasattr(self, "image_path"):
            self.predicted_labels = predict_image(
                self.model_path, self.image_path)
            self.put_result()

    def search_newest_model(self, path):
        files = os.listdir(path)
        newest = max([path+f for f in files], key=os.path.getmtime)
        return newest

    def retrieve_folder(self, event):
        if hasattr(self, "img1"):
            self.canvas.delete("all")
        self.image_path = event.data
        if self.image_path.startswith("{"):
            self.image_path = self.image_path[1:-1]
        self.true_label = os.path.basename(os.path.dirname(self.image_path))
        self.put_image_from_directory(self.image_path, self.true_label)
        self.transformed_img, transform = transformation_task(self.image_path)
        self.transformed_img = ImageTk.PhotoImage(
            Image.fromarray(self.transformed_img))
        self.put_image_from_data(f"Tranformed Img: {transform}")

    def put_image_from_directory(self, path_img, title_img):
        self.img1 = ImageTk.PhotoImage(Image.open(path_img))
        self.canvas.create_image(10, 32, anchor=tk.NW, image=self.img1)
        self.canvas.create_text(10 + self.img1.width() // 2, 20,
                                text=title_img,
                                font=("Arial", 16), fill="White")

    def put_image_from_data(self, title_img):
        self.canvas.create_image(
            510, 32, anchor=tk.NW, image=self.transformed_img)
        self.canvas.create_text(510 + self.transformed_img.width() // 2, 20,
                                text=title_img, font=("Arial", 16),
                                fill="White")


if __name__ == "__main__":
    my_app(os.getcwd() + "/submission/model/")
