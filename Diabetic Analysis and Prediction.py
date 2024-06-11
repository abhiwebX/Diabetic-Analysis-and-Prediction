import numpy as np
import pandas as pd
import tkinter as tk
import webbrowser
from tkinter import messagebox
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separating the features and target
X = diabetes_dataset.drop(columns="Outcome", axis=1)
Y = diabetes_dataset['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the classifier
Classifier = SVC(kernel='linear')
Classifier.fit(X_train, Y_train)

# Function to predict diabetes
def predict_diabetes():
    global X_train, Y_train
    # Retrieve values from entry fields
    pregnancies = float(entry_pregnancies.get())
    glucose = float(entry_glucose.get())
    blood_pressure = float(entry_blood_pressure.get())
    skin_thickness = float(entry_skin_thickness.get())
    insulin = float(entry_insulin.get())
    bmi = float(entry_bmi.get())
    diabetes_pedigree = float(entry_diabetes_pedigree.get())
    age = float(entry_age.get())

    # Perform prediction using the provided values
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    prediction = Classifier.predict(input_data)
    if prediction[0] == 0:
        result_label.config(text="The person is Non-Diabetic")
    else:
        result_label.config(text="The person is Diabetic")
    
    # Display the user entered values above the result
    entered_values_label.config(text=f"User Entered Values:\n\n"
                                      f"Pregnancies: {pregnancies}\n\n"
                                      f"Glucose: {glucose}\n\n"
                                      f"Blood Pressure: {blood_pressure}\n\n"
                                      f"Skin Thickness: {skin_thickness}\n\n"
                                      f"Insulin: {insulin}\n\n"
                                      f"BMI: {bmi}\n\n"
                                      f"Diabetes Pedigree: {diabetes_pedigree}\n\n"
                                      f"Age: {age}\n\n",
                                    font=("Arial", 16))  # Increase font size to 16
                                      

# Create Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry('700x600')

# Define entry fields for input data
entry_pregnancies = tk.Entry(root)
entry_glucose = tk.Entry(root)
entry_blood_pressure = tk.Entry(root)
entry_skin_thickness = tk.Entry(root)
entry_insulin = tk.Entry(root)
entry_bmi = tk.Entry(root)
entry_diabetes_pedigree = tk.Entry(root)
entry_age = tk.Entry(root)

# Place entry fields on the GUI using grid layout
entry_pregnancies.grid(row=0, column=1, padx=10, pady=(20, 10))
entry_glucose.grid(row=1, column=1, padx=10, pady=(20, 10))
entry_blood_pressure.grid(row=2, column=1, padx=10, pady=(20, 10))
entry_skin_thickness.grid(row=3, column=1, padx=10, pady=(20, 10))
entry_insulin.grid(row=4, column=1, padx=10, pady=(20, 10))
entry_bmi.grid(row=5, column=1, padx=10, pady=(20, 10))
entry_diabetes_pedigree.grid(row=6, column=1, padx=10, pady=(20, 10))
entry_age.grid(row=7, column=1, padx=10, pady=(10, 20))

# Create labels for entry fields
labels = []
label_texts = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
for i, label_text in enumerate(label_texts):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=10, sticky='e')
    labels.append(label)


# function to link website to btn
def openlink():
    webbrowser.open('https://abhiwebx.github.io/myDocApp_done/')
    

# Create predict button
predict_button = tk.Button(root, text="Predict", command=predict_diabetes , bg="green", fg="white", bd=5, padx=20, pady=5, relief=tk.RAISED, font=('Arial', 12)).grid(row=8, column=0, columnspan=2, pady=10)
#predict_button.grid(row=8, column=0, columnspan=2, pady=10)

#btn linked to website  makesure command - openlink and row = 3
predict_button = tk.Button(root, text="Appointment", command=openlink , bg="black", fg="white", bd=5, padx=20, pady=5, relief=tk.RAISED, font=('italic', 12)).grid(row=9, column=0, columnspan=2, pady=10)



# Create frame to display user entered values and prediction
frame = tk.Frame(root, width=500, height=600)
frame.grid(row=0, column=2, rowspan=9, padx=10, pady=10)

# Create label to display entered values
entered_values_label = tk.Label(frame, text="")
entered_values_label.pack()

# Create label to display prediction result
result_label = tk.Label(frame, text="", font=("Arial", 16))
result_label.pack()

root.mainloop()
