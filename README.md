<H3>Name: GANANANTH H</H3>
<H3>Register no : 212225230070</H3>
<H3>Date : 3-11-2026</H3>
<H3>Experiment No. 2 </H3>

## Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
    ''' 
    import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("diabetes.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Replace impossible zero values with NaN
cols = ['Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI']

df[cols] = df[cols].replace(0, np.nan)

# Fill missing values using median
for col in cols:
            df[col] = df[col].fillna(df[col].median())

 # Check duplicates and missing values again
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())
print("\nDataset Shape After Preprocessing:")
print(df.shape)



# Class distribution
df['Outcome'].value_counts().plot(kind='bar',title='Diabetes Class Distribution')

plt.xlabel('Outcome (0 = Non-Diabetic, 1 = Diabetic)')
plt.ylabel('Count')
plt.show()

        # Scatter plot
plt.scatter(df[df['Outcome'] == 0]['Glucose'],
                    df[df['Outcome'] == 0]['BMI'],
                                label='Non-Diabetic')

plt.scatter(df[df['Outcome'] == 1]['Glucose'],
                                            df[df['Outcome'] == 1]['BMI'],
                                                        label='Diabetic')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.title('Diabetes Prediction')
plt.legend()
plt.show()

X = df.drop('Outcome', axis=1).values

y = df['Outcome'].values

print("Input Features:")
print(X[:5])

print("\nTarget Output:")
print(y[:5])

X_train, X_test, y_train, y_test = train_test_split(
      X, y,
          test_size=0.2,
              random_state=42,
                  stratify=y
                  )

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("Scaled Training Data:")
print(X_train[:5])

X_train = np.c_[np.ones(X_train.shape[0]), X_train]

X_test = np.c_[np.ones(X_test.shape[0]), X_test]

print("Input Vector X:")
print(X_train[:5])

Y_train = np.where(y_train == 1, 1, -1)

Y_test = np.where(y_test == 1, 1, -1)

print("Desired Output Y:")
print(Y_train[:10])

W = np.zeros(X_train.shape[1])

learning_rate = 0.01

epochs = 100

print("Initial Weight Vector:")
print(W)

print("Learning Rate:", learning_rate)

errors = []

for epoch in range(epochs):
    total_error = 0
    for i in range(len(X_train)):
        v = np.dot(W, X_train[i])
        T = 1 if v >= 0 else -1
        if Y_train[i] != T:
            W = W + learning_rate * (Y_train[i] - T) * X_train[i]
            total_error += 1
    errors.append(total_error)

print("Final Weight Vector:")
print(W)

plt.plot(range(1, epochs + 1), errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')

plt.title('Perceptron Training Error')

plt.grid()
plt.show()

y_pred = []

for i in range(len(X_test)):
  v = np.dot(W, X_test[i])
  T = 1 if v >= 0 else -1
  y_pred.append(T)
y_pred = np.array(y_pred)
accuracy = accuracy_score(Y_test, y_pred)
print("Actual Output:")
print(Y_test[:20])

print("\nPredicted Output:")
print(y_pred[:20])
print("\nAccuracy:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    Y_test,
    y_pred,
    labels=[-1, 1],
    target_names=['Non-Diabetic', 'Diabetic'],
    zero_division=0))

    
# New patient data
# Order: Pregnancies, Glucose, BloodPressure,
# SkinThickness, Insulin, BMI,
# DiabetesPedigreeFunction, Age

new_data = np.array([[2, 150, 80, 25, 120, 32.5, 0.5, 35]])

# Apply the same feature scaling
new_data_scaled = scaler.transform(new_data)

# Add bias
new_data_scaled = np.c_[np.ones(new_data_scaled.shape[0]), new_data_scaled]

# Prediction
v = np.dot(W, new_data_scaled[0])

prediction = 1 if v >= 0 else -1

# Display result
if prediction == 1:
    print("Prediction: Diabetic")
else:
  print("Prediction: Non-Diabetic") 
    
    '''

# OUTPUT:

    '''
    
    <img width="912" height="318" alt="image" src="https://github.com/user-attachments/assets/9e8c765e-997f-4daa-8c50-96657e56b6fb" />

    <img width="802" height="421" alt="image" src="https://github.com/user-attachments/assets/75b4c16c-b83b-48dd-9726-3a18e3084da7" />

    <img width="722" height="588" alt="image" src="https://github.com/user-attachments/assets/a6af921a-b0ae-4024-8d19-570e6b7104db" />
    
    <img width="1007" height="601" alt="image" src="https://github.com/user-attachments/assets/e7b2a58b-fed6-4ec7-87ca-eb3681919a1d" />
    <img width="881" height="590" alt="image" src="https://github.com/user-attachments/assets/ff3bb9a1-1efd-4909-a3ce-ef248ad4173a" />

    <img width="811" height="230" alt="image" src="https://github.com/user-attachments/assets/cf122202-8a6e-45b6-8c0d-8d40a6210e2a" />\
    <img width="380" height="77" alt="image" src="https://github.com/user-attachments/assets/5b866f04-30bd-45e2-86d0-46e9ddb23b06" />
    <img width="990" height="377" alt="image" src="https://github.com/user-attachments/assets/e1877292-6428-4a6f-a04c-945f86d98d98" />
    <img width="922" height="276" alt="image" src="https://github.com/user-attachments/assets/d11fa57e-3dfa-4a6d-919e-3d37dae4b390" />
    <img width="431" height="77" alt="image" src="https://github.com/user-attachments/assets/e32d346f-896c-411d-ba97-7889205cf97e" />
    <img width="568" height="107" alt="image" src="https://github.com/user-attachments/assets/1d78391d-d453-4e7f-b8b1-b06ac2702c44" />
    <img width="908" height="100" alt="image" src="https://github.com/user-attachments/assets/07456b06-9797-4081-9254-2728fbd6c79c" />
    <img width="975" height="641" alt="image" src="https://github.com/user-attachments/assets/2f677bed-d1f7-431b-9e22-102edc41b395" />
    <img width="825" height="550" alt="image" src="https://github.com/user-attachments/assets/1c59b3db-fe48-41e7-8327-5dedc8f70019" />
    <img width="871" height="595" alt="image" src="https://github.com/user-attachments/assets/e776e886-38b8-45e8-b97e-047b09eb5abf" />

'''

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
