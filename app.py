import streamlit as st


import numpy as np
import pandas as pd
import csv
import re
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
# get_ipython().system('pip install pyttsx3')
# import pyttsx3
#starting 
# engine = pyttsx3.init()

# def text_to_speech(text):
#     engine.setProperty('rate', 150)    # Speed percent (can go over 100)
#     engine.setProperty('volume', 0.9)  # Volume 0-1
#     engine.say(text)
#     engine.runAndWait()

#loading the training and testing dataset
training = pd.read_csv('Training.csv')
testing= pd.read_csv('Testing.csv')

cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']

reduced_data = training.groupby(training['prognosis']).max()
reduced_data.head()
reduced_data.index = reduced_data.index.str.strip()
reduced_data.to_csv("a.csv", index=False)

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
set(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)

clf1 = DecisionTreeClassifier(max_depth=45)  

clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
#st.header("Decision Tree classifier")
#st.success(f"Mean Score: {scores.mean()}")
train_accuracy = clf.score(x_train, y_train)
val_accuracy = scores.mean()

#streamlit output
#st.info(f"Training Accuracy: {train_accuracy:}")
#st.info(f"Validation Accuracy (Cross-Validation): {val_accuracy:}")

#graph part
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True)
plt.show()


from sklearn.metrics import accuracy_score
# Predicting on the test set
y_pred = clf.predict(x_test)
# Calculating accuracy
test_accuracy = accuracy_score(y_test, y_pred)
# st.infoing accuracy
#st.info(f"Test Accuracy: {test_accuracy}")



path = clf1.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train and validate models with different pruning levels
clfs = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(x_train, y_train)
    clfs.append(tree)

# Compare performance for each alpha
train_scores = [tree.score(x_train, y_train) for tree in clfs]
test_scores = [cross_val_score(tree, x_test, y_test, cv=3).mean() for tree in clfs]

#st.info(f"Best model test score: {max(test_scores):.2f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, label="Training accuracy")
plt.plot(ccp_alphas, test_scores, label="Validation accuracy")
plt.xlabel("Effective Alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



model = SVC(C=0.1, kernel='rbf')

model.fit(x_train,y_train)
#st.info(f"Accuracy score for svm:  {model.score(x_test,y_test)}")
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


importances = clf.feature_importances_
#for i, imp in enumerate(importances):
#   st.info(f"Feature {features[i]} importance: {imp:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(features, importances, color='skyblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Feature scaling for models that are sensitive to feature magnitudes
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --- SVM Model with Higher C and Linear Kernel ---

svm_model = SVC(C=1.0, kernel='linear', random_state=42)
svm_model.fit(x_train_scaled, y_train)
svm_train_score = svm_model.score(x_train_scaled, y_train)
svm_test_score = svm_model.score(x_test_scaled, y_test)

# --- Random Forest Model with Regularization ---
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=42)
rf_model.fit(x_train, y_train)  # No scaling needed for Random Forest
rf_train_score = rf_model.score(x_train, y_train)
rf_test_score = rf_model.score(x_test, y_test)

# --- K-Nearest Neighbors Model with More Neighbors ---
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(x_train_scaled, y_train)
knn_train_score = knn_model.score(x_train_scaled, y_train)
knn_test_score = knn_model.score(x_test_scaled, y_test)

# --- Cross-Validation using Stratified K-Fold (to balance class distribution) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_cv_score = cross_val_score(svm_model, x_train_scaled, y_train, cv=cv).mean()
rf_cv_score = cross_val_score(rf_model, x_train, y_train, cv=cv).mean()
knn_cv_score = cross_val_score(knn_model, x_train_scaled, y_train, cv=cv).mean()

# --- Results ---

#st.info(f"SVM Model Test Accuracy: {svm_test_score}")
#st.info(f"Random Forest Model Test Accuracy: {rf_test_score}")
#st.info(f"KNN Model Test Accuracy: {knn_test_score}")

#st.info(f"SVM Model Cross-Validation Score: {svm_cv_score}")
#st.info(f"Random Forest Model Cross-Validation Score: {rf_cv_score}")
#st.info(f"KNN Model Cross-Validation Score: {knn_cv_score}")


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
symptoms_dict = {}
for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        st.info("You should take the consultation from doctor. ") #OUTPUT SEVERITY TO STREAMLIT
    else:
        st.info("It might not be that bad but you should take precautions.") #OUTPUT SEVERITY TO STREAMLIT

def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


# Ensure data integrity
data = {
    "Disease": [
        "(vertigo) Paroymsal  Positional Vertigo", "AIDS", "Acne", "Alcoholic hepatitis", "Allergy", "Arthritis", 
        "Bronchial Asthma", "Cervical spondylosis", "Chicken pox", "Chronic cholestasis", "Common Cold", "Dengue", 
        "Diabetes", "Dimorphic hemmorhoids(piles)", "Drug Reaction", "Fungal infection", "GERD", 
        "Gastroenteritis", "Heart attack", "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", 
        "Hypertension", "Hyperthyroidism", "Hypoglycemia", "Hypothyroidism", "Impetigo", "Jaundice", "Malaria", "Migraine", 
        "Osteoarthristis", "Paralysis (brain hemorrhage)", "Peptic ulcer disease", "Pneumonia", "Psoriasis", "Tuberculosis", 
        "Typhoid", "Urinary tract infection", "Varicose veins", "hepatitis A"
    ],
    "Medicine": [
        "Meclizine and Betahistine", "Tenofovir and Emtricitabine and Efavirenz", "Tretinoin and Adapalene and Doxycycline and Minocycline", 
        "Prednisolone and Pentoxifylline", "Diphenhydramine and Loratadine and Fluticasone", "Ibuprofen and Naproxen and Methotrexate", 
        "Salbutamol and Fluticasone and Budesonide", "Ibuprofen and Cyclobenzaprine", "Acyclovir and Calamine lotion", 
        "Ursodeoxycholic acid", "Paracetamol and Saline nasal spray and Pseudoephedrine", "Paracetamol and Oral rehydration salts", 
        "Insulin glargine and Insulin lispro and Metformin and Glipizide", "Hydrocortisone cream and Docusate sodium", 
        "Hydrocortisone and Diphenhydramine", "Clotrimazole and Fluconazole", "Omeprazole and Esomeprazole and Ranitidine", 
        "Oral rehydration salts and Ondansetron", "Aspirin and Clopidogrel and Atenolol and Atorvastatin", 
        "Tenofovir and Entecavir", "Sofosbuvir and Ledipasvir and Daclatasvir", "Pegylated interferon alpha", "Ribavirin and Supportive care", 
        "Lisinopril and Ramipril and Amlodipine and Hydrochlorothiazide", "Methimazole and Propylthiouracil and Propranolol", 
        "Glucose tablets and Glucagon", "Levothyroxine and Euthyrox and Synthroid", "Mupirocin and Retapamulin", 
        "Supportive care and Antibiotics (for infections) and Antivirals (for hepatitis)", 
        "Artesunate and Chloroquine and Artemisinin-based combination therapies (ACTs)", "Sumatriptan and Rizatriptan and Ibuprofen", 
        "Ibuprofen and Diclofenac and Glucosamine sulfate", "Rehabilitation therapy and Paracetamol and Anticoagulants (if required)", 
        "Omeprazole and Lansoprazole and Ranitidine and Clarithromycin and Amoxicillin", "Amoxicillin and Levofloxacin and Oseltamivir", 
        "Betamethasone and Fluocinonide and Calcipotriene and Adalimumab and Etanercept", 
        "Isoniazid and Rifampin and Pyrazinamide and Ethambutol", "Ciprofloxacin and Ceftriaxone", "Ciprofloxacin and Nitrofurantoin", 
        "Compression stockings and Sclerotherapy and Endovenous laser therapy (EVLT) and Surgery", "Supportive care and Vaccination"
    ]
}

df = pd.DataFrame(data)
csv_file_path = "diseases_and_medicines17.csv"
df.to_csv(csv_file_path, index=False)

def getMedicineDict():
    global medicineDictionary
    with open('diseases_and_medicines17.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            medicineDictionary[row[0]] = row[1]


# Function to retrieve medicine recommendations
def getMedicine(disease):
    if disease in medicineDictionary:
        pass
        st.info(f"Recommended medicine for {disease}: {medicineDictionary[disease]}") #OUTPUT MEDICINE TO STREAMLIT
    else:
        pass
        st.info(f"No medicine information available for {disease}") #OUTPUT MEDICINE TO STREAMLIT

def getInfo():
    st.title("MEDIFORECASTER: MULTI DISEASE PREDICTION SYSTEM")
    st.header("Enter your name:")
    username=st.text_input("Name",key="username")
    if not username:
        st.warning("Enter you name")
        st.stop()

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)
    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

def tree_to_code(tree, feature_names):
    # Get the underlying tree structure from the classifier
    tree_ = tree.tree_

def tree_to_code(tree, feature_names):
    # Get the underlying tree structure from the classifier
    tree_ = tree.tree_

    # Map feature indices to feature names for easy reference
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    # Prepare a list of symptom names for validation
    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []  # List to hold symptoms present during the recursion

    import streamlit as st

    st.title("Symptom Checker")

# Use text-to-speech to prompt the user for input
    # engine.say("Enter the symptom you are experiencing.")

    disease_input=st.text_input("Enter the symptoms you are experiencing :",key="disease_input")
    if not disease_input:
        st.warning("please Enter a symptom")
        st.stop()
    
    conf,cnf_dis=check_pattern(chk_dis,disease_input)
    if conf==1 and cnf_dis:
        st.info("Related Symptoms Found")
        selected_symptom=st.selectbox("Select the symptom you meant:", cnf_dis, key="selected_symtptom")
        st.success(f"you selected {selected_symptom}")
    else:
        st.warning("Enter a valid symptom")
        st.stop()

    while True:
        try:
            # Ask the user how many days they've been experiencing the symptoms
            num_days = st.number_input("For how many days?", step=1)  #INPUT NO. OF DAYS TO STREAMLIT
            break  # Exit loop on valid input
        except ValueError:
            st.info("Enter a valid input.")  #ASK FOR VALID NO OF DAYS 

    # Recursive function to traverse the decision tree
    def recurse(node, depth):
        indent = "  " * depth  # Create indentation based on depth for clarity
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # Check if node is a decision node
            name = feature_name[node]  # Get the feature name
            threshold = tree_.threshold[node]  # Get the threshold for the decision

            # Determine if the input symptom matches the feature
            val = 1 if name == disease_input else 0

            # Recursively traverse left or right based on the threshold
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)  # Traverse left subtree
            else:
                symptoms_present.append(name)  # Record the symptom
                recurse(tree_.children_right[node], depth + 1)  # Traverse right subtree
        else:
            # If a leaf node is reached, get the predicted disease
            present_disease = print_disease(tree_.value[node])

            # st.info the predicted disease at this step
            st.info(f"{indent}Predicted disease at this step: {present_disease[0]}") #OUTPUT PREDICTED DISEASE AT INTERMEDIATE STEP

            # Get relevant symptoms for the predicted disease
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            # Use text-to-speech to ask the user about additional symptoms
            # engine.say("Are you experiencing any symptoms?")
            st.info("Are you experiencing any:") #ASK FOR ADDITIONAL SYMPTOM 

            symptoms_exp = []  # List to hold additional symptoms

            counter=1

            # Ask the user about each of the relevant symptoms
            for syms in list(symptoms_given):
                # engine.say(f"{syms}, are you experiencing it?")
                st.info(f"{syms}? : ") #ASK FOR RELEVEANT SYMPTOMS 
                
                inp=st.text_input("Any Relevant Symptoms as per previous input?",key=counter)
                counter+=1
                if not inp:
                    st.warning("Input required [yes/no]")
                    st.stop()
                st.success(f"you have entered {inp}")
                #counter+=1
                    #st.session_state.inp = st.text_input("Any Relevant Symptoms as per previous input?" ,key=unique_key)  #INPUT RELEVANT SYMPTOMS AS PER THE PREVIOUS ONES TO STREAMLIT
                    

                # st.info the user's response after each inquiry
                st.info(f"You answered: {inp}") #OUTPUT USER'S YES/NO CHOICE

                if inp == "yes":
                    symptoms_exp.append(syms)   # Add to symptoms list if the answer is yes

                # After each symptom inquiry, perform prediction
                second_prediction = sec_predict(symptoms_exp)  # Perform a second prediction
                st.info(f"Current prediction based on your symptoms: {present_disease[0]} or {second_prediction[0]}") #OUTPUT SECOND PREDICTION

            # Evaluate the condition based on symptoms and number of days
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:  # Compare predictions
                # If both predictions match
                # engine.say(f"You may have {present_disease[0]}.")
                st.info(f"You may have {present_disease[0]}.") #FINAL PREDICTION
                st.info(description_list[present_disease[0]])  #Show disease description

            else:
                # If predictions differ, inform the user
                # engine.say(f"You may have {present_disease[0]} or {second_prediction[0]}.")
                st.success(f"You may have {present_disease[0]} or {second_prediction[0]}.") # FINAL PREDICTION output
                st.success(description_list[present_disease[0]])  # Show description of first disease output
                st.info(description_list[second_prediction[0]])  # Show description of second disease output

            # Display precautions for the predicted disease
            precaution_list = precautionDictionary[present_disease[0]]
            st.info("Take the following measures:") # PRECAUTIONS output
            for i, j in enumerate(precaution_list):  # List precautions
                st.info(f"{i + 1} ) {j}") #OUTPUT PRECAUTIONS 

            # Get recommended medicine for the predicted disease
            getMedicine(present_disease[0])  # Display medicine recommendations

    # Start the recursion from the root of the tree
    recurse(0, 1)

# Load medicine data from CSV
def getMedicineDict():
    global medicineDictionary
    medicineDictionary = {}
    with open('diseases_and_medicines17.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            medicineDictionary[row[0]] = row[1]

# Function to retrieve medicine recommendations
def getMedicine(disease):
    if disease in medicineDictionary:
        st.info(f"Recommended medicine for {disease}: {medicineDictionary[disease]}") #OUTPUT MEDICINE 
        pharma_link = f"https://www.pharmeasy.in/search/all?name={medicineDictionary[disease].replace(' ', '%20')}"
        st.markdown(f"[Buy {medicineDictionary[disease]} on PharmEasy]({pharma_link})", unsafe_allow_html=True)
        st.markdown(f"[Consult a Doctor for {disease} on PharmEasy](https://pharmeasy.in/doctor-consultation/landing?src=homecard)", unsafe_allow_html=True)
    else:
        st.info(f"No medicine information available for {disease}") #OUTPUT NO MEDICINE

getSeverityDict()  # Load severity data
getDescription()   # Load symptom descriptions
getprecautionDict()# Load precaution measures
getMedicineDict()  # Load disease-to-medicine mapping
getInfo()  # Gather user's name and greeting
tree_to_code(clf, cols)  # Call the tree_to_code function with the classifier and features




plt.figure(figsize=(20, 10))
# Plot the decision tree
plot_tree(clf, 
          feature_names=features,  # Column names as feature names
          class_names=le.classes_,  # Class labels
          filled=True,  # Add colors for better readability
          rounded=True,  # Use rounded nodes
          fontsize=10)  # Font size for better readability
# Show the tree
plt.show()

from sklearn.tree import export_text
# Generate text representation
tree_text = export_text(clf, feature_names=list(features))
#st.info(tree_text)


plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=features, class_names=le.classes_, filled=True, rounded=True, fontsize=10)
plt.savefig("decision_tree_visualization.png")

# from PIL import Image

# # Load an image
# image = Image.open("decision_tree_visualization.png")

# # Display the image
# st.image(image, caption="Example Image", width=None)

#import joblib
# Save the model to a .pkl file
#model_filename = 'decision_tree_model.pkl'
#joblib.dump(clf, model_filename)
#st.info(f"Model saved as {model_filename}")

#import pickle
# Example model creation
#clf1 = DecisionTreeClassifier(max_depth=45)  
#clf = clf1.fit(x_train,y_train)
#model = clf
# Save the model
#with open('decision_tree5_model.pkl', 'wb') as f:
#    pickle.dump(model, f)
