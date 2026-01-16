from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle


#load databases 
symptom_data = pd.read_csv('Dataset/symtoms_df.csv')
precaution_data = pd.read_csv('Dataset/precautions_df.csv')
workout_data = pd.read_csv('Dataset/workout_df.csv')
description_data = pd.read_csv('Dataset/description.csv')
medication_data = pd.read_csv('Dataset/medications.csv')
diets_data = pd.read_csv('Dataset/diets.csv')


#Load Model
# svc_loaded = pickle.load(open('Models/svc.pkl','rb'))
feature_columns = pickle.load(open("Models/features.pkl", "rb"))

def helper(predicted_disease):

    predicted_disease = predicted_disease.strip()

    # Description
    description_series = description_data[
        description_data['Disease'] == predicted_disease
    ]['Description']
    description = " ".join(description_series.astype(str).values)

    # Precautions
    precaution_df = precaution_data[
        precaution_data['Disease'] == predicted_disease
    ][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    precaution = precaution_df.values.flatten().tolist()

    # Medication
    medication = medication_data[
        medication_data['Disease'] == predicted_disease
    ]['Medication'].astype(str).tolist()

    # Diet
    diet = diets_data[
        diets_data['Disease'] == predicted_disease
    ]['Diet'].astype(str).tolist()

    # Workout
    workout = workout_data[
        workout_data['disease'] == predicted_disease
    ]['workout'].astype(str).tolist()

    return description, precaution, medication, diet, workout


# def helper(predicted_disease):
    
#     description = description_data[description_data['Disease'] == predicted_disease]['Description']
#     description = " ".join([w for w in description])
    
#     c = precaution_data[precaution_data['Disease']== predicted_disease][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
#     # precaution = [col for col in precaution.values]
#     precaution = c.values.flatten().tolist()

#     medication = medication_data[medication_data['Disease']== predicted_disease]['Medication']
#     medication = [m for m in medication.values]
    
#     diet = diets_data[diets_data['Disease']== predicted_disease]['Diet']
#     diet = [ d for d in diet.values]
    
#     workout = workout_data[workout_data['disease']== predicted_disease]['workout']
    
#     return description,precaution,medication,diet,workout

# symptoms_dict = {"itching":0,"skin_rash":1,"nodal_skin_eruptions":2,"continuous_sneezing":3,"shivering":4,"chills":5,"joint_pain":6,"stomach_pain":7,"acidity":8,"ulcers_on_tongue":9,"muscle_wasting":10,"vomiting":11,"burning_micturition":12,"spotting_urination":13,"fatigue":14,"weight_gain":15,"anxiety":16,"cold_hands_and_feets":17,"mood_swings":18,"weight_loss":19,"restlessness":20,"lethargy":21,"patches_in_throat":22,"irregular_sugar_level":23,"cough":24,"high_fever":25,"sunken_eyes":26,"breathlessness":27,"sweating":28,"dehydration":29,"indigestion":30,"headache":31,"yellowish_skin":32,"dark_urine":33,"nausea":34,"loss_of_appetite":35,"pain_behind_the_eyes":36,"back_pain":37,"constipation":38,"abdominal_pain":39,"diarrhoea":40,"mild_fever":41,"yellow_urine":42,"yellowing_of_eyes":43,"acute_liver_failure":44,"fluid_overload":45,"swelling_of_stomach":46,"swelled_lymph_nodes":47,"malaise":48,"blurred_and_distorted_vision":49,"phlegm":50,"throat_irritation":51,"redness_of_eyes":52,"sinus_pressure":53,"runny_nose":54,"congestion":55,"chest_pain":56,"weakness_in_limbs":57,"fast_heart_rate":58,"pain_during_bowel_movements":59,"pain_in_anal_region":60,"bloody_stool":61,"irritation_in_anus":62,"neck_pain":63,"dizziness":64,"cramps":65,"bruising":66,"obesity":67,"swollen_legs":68,"swollen_blood_vessels":69,"puffy_face_and_eyes":70,"enlarged_thyroid":71,"brittle_nails":72,"swollen_extremeties":73,"excessive_hunger":74,"extra_marital_contacts":75,"drying_and_tingling_lips":76,"slurred_speech":77,"knee_pain":78,"hip_joint_pain":79,"muscle_weakness":80,"stiff_neck":81,"swelling_joints":82,"movement_stiffness":83,"spinning_movements":84,"loss_of_balance":85,"unsteadiness":86,"weakness_of_one_body_side":87,"loss_of_smell":88,"bladder_discomfort":89,"foul_smell_of_urine":90,"continuous_feel_of_urine":91,"passage_of_gases":92,"internal_itching":93,"toxic_look_(typhos)":94,"depression":95,"irritability":96,"muscle_pain":97,"altered_sensorium":98,"red_spots_over_body":99,"belly_pain":100,"abnormal_menstruation":101,"dischromic_patches":102,"watering_from_eyes":103,"increased_appetite":104,"polyuria":105,"family_history":106,"mucoid_sputum":107,"rusty_sputum":108,"lack_of_concentration":109,"visual_disturbances":110,"receiving_blood_transfusion":111,"receiving_unsterile_injections":112,"coma":113,"stomach_bleeding":114,"distention_of_abdomen":115,"history_of_alcohol_consumption":116,"fluid_overload.1":117,"blood_in_sputum":118,"prominent_veins_on_calf":119,"palpitations":120,"painful_walking":121,"pus_filled_pimples":122,"blackheads":123,"scurring":124,"skin_peeling":125,"silver_like_dusting":126,"small_dents_in_nails":127,"inflammatory_nails":128,"blister":129,"red_sore_around_nose":130,"yellow_crust_ooze":131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# def predict_disease(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))

#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc_loaded.predict([input_vector])[0]]

# def predict_disease(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))

#     for item in patient_symptoms:
#         item = item.lower().replace(" ", "_")
#         if item in symptoms_dict:
#             input_vector[symptoms_dict[item]] = 1
#         else:
#             print(f"Unknown symptom ignored: {item}")

#     return diseases_list[svc_loaded.predict([input_vector])[0]]

# get feature order exactly as training
# symptom_columns = symptom_data.columns.tolist()

svc_loaded = pickle.load(open("Models/svc.pkl", "rb"))
def predict_disease(patient_symptoms):
    input_vector = np.zeros(len(feature_columns))

    for symptom in patient_symptoms:
        symptom = symptom.lower().replace(" ", "_")
        if symptom in feature_columns:
            idx = feature_columns.index(symptom)
            input_vector[idx] = 1

    prediction = svc_loaded.predict(input_vector.reshape(1, -1))[0]
    return diseases_list[prediction]


      

app = Flask(__name__)

#creaing routes
@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')

    if not symptoms:
        return render_template('index.html', error="No symptoms entered")

    user_symptoms = [s.strip() for s in symptoms.split(',')]
    user_symptoms = [symptom.strip("[]'") for symptom in user_symptoms]

    predicted_disease = predict_disease(user_symptoms)

    # 🔍 DEBUG LINE (PUT THIS HERE)
    print("DEBUG:", predicted_disease)

    description, precaution, medication, diet, workout = helper(predicted_disease)

    return render_template(
        'index.html',
        predicted_disease=predicted_disease,
        description=description,
        precaution=precaution,
        medication=medication,
        diet=diet,
        workout=workout
    )

    

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')

#python Main
if __name__ == "__main__":
    app.run(debug=True)