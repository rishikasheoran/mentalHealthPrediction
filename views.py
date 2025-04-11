from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login as auth_login,logout
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from django.core.files.storage import FileSystemStorage
import os
from django.conf import settings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from django.contrib import messages



def signup(request):
    errors = {}  

    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if User.objects.filter(username=uname).exists():
            errors['username'] = "Username already taken"

        if User.objects.filter(email=email).exists():
            errors['email'] = "Email already in use"

        if pass1 != pass2:
            errors['password'] = "Passwords do not match"

        if errors:  
            return render(request, 'signup.html', {'errors': errors})

        else:
            my_user = User.objects.create_user(username=uname, email=email, password=pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'signup.html')

def login_view(request):
    errors = {}  

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        if not username:
            errors['username'] = "Username is required."
        if not password:
            errors['password'] = "Password is required."

        if not errors:
            if not User.objects.filter(username=username).exists():
                errors['username'] = "User does not exist."
            else:
                user = authenticate(request, username=username, password=password)

                if user is not None:
                    auth_login(request, user)
                    
                    request.session['username'] = user.username 
                    request.session['email'] = user.email 
                    print( 'you are :', request.session.get('email'))

                    return redirect('home')  
                else:
                    errors['login'] = "Invalid password."

    return render(request, 'login.html', {'errors': errors})




@login_required(login_url='login')
def homepage(request):
  
    return render(request, "home.html")



def logout_view(request):
    logout(request)
    request.session.flush() 
    return redirect('login')


def about(request):
    return render(request,"about.html")



@login_required(login_url='login')
def upload_file(request):
    if request.method == "POST" and request.FILES.get("uploaded_file"):
        uploaded_file = request.FILES["uploaded_file"]
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)

        with open(file_path, "wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        request.session["dataset_path"] = file_path  
        print("File uploaded successfully, redirecting to train_model...")  
        
        return redirect("train_model")  

    print("No file uploaded, redirecting to home...")  
    return redirect("home")


def train_model(dataset_path, model_output_path, scaler_output_path):
    try:
        print(f" Loading dataset from: {dataset_path}")

        if not os.path.exists(dataset_path):
            return "Dataset file not found!"

        data = pd.read_csv(dataset_path)

        if data.empty:
            return "Dataset is empty or corrupted!"

        print(f" Dataset loaded successfully with shape: {data.shape}")

        required_columns = [
            "age", "gender", "occupation", "sleep_duration", "screen_time",
            "stress_level", "mood_swings", "physical_activity",
            "anxiety_symptoms", "depressive_feelings", "suicidal_thoughts", 
            "family_history", "mental_health_condition", "happiness_level", "concentration",
            "headaches", "social_interaction", "work_hours"
        ]

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return f" Missing columns in dataset: {', '.join(missing_columns)}"

        print(f" Dataset contains all required columns.")

        if not pd.api.types.is_numeric_dtype(data["gender"]):
            print("ℹ Converting gender column to numeric.")
            data["gender"] = data["gender"].map({"Male": 1, "Female": 0, "Other": 2}).fillna(0)

        if not pd.api.types.is_numeric_dtype(data["occupation"]):
            print("ℹ Converting occupation column to numeric.")
            data["occupation"] = data["occupation"].map({"Student": 0, "Employee": 1, "Self-employed": 2, "Unemployed": 3}).fillna(0)

        if not pd.api.types.is_numeric_dtype(data["physical_activity"]):
            print("ℹ Converting physical_activity column to numeric.")
            data["physical_activity"] = data["physical_activity"].map({"Low": 0, "Moderate": 1, "High": 2}).fillna(0)

    
        X = data.drop(columns=["mental_health_condition"])
        y = data["mental_health_condition"]

        print(f" Feature matrix X shape: {X.shape}")
        print(f" Target variable y shape: {y.shape}")

        X.fillna(X.mean(), inplace=True)

   
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f" Split into train ({X_train.shape}) and test ({X_test.shape}) sets.")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(" Applied feature scaling.")


        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        model.fit(X_train_scaled, y_train)

        print(" Model training completed.")

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred) * 100


        joblib.dump(model, model_output_path)
        joblib.dump(scaler, scaler_output_path)

        print(f" Model saved at {model_output_path}")
        print(f" Scaler saved at {scaler_output_path}")

        return accuracy  

    except Exception as e:
        return f" Training Error: {str(e)}"



def train_model_view(request):
    dataset_path = request.session.get("dataset_path")
    print(f"Dataset path in session: {dataset_path}")

    if not dataset_path or not os.path.exists(dataset_path):
        messages.error(request, "Dataset not found! Please upload a valid file.")
        return redirect("upload_file")

    model_output_path = os.path.join(settings.MEDIA_ROOT, "mental_health_model.pkl")
    scaler_output_path = os.path.join(settings.MEDIA_ROOT, "scaler.pkl")

    result = train_model(dataset_path, model_output_path, scaler_output_path)

    if isinstance(result, str):  
        return render(request, "train_model.html", {"error": result})  

    return render(request, "train_model.html", {"success": True})  


def predict(request):
    print(" Debug: predict() function called.")  

    if request.method == "POST":
        try:
            print(" Debug: Form submitted with POST request.")  
            
            
            username = request.session.get("username", "User") 

            model_path = os.path.join(settings.MEDIA_ROOT, "mental_health_model.pkl")
            scaler_path = os.path.join(settings.MEDIA_ROOT, "scaler.pkl")

            print(f" Debug: Looking for model at {model_path}")
            print(f" Debug: Looking for scaler at {scaler_path}")

            if not os.path.exists(model_path):
                print(f" Error: Model file not found at {model_path}")
                return render(request, "prediction_form.html", {"error": "Model file not found!"})
            
            if not os.path.exists(scaler_path):
                print(f" Error: Scaler file not found at {scaler_path}")
                return render(request, "prediction_form.html", {"error": "Scaler file not found!"})

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            print("Debug: Model and scaler loaded successfully.")  

            form_data = [
                request.POST.get("age"),
                request.POST.get("gender"),
                request.POST.get("occupation"),
                request.POST.get("sleep_duration"),
                request.POST.get("physical_activity"),
                request.POST.get("screen_time"),
                request.POST.get("stress_level"),
                request.POST.get("mood_swings"),
                request.POST.get("concentration"),
                request.POST.get("happiness_level"),
                request.POST.get("social_interaction"),
                request.POST.get("headaches"),
                request.POST.get("work_hours"),
                request.POST.get("anxiety_symptoms"),
                request.POST.get("depressive_feelings"),
                request.POST.get("suicidal_thoughts"),
                request.POST.get("family_history"),
            ]

            print(" Debug: Form Data Extracted ->", form_data)  
            
            try:
                form_data = [float(value) for value in form_data]
                print(" Debug: Form Data Converted to Float ->", form_data)  
            except ValueError as e:
                print(f" Error converting data: {e}")
                return render(request, "prediction_form.html", {"error": "Invalid input format! Ensure all values are numbers."})

            try:
                form_data = np.array(form_data).reshape(1, -1)
                scaled_input = scaler.transform(form_data)
            except Exception as e:
                print(f" Error scaling input: {e}")
                return render(request, "prediction_form.html", {"error": "Error processing input!"})

            try:
                prediction = model.predict(scaled_input)[0]
                print(" Debug: Prediction made successfully ->", prediction)  
            except Exception as e:
                print(f" Error making prediction: {e}")
                return render(request, "prediction_form.html", {"error": "Prediction error!"})

            risk_levels = {
                0: "Low Mental Health Risk 😊",
                1: "Moderate Mental Health Risk 😐",
                2: "High Mental Health Risk 😞"
            }

            advice_text = {
                0: "You're doing great! Keep maintaining a healthy lifestyle.",
                1: "Take care of your mental well-being. Engage in stress-reducing activities.",
                2: "Consider seeking professional help. Your mental health is important!"
            }

            health_status = risk_levels.get(prediction, "Unknown Risk Level")
            advice = advice_text.get(prediction, "Stay positive and take care!")

            return render(request, "result.html", {
                "username": username, 
                "prediction": health_status,
                "health_status": health_status,
                "advice": advice
            })

        except Exception as e:
            print(f" General Error: {e}")
            return render(request, "prediction_form.html", {"error": "An unexpected error occurred!"})

    return render(request, "prediction_form.html")