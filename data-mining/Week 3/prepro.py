import pandas as pd
import numpy as np
import typer
import re
from sklearn.impute import KNNImputer

def fix_date(date):
        try:
            return pd.to_datetime(date, infer_datetime_format=True, dayfirst=True).strftime('%d-%m-%Y')
        except Exception:
            return "Unknown/Invalid Format"
        
def clean_number(phone):
        phone = re.sub(r'\D', '', str(phone))  
        return phone if len(phone) >= 10 and len(phone) <= 12 else "Unknown/Invalid Format" 

blood_type_mapping = {
        "A positive": "A+", "A negative": "A-",
        "B positive": "B+", "B negative": "B-",
        "O positive": "O+", "O negative": "O-",
        "AB positive": "AB+", "AB negative": "AB-"
    }
def convert_blood_type(blood_type):
        return blood_type_mapping.get(blood_type, blood_type)

def preprocess_data(file_path: str):
    df = pd.read_csv(file_path)
    
    #Data Cusdirty
    if 'Purchase_Date' in df.columns:
        df["Purchase_Date"] = df["Purchase_Date"].apply(fix_date)

    if 'Phone_Number' in df.columns:
        df["Phone_Number"] = df["Phone_Number"].apply(clean_number)

    if 'CustomerID' in df.columns:
        df["CustomerID"] = df["ID"].apply(lambda x: f"CUST_{x}")

    if 'Country' in df.columns:
        df['Country'] = df['Country'].apply(
        lambda x: "United Kingdom" if str(x).strip().lower() in ["united kingdom", "uk"] else
                    "United States" if str(x).strip().lower() in ["united states", "usa", "us"] else 
                    "Canada" if str(x).strip().lower() in ["canada", "can"] else x)

    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        imputer = KNNImputer(n_neighbors=5)
        df[['Quantity']] = imputer.fit_transform(df[['Quantity']])
        df['Quantity'] = df['Quantity'].round().astype(int)

    if 'Product_Category' in df.columns:
        df["Product_Category"] = df["Product_Category"].fillna("Unknown/Undifined")
        df['Product_Category'] = df['Product_Category'].apply(
        lambda x: "Electronics" if str(x).strip().lower() in ["electronics", "electornics"] else x)

    #Data Hospital
    if 'Age' in df.columns:    
        df['Age'] = df['Age'].abs()
    if 'Doctor' in df.columns:
        df["Doctor"] = df["Doctor"].fillna("Unknown/Undifined")
    if 'Medication' in df.columns:
        df["Medication"] = df["Medication"].fillna("Unknown/Undifined")
    if 'InsuranceProvider' in df.columns:
        df["InsuranceProvider"] = df["InsuranceProvider"].fillna("Unknown/None")
    if 'DischargeDate' in df.columns:
        df["DischargeDate"] = df["DischargeDate"].fillna("Unknown/Undifined")
        df["DischargeDate"] = df["DischargeDate"].apply(fix_date)
    if 'AdmissionDate' in df.columns:
        df["AdmissionDate"] = df["AdmissionDate"].apply(fix_date)
    if 'AppointmentDate' in df.columns:
        df["AppointmentDate"] = df["AppointmentDate"].apply(fix_date)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].apply(
        lambda x: "Female" if str(x).strip().lower() in ["female", "f"] else
                    "Male" if str(x).strip().lower() in ["male", "m"] else 
                    "Unknown/Undifined" if str(x).strip().lower() in ["unknown", "u"] else x)
    if 'Department' in df.columns:
        df['Department'] = df['Department'].apply(
        lambda x: "Orthopedics" if str(x).strip().lower() in ["orthopedics", "ortho"] else
                    "Endocrinology" if str(x).strip().lower() in ["endocrinology", "endo"] else
                    "Cardiology" if str(x).strip().lower() in ["cardiology", "cardilogy", "cardio"] else x)
    if 'Diagnosis' in df.columns:
        df['Diagnosis'] = df['Diagnosis'].apply(
        lambda x: "Hypertension" if str(x).strip().lower() in ["hypertension", "hypertensyon"] else
                    "COPD" if str(x).strip().lower() in ["copd"] else 
                    "Fracture" if str(x).strip().lower() in ["fracture"] else 
                    "Appendicitis" if str(x).strip().lower() in ["appendicitis", "appendycytys"] else 
                    "Diabetes" if str(x).strip().lower() in ["diabetes", "dyabetes"] else x)
    if 'BloodType' in df.columns:
        df["BloodType"] = df["BloodType"].astype(str).apply(convert_blood_type)

    #Data Government
    if 'Name' in df.columns:
        df['Name'] = df['Name'].str.title()
    if 'BirthDate' in df.columns:
        df['BirthDate'] = df['BirthDate'].apply(fix_date)
    if 'TaxID' in df.columns:
        df['TaxID'] = df['TaxID'].fillna("Unknown/None")
    if 'CriminalRecord' in df.columns:
        df['CriminalRecord'] = df['CriminalRecord'].fillna("Unknown/None")
    if 'ImmigrationYear' in df.columns:
        df['ImmigrationYear'] = df['ImmigrationYear'].fillna("Unknown/None")
    if 'Nationality' in df.columns:
        df['Nationality'] = df['Nationality'].apply(
        lambda x: "France" if str(x).strip().lower() in ["france", "fr"] else
                    "India" if str(x).strip().lower() in ["india", "ind", "bharat"] else 
                    "United States" if str(x).strip().lower() in ["usa", "us", "america", "united states"] else x)
    if 'VotingStatus' in df.columns:
        df['VotingStatus'] = df['VotingStatus'].apply(
        lambda x: "Yes" if str(x).strip().lower() in ["yes", "y", "registered"] else
                    "No" if str(x).strip().lower() in ["no", "n"] else 
                    "Unknown" if str(x).strip().lower() in [""] else "Unknown")
    if 'CriminalRecord' in df.columns:    
        df['CriminalRecord'] = df['CriminalRecord'].apply(
        lambda x: "Yes" if str(x).strip().lower() in ["yes"] else
                    "No" if str(x).strip().lower() in ["no"] else 
                    "Pending" if str(x).strip().lower() in ["pending"] else "Unknown/None")
    if 'MaritalStatus' in df.columns:    
        df['MaritalStatus'] = df['MaritalStatus'].apply(
        lambda x: "Single" if str(x).strip().lower() in ["single", "s"] else x)
    if 'EducationLevel' in df.columns:
        df['EducationLevel'] = df['EducationLevel'].apply(
        lambda x: "High School" if str(x).strip().lower() in ["high school", "hs"] else x)

    output_path = file_path.replace(".csv", "_cleaned.csv")
    df.to_csv(output_path, index=False)
    print(f"Preprocessed file saved as: {output_path}")

if __name__ == "__main__":
    import typer
    typer.run(preprocess_data)
