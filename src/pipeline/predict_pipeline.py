import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifact","model.pkl")
            preprocessor_path=os.path.join('artifact','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        status: int,
        duration: int,
        credit_history:int,
        purpose: int,
        amount: str,
        savings: int,
        employment_duration: int,
        installment_rate:int,
        personal_status_sex:int,
        other_debtors:int,
        present_residence:int,
        property:int,
        age:int,
        other_installment_plans:int,
        housing:int,
        number_credits:int,
        job:int,
        people_liable:int,
        telephone:int,
        foreign_worker:int):

        self.status = status

        self.duration = duration

        self.credit_history = credit_history

        self.purpose = purpose

        self.amount = amount

        self.savings = savings

        self.employment_duration = employment_duration
        
        self.installment_rate = installment_rate

        self.personal_status_sex = personal_status_sex

        self.other_debtors = other_debtors

        self.present_residence = present_residence

        self.property = property

        self.age = age

        self.other_installment_plans = other_installment_plans

        self.housing = housing

        self.number_credits = number_credits

        self.job = job

        self.people_liable = people_liable

        self.telephone = telephone

        self.foreign_worker = foreign_worker



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "status": [self.status],
                "duration": [self.duration],
                "purpose": [self.purpose],
                "amount": [self.amount],
                "savings": [self.savings],
                "employment_duration": [self.employment_duration],
                "installment_rate": [self.installment_rate],
                "personal_status_sex": [self.personal_status_sex],
                "other_debtors": [self.other_debtors],
                "present_residence": [self.present_residence],
                "property": [self.property],
                "age": [self.age],
                "other_installment_plans": [self.other_installment_plans],
                "housing": [self.housing],
                "number_credits": [self.number_credits],
                "job": [self.job],
                "people_liable": [self.people_liable],
                "telephone": [self.telephone],
                "foreign_worker": [self.foreign_worker],
                "credit_history": [self.credit_history]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)