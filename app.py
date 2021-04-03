from pycaret.classification import *
from imblearn.over_sampling import *
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('deployment_02042021')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_camp = Image.open('campaign.png')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict marketing campaigns succcess')
    st.sidebar.success('https://www.payback.pl')
    
    st.sidebar.image(image_camp)

    st.title("Campaign Success Prediction App")

    if add_selectbox == 'Online':

        input1 = st.number_input('feat_1', min_value=1, max_value=25, value=3)
        input2 = st.number_input('feat_2', min_value=0, max_value=55, value=15)
        input3 = st.number_input('feat_3', min_value=0, max_value=50, value=10)
        input4 = st.number_input('feat_4', min_value=0, max_value=100, value=25)
        input5 = st.selectbox('feat_5', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        input6 = st.number_input('feat_6', min_value=0, max_value=65, value=13)
        input7 = st.number_input('feat_7', min_value=0, max_value=45, value=10)
        input8 = st.number_input('feat_8', min_value=0, max_value=50, value=11)
        input9 = st.number_input('feat_9', min_value=0, max_value=55, value=14)
        input10 = st.number_input('feat_10', min_value=-30, max_value=50, value=3)
        input11 = st.number_input('feat_11', min_value=-40, max_value=35, value=-3)
        input12 = st.number_input('feat_12', min_value=-65, max_value=80, value=0)
        input13 = st.number_input('feat_13', min_value=0, max_value=5, value=3)
        input14 = st.number_input('feat_14', min_value=0, max_value=5, value=3)
        input15 = st.number_input('feat_15', min_value=0, max_value=25, value=10)
        input16 = st.number_input('feat_16', min_value=0, max_value=15, value=5)
        input17 = st.number_input('feat_17', min_value=0, max_value=4, value=1)
        input18 = st.number_input('feat_18', min_value=0, max_value=4, value=1)
        input19 = st.number_input('feat_19', min_value=0, max_value=1, value=0.2)
        input20 = st.number_input('feat_20', min_value=0, max_value=1, value=0.06)

        output=""

        input_dict = {'input1' : feat_1, 'input2' : feat_2, 'input3' : feat_3, 'input4' : feat_4, 'input5' : feat_5, 'input6' : feat_6, 'input7' : feat_7, 'input8' : feat_8, 'input9' : feat_9, 'input10' : feat_10, 'input11' : feat_11, 'input12' : feat_12, 'input13' : feat_13, 'input14' : feat_14, 'input15' : feat_15, 'input16' : feat_16, 'input17' : feat_17, 'input18' : feat_18, 'input19' : feat_19, 'input20' : feat_20}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
