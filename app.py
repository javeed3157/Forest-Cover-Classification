import os.path

from flask import Flask, request, render_template, send_file
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# from pandas_profiling import ProfileReport
import joblib
app = Flask(__name__)

scaler = MinMaxScaler()

@app.route('/', methods = ['GET', 'POST'])
def upload_csv():
    return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
       model = joblib.load("rf_smotekbest.pkl")
       f = request.files['file']
       df = pd.read_csv(f)
       df = df[['Elevation','Slope','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type6','Soil_Type10','Soil_Type12','Soil_Type13','Soil_Type22','Soil_Type23','Soil_Type29','Soil_Type30','Soil_Type38','Soil_Type39','Soil_Type40']]
       scaled_test_data = scaler.fit_transform(df)
       output = model.predict(scaled_test_data)
       output_df = pd.DataFrame(output, columns=['Cover_Type'])
       output_df["Cover_Type"] = output_df["Cover_Type"].map(
           {1: "1-Lodgepole_Pine", 2: "2-Spruce_Fir", 3: "3-Douglas_fir", 4: "4-Krummholz", 5: "5-Ponderosa_Pine", 6: "6-Aspen",
            7: "7-Cottonwood_Willow"})
       df['Cover_Type'] = output_df
       df.to_csv('predictions.csv')
       # print(df2)
       # d = {}
       # for i in output_df['Class']:
       #     if i in d:
       #         d[i] += 1
       #     else:
       #         d[i] = 1
       # print(d)
       return render_template("predict.html")


@app.route("/download", methods=['POST'])
def downloadFile():
    try:
        if(os.path.isfile(r"C:\Users\SHAIK JAVEED SUHAIL\Desktop\forests\Forest-Cover-Classification\predictions.csv")):
            csv_path = r"C:\Users\SHAIK JAVEED SUHAIL\Desktop\forests\Forest-Cover-Classification\predictions.csv"
            csv_file = "predictions.csv"
            return send_file(csv_path,as_attachment=True,attachment_filename=csv_file)
    except Exception:
        return Exception
# main driver function
if __name__ == '__main__':
    app.run(debug=True)