
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fastinput')
def fastinput():
    return render_template('fastinput.html')

@app.route('/trysensor')
def trysensor():
    return render_template('trysensor.html')

@app.route('/aboutsensormodel')
def aboutsensormodel():
    return render_template('aboutsensormodel.html')

@app.route('/aboutmanualmodel')
def aboutmanualmodel():
    return render_template('aboutmanualmodel.html')

@app.route('/explanation')
def explanation():
    return render_template('explanation.html')

@app.route('/pos')
def pos():
    return render_template('pos.html')

@app.route('/neg')
def neg():
    return render_template('neg.html')

@app.route('/predict', methods=['POST'])
def predict():
    # load the save model and scaler
    xgb_clf = joblib.load("xgb_clf.save")
    scaler = joblib.load("scaler.save")

    # transform the input text to numpy array
    form_input = request.form.to_dict()
    in_scores = []
    for k,v in form_input.items():
        in_scores.append(int(v))

    total = sum(in_scores)
    in_scores.append(total)
    X_test = [in_scores]
    X_scaled = scaler.transform(X_test)

    pred = xgb_clf.predict(X_scaled)

    if pred[0] == 0:
        return render_template('neg.html')
    else:
        return render_template('pos.html')

    return render_template('err.html')


@app.route('/sensorpred')
def sensorpred():

    # status sensor only record ON/OFF status
    status_sensors = [
        'D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010', 'D011', 'D012', 'D013', 'D014', 'D015', 'D016', 'D017', 'D018', 
        'E002', 'E003', 'F001', 'F002', 
        'I001', 'I002', 'I006', 'I010', 'I011', 'I012', 
        'L001', 'L002', 'L003', 'L004', 'L005', 'L006', 'L007', 'L008', 'L009', 'L010', 'L011', 
        'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010', 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023', 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M036', 'M037', 'M038', 'M039', 'M040', 'M041', 'M042', 'M043', 'M044', 'M045', 'M046', 'M047', 'M048', 'M049', 'M050', 'M051', 'M052', 
        'MA201', 'MA202', 'MA203', 'MA204', 'MA205', 'MA207', 
        'SS001', 'SS002', 'SS003', 'SS004', 'SS005', 'SS006', 'SS007', 'SS008', 'SS009', 'SS010', 'SS011', 'SS012', 'SS015', 'SS016', 'SS017', 'SS018', 'SS019', 'SS020', 'SS021', 
    ]

    # digit sensor record the sensor value of that moment
    digit_sensors = [
        'LL002', 'LL003', 'LL004', 'LL005', 'LL006', 'LL007', 'LL008', 'LL009', 'LL010', 'LL011', 
        'LS001', 'LS002', 'LS003', 'LS004', 'LS005', 'LS006', 'LS007', 'LS008', 'LS009', 'LS010', 'LS011', 'LS012', 'LS013', 'LS014', 'LS015', 'LS016', 'LS017', 'LS018', 'LS019', 'LS020', 'LS021', 'LS022', 'LS023', 'LS024', 'LS025', 'LS026', 'LS027', 'LS028', 'LS029', 'LS030', 'LS031', 'LS032', 'LS033', 'LS034', 'LS035', 'LS036', 'LS037', 'LS038', 'LS039', 'LS042', 'LS043', 'LS044', 'LS045', 'LS046', 'LS047', 'LS048', 'LS049', 'LS050', 'LS051', 'LS201', 'LS202', 'LS203', 'LS204', 'LS205', 'LS206', 'LS207', 
        'P001',  
        'T001', 'T002', 'T003', 'T004', 'T005', 'T101', 'T102', 'T103', 'T104', 'T105', 'T106', 'T107', 'T108', 'T109', 'T110', 'T111'
    ]

    #---------Train Model--------
    df = pd.read_csv("standar.csv")
    y = df['diagnosis']
    X = df.drop(['diagnosis'], axis = 1)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    s_model = scaler.fit(X)
    # X_scaled = s_model.transform(X)

    # output_map = {3: 0, 4: 0, 5: 0, 7: 0, 8: 0, 9: 0, 1: 1, 2: 1, 6: 1, 10: 0}
    # y_relabel = y.map(output_map)
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_relabel, test_size=0.2)

    df2 = pd.read_csv("resample.csv")
    y_resampled = df2['diagnosis']
    X_resampled = df2.drop(['diagnosis'], axis = 1)

    clf = GradientBoostingClassifier(n_estimators = 200, max_depth = 200)
    clf.fit(X_resampled, y_resampled)

    test_data = pd.DataFrame(columns = status_sensors + digit_sensors)

    read_test = pd.read_csv('test.txt', sep=' ', header=None, names=["time", "sensor", "reads", "col04"])
    # get how many sensor appeared in the file
    test_dict = read_test['sensor'].value_counts().to_dict()

    for k, v in test_dict.items():
      if k in digit_sensors:
        # count the sum of the data value
        total_value = read_test.loc[read_test["sensor"] == k, "reads"].astype(float).sum()
        # calculate average value of sensor
        test_dict[k] = round(total_value/v, 4)

    new_row = {}
    for key in status_sensors + digit_sensors:
    # if the sensor appeared in the file, get the number of it appears or the avg
    # otherwise set it to zero
      if key not in test_dict.keys():
        new_row[key] = 0.0
      else:
        new_row[key] = float(test_dict[key])

    # add the data to the dataframe
    test_data = test_data.append(new_row, ignore_index=True)

    test_scaled = s_model.transform(test_data)

    pred = clf.predict(test_scaled)

    if pred[0] == 0:
        return render_template('neg.html')
    else:
        return render_template('pos.html')

    return render_template('err.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
