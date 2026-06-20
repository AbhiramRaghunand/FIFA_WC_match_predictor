import joblib
from build_func import build_feature

model=joblib.load('C:\ABHIRAM\projects\FifaWCPredictor\models\model.pkl')

def predict_match(team_home,team_away,neutral):
    feature_df=build_feature(team_home,team_away,neutral)

    prediction=model.predict(feature_df)[0]
    prob=model.predict_proba(feature_df)[0]

    classes=model.classes_

    prob_dict=dict(zip(classes,prob))

    return {
        'predicted_result':prediction,
        'probabilities':prob_dict
    }

# predict_match("Brazil","Argentina",neutral=True)
# predict_match("Argentina","Brazil",neutral=True)
# predict_match("Brazil","Haiti",neutral=True)