import streamlit as st
from predict import predict_match
from build_func import model_state

st.title("FIFA World Cup 2026 Match Predictor")

team_list = sorted(model_state['elo_ratings'].keys())

col1, col2 = st.columns(2)

with col1:
    team_home = st.selectbox("Home Team", team_list)

with col2:
    team_away = st.selectbox("Away Team", team_list)

neutral = st.checkbox("Neutral venue?", value=True)

if st.button("Predict Match"):
    if team_home == team_away:
        st.error("Please select two different teams.")
    else:
        result = predict_match(team_home, team_away, neutral)
        
        st.subheader(f"Prediction: {result['predicted_result']}")
        
        st.write("Probabilities:")
        for outcome, prob in result['probabilities'].items():
            st.write(f"{outcome}: {prob:.1%}")