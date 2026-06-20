# FIFA World Cup 2026 Match Predictor

A machine learning model that predicts the outcome (home win / draw / away win) of international football matches, built on ~25 years of historical results and deployed as an interactive Streamlit app.

Try it: pick any two national teams, and the model returns predicted probabilities for each outcome based on Elo strength, recent form, head-to-head history, and goal-scoring tendencies.

---

## How It Works

### 1. Data

Built on the [`martj42/international_results`](https://github.com/martj42/international_results) dataset — ~49,000 international football matches dating back to 1872. This project filters to **2000–2025** to keep the data representative of the modern game.

### 2. Tournament Tiering

Not all matches are equal. A World Cup match should influence a team's rating far more than a friendly. Every match is bucketed into one of four tiers:

| Tier | Examples |
|---|---|
| 1 | FIFA World Cup, UEFA Euro, Copa América, AFCON, Asian Cup, Gold Cup |
| 2 | World Cup/Euro qualifiers, Nations Leagues |
| 3 | Regional cups (CECAFA, Gulf Cup, AFF Championship, etc.) |
| 4 | Friendlies |

Multi-sport games (Olympics, Asian Games, etc.) and non-FIFA competitions (CONIFA, etc.) are excluded entirely.

### 3. Elo Rating System

A custom Elo implementation, built from scratch and processed strictly in chronological order to avoid lookahead leakage:

- All teams start at a neutral baseline (1500)
- Ratings update after every match using the standard Elo expected-score formula
- The **K-factor** (how much a single result moves the rating) is scaled by:
  - **Tournament tier** — World Cup results move ratings far more than friendlies
  - **Goal margin** — a 4-0 win moves ratings more than a 1-0 win, with diminishing returns for blowouts
- Validated by checking that final ratings match real-world intuition (Spain, Argentina, France, Brazil near the top) and that individual teams' rating trajectories track real tournament performances over time

### 4. Feature Engineering

Beyond Elo (long-run team strength), the model uses features that capture short-term team state:

- **Recent form** — rolling win rate over a team's last 10 matches, regardless of home/away status
- **Head-to-head record** — historical results between the specific two teams in a matchup, tracked with a consistent alphabetical-pair key so it's independent of which team is "home" in any given meeting
- **Goal-scoring tendency** — rolling average goals scored and conceded per team, separate from win/loss form, to capture attacking/defensive style

All features are computed strictly using only information available *before* each match — verified by checking that brand-new teams/pairings default to neutral values (1500 Elo, 0.5 form/h2h) and that values only update after a match's outcome is recorded.

### 5. Modeling

Several models were compared on an identical, **time-based** train/test split (train on pre-2023 matches, test on 2023+) to simulate genuine future prediction:

| Model | Accuracy | Draw Recall | Macro F1 |
|---|---|---|---|
| Logistic Regression | **58.2%** | **0.23** | **0.52** |
| Random Forest | 60.0% | 0.08 | 0.48 |
| Gradient Boosting | 60.3% | 0.00 | 0.43 |
| Gradient Boosting (tuned) | 59.5% | 0.03 | 0.46 |

**Logistic Regression was chosen as the final model**, despite *not* having the highest raw accuracy. The tree-based models achieve higher accuracy by almost entirely ignoring draws and leaning on the majority class (home wins) — Gradient Boosting's draw recall of 0.00 means it essentially never predicts a draw. Logistic Regression, combined with `class_weight='balanced'`, gives the most genuinely useful, balanced performance across all three outcomes.

This is a deliberate lesson from the project: **accuracy alone is a misleading metric under class imbalance**, and the "best" model by raw accuracy is not always the most useful one.

### Final model performance
- **Accuracy: 58.2%** (naive "always predict home win" baseline: ~47%)
- **Draw recall: 0.23** — predicting draws is a known hard problem in football analytics generally, since draws are heavily influenced by in-game tactics and game-state management that no pre-match feature can capture

---

## Known Limitations

- **Tournament stage (group vs. knockout) is not modeled.** The source dataset doesn't track match round/stage, and reliably inferring it would require merging with a separate, stage-labeled dataset. Scoped out as a future improvement.
- **Home/away feature asymmetry.** Swapping which team is "home" in a hypothetical matchup doesn't always produce a perfectly mirrored prediction. This is an inherent property of Logistic Regression learning independent weights per column (`elo_home` vs `elo_away`, etc.) rather than a bug — confirmed by testing that all underlying features swap correctly, and that removing redundant diff features didn't change the behavior.
- **Outlier matches** (e.g. 20+ goal blowouts that occasionally occur in international football) can temporarily skew a team's rolling goal-scoring averages for several matches afterward.
- **No player-level data** — the model only uses team-level historical results, not squad changes, injuries, or individual form.

---

## Project Structure

```
project/
├── data/
    |__ international_results        # raw historical match data
│   └── Processed_result.csv         # Data with elo system
|   |__ Processed_result_v2.csv      # Data with expanded feature set(Recent form, Goal scoring, H2H)
├── models/
│   ├── model.pkl                    # trained Logistic Regression pipeline
│   └── model_state.pkl              # Elo ratings + form/h2h/goal history (for predicting new matches)
|   |__ elo_retings.pkl
├── src/
│   ├── build_func.py                # feature-building functions (Elo, form, h2h, goals)
│   ├── predict.py                   # loads model + state, exposes predict_match()
│   └── app.py                       # Streamlit UI
└── notebooks/                       # data cleaning, Elo validation, feature engineering, model comparison
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Select a home team, an away team, and whether the match is at a neutral venue — the app returns predicted probabilities for home win / draw / away win.

---

## Built With

Python, pandas, scikit-learn, Streamlit, joblib

## Data Source

[martj42/international_results](https://github.com/martj42/international_results) — International football results from 1872 to present.
