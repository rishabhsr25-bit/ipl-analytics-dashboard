import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import IPLDataLoader
from src.predictor import IPLPredictor

# Page Setup
st.set_page_config(page_title="IPL Analytics & Prediction Dashboard", layout="wide", page_icon="🏏")

st.title("🏏 IPL Analytics & Win Prediction Dashboard")
st.markdown("---")

# Initialize our backend components
data_loader = IPLDataLoader(matches_path="data/matches.csv", deliveries_path="data/deliveries.csv")
predictor = IPLPredictor()

# Sidebar Setup for Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Historical Analysis", "Live Match Predictor", "Player Insights"])

# Sidebar Data Status Check
st.sidebar.markdown("---")
st.sidebar.subheader("Data Pipeline Status")
if data_loader.load_raw_data():
    st.sidebar.success("Database Connected Successfully!")
    matches_df = data_loader.get_matches()
    deliveries_df = data_loader.get_deliveries()
else:
    st.sidebar.warning("Awaiting CSV datasets in the /data folder.")
    st.stop() # Stops execution if data is missing

# Tab/Page Router Logic
if page == "Historical Analysis":
    st.header("Historical Trends & Team Performance")
    
    # 1. High-Level KPI Metrics Row
    total_matches = matches_df.shape[0]
    total_seasons = matches_df['season'].nunique()
    total_venues = matches_df['venue'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Matches Processed", value=total_matches)
    col2.metric(label="Seasons Tracked", value=total_seasons)
    col3.metric(label="Unique Venues", value=total_venues)
    
    st.markdown("---")
    
    # 2. Visualization Row
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Most Successful IPL Teams")
        # Count wins per team and sort
        win_counts = matches_df['winner'].value_counts().reset_index()
        win_counts.columns = ['Team', 'Wins']
        
        # Create an interactive horizontal bar chart using Plotly
        fig_wins = px.bar(
            win_counts, 
            x='Wins', 
            y='Team', 
            orientation='h',
            color='Wins',
            color_continuous_scale='Viridis',
            labels={'Wins': 'Total Match Wins', 'Team': ''}
        )
        fig_wins.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
        st.plotly_chart(fig_wins, use_container_width=True)
        
    with right_col:
        st.subheader("Toss Decision Impact")
        # Calculate toss choice percentages
        toss_decision = matches_df['toss_decision'].value_counts().reset_index()
        toss_decision.columns = ['Decision', 'Count']
        
        fig_toss = px.pie(
            toss_decision, 
            values='Count', 
            names='Decision', 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_toss.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_toss, use_container_width=True)

elif page == "Live Match Predictor":
    st.header("🔮 Real-Time Win Probability Predictor")
    st.markdown("Enter the current match situation during the 2nd Innings to get live win probabilities.")

    # Dynamically get the list of teams and venues from our dataset
    teams = sorted(matches_df['team1'].unique())
    venues = sorted(matches_df['venue'].unique())

    # Row 1: Team & Venue Selection
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Batting Team (Chasing)", teams, index=teams.index('Chennai Super Kings') if 'Chennai Super Kings' in teams else 0)
    with col2:
        bowling_team = st.selectbox("Bowling Team (Defending)", teams, index=teams.index('Mumbai Indians') if 'Mumbai Indians' in teams else 1)

    venue = st.selectbox("Venue", venues)

    st.markdown("---")
    st.subheader("Current Match Situation")
    
    # Row 2: Match Stats Inputs
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        target = st.number_input("Target Score", min_value=1, max_value=300, step=1, value=180)
    with col4:
        score = st.number_input("Current Score", min_value=0, max_value=300, step=1, value=90)
    with col5:
        # We use decimal for overs (e.g., 10.4 means 10 overs and 4 balls)
        overs = st.number_input("Overs Completed", min_value=0.0, max_value=19.5, step=0.1, value=10.0)
    with col6:
        wickets = st.number_input("Wickets Down", min_value=0, max_value=9, step=1, value=2)

    # Predict Button Logic
    if st.button("Predict Win Probability", use_container_width=True, type="primary"):
        if batting_team == bowling_team:
            st.error("Batting and Bowling teams cannot be the same!")
        else:
            # 1. Calculate the behind-the-scenes ML features
            runs_left = target - score
            
            # Convert decimal overs (like 10.4) to balls bowled
            completed_overs = int(overs)
            balls_in_current_over = int(round((overs - completed_overs) * 10))
            balls_bowled = (completed_overs * 6) + balls_in_current_over
            balls_left = 120 - balls_bowled
            
            wickets_left = 10 - wickets
            
            crr = score / (balls_bowled / 6) if balls_bowled > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            # 2. Package into a DataFrame for the model
            input_data = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'venue': [venue],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets_left': [wickets_left],
                'target': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            # 3. Call the Predictor
            if predictor.load_model():
                probabilities = predictor.predict_probability(input_data)
                
                st.markdown("---")
                st.subheader("Live Win Probability")
                
                # Create a visual probability gauge using columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric(label=f"🏏 {batting_team}", value=f"{probabilities['batting_team_win_prob']}%")
                with res_col2:
                    st.metric(label=f"🎯 {bowling_team}", value=f"{probabilities['bowling_team_win_prob']}%")
                    
            else:
                st.error("Model not found! Please run train_model.py first.")

elif page == "Player Insights":
    st.header("📊 Batsman & Bowler Performance Matrix")
    st.markdown("Search for an individual player to see their all-time IPL statistics.")

    # Get a list of all players (batters and bowlers)
    all_batters = deliveries_df['batter'].dropna().unique().tolist()
    all_bowlers = deliveries_df['bowler'].dropna().unique().tolist()
    all_players = sorted(list(set(all_batters + all_bowlers)))

    # Player Selection
    selected_player = st.selectbox("Search for a Player", all_players, index=all_players.index('V Kohli') if 'V Kohli' in all_players else 0)

    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"🏏 Batting Stats for {selected_player}")
        player_batting = deliveries_df[deliveries_df['batter'] == selected_player]
        
        if not player_batting.empty:
            total_runs = player_batting['batsman_runs'].sum()
            balls_faced = len(player_batting[player_batting['extra_runs'] == 0]) # Exclude wides
            strike_rate = (total_runs / balls_faced) * 100 if balls_faced > 0 else 0
            
            # Calculate dismissals to find average
            dismissals = player_batting[player_batting['player_dismissed'] == selected_player].shape[0]
            average = total_runs / dismissals if dismissals > 0 else total_runs
            
            st.metric(label="Total Runs", value=total_runs)
            st.metric(label="Strike Rate", value=f"{strike_rate:.2f}")
            st.metric(label="Batting Average", value=f"{average:.2f}")
            
            # Show a quick boundaries breakdown
            fours = player_batting[player_batting['batsman_runs'] == 4].shape[0]
            sixes = player_batting[player_batting['batsman_runs'] == 6].shape[0]
            st.caption(f"Boundaries: {fours} Fours | {sixes} Sixes")
        else:
            st.info(f"{selected_player} has no recorded batting stats in this dataset.")

    with col2:
        st.subheader(f"🎯 Bowling Stats for {selected_player}")
        player_bowling = deliveries_df[deliveries_df['bowler'] == selected_player]
        
        if not player_bowling.empty:
            # Wickets (excluding run outs/retired hurts usually not credited to bowler)
            valid_dismissals = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
            wickets_taken = player_bowling[player_bowling['dismissal_kind'].isin(valid_dismissals)].shape[0]
            
            runs_conceded = player_bowling['total_runs'].sum()
            balls_bowled = len(player_bowling[~player_bowling['extra_runs'].isin([1])]) # Simple approx for legal deliveries
            overs_bowled = balls_bowled / 6
            
            economy = runs_conceded / overs_bowled if overs_bowled > 0 else 0
            
            st.metric(label="Total Wickets", value=wickets_taken)
            st.metric(label="Economy Rate", value=f"{economy:.2f}")
            st.metric(label="Runs Conceded", value=runs_conceded)
            
        else:
            st.info(f"{selected_player} has no recorded bowling stats in this dataset.")