import pandas as pd
import numpy as np
import os

class IPLDataLoader:
    def __init__(self, matches_path: str, deliveries_path: str):
        self.matches_path = matches_path
        self.deliveries_path = deliveries_path
        self.matches_df = None
        self.deliveries_df = None
        
        # Mapping old franchise names to their current active names
        self.team_name_mapping = {
            'Delhi Daredevils': 'Delhi Capitals',
            'Kings XI Punjab': 'Punjab Kings',
            'Deccan Chargers': 'Sunrisers Hyderabad',
            'Rising Pune Supergiant': 'Rising Pune Supergiants', # Standardize typo
            'Pune Warriors': 'Rising Pune Supergiants', # Grouping Pune franchises (optional, but good for ML)
            'Royal Challengers Bangalore': 'Royal Challengers Bengaluru'
        }

    def load_raw_data(self):
        """Loads raw CSV data from the data directory."""
        if os.path.exists(self.matches_path) and os.path.exists(self.deliveries_path):
            self.matches_df = pd.read_csv(self.matches_path)
            self.deliveries_df = pd.read_csv(self.deliveries_path)
            self.clean_data() # Automatically clean upon loading
            return True
        return False

    def clean_data(self):
        """Standardizes team names and filters out incomplete matches."""
        if self.matches_df is None or self.deliveries_df is None:
            return

        # 1. Standardize Team Names in Matches dataset
        for col in ['team1', 'team2', 'toss_winner', 'winner']:
            if col in self.matches_df.columns:
                self.matches_df[col] = self.matches_df[col].replace(self.team_name_mapping)

        # 2. Standardize Team Names in Deliveries dataset
        for col in ['batting_team', 'bowling_team']:
            if col in self.deliveries_df.columns:
                self.deliveries_df[col] = self.deliveries_df[col].replace(self.team_name_mapping)

        # 3. Filter out 'No Result' or rain-abandoned matches
        # We only want matches where a clear winner was decided normally or by DLS
        self.matches_df = self.matches_df[self.matches_df['result_type'] != 'no result']

        # 4. Handle missing cities/venues
        self.matches_df['venue'] = self.matches_df['venue'].fillna('Unknown Venue')

    def get_matches(self):
        return self.matches_df

    def get_deliveries(self):
        return self.deliveries_df
    def prepare_ml_data(self):
        """Transforms raw ball-by-ball data into ML features for 2nd Innings Chases."""
        if self.matches_df is None or self.deliveries_df is None:
            return None

        # 1. Calculate the Target Score (1st Innings total + 1)
        total_score_df = self.deliveries_df.groupby(['match_id', 'innings'])['total_runs'].sum().reset_index()
        first_innings = total_score_df[total_score_df['innings'] == 1].copy()
        first_innings['target'] = first_innings['total_runs'] + 1
        first_innings = first_innings[['match_id', 'target']]

        # 2. Merge target and match details (Venue, Winner)
        match_info = self.matches_df[['match_id', 'venue', 'winner']]
        match_df = match_info.merge(first_innings, on='match_id')

        # 3. Filter deliveries for 2nd Innings only
        chase_df = self.deliveries_df[self.deliveries_df['innings'] == 2].copy()
        chase_df = chase_df.merge(match_df, on='match_id')

        # 4. Feature Engineering: Match Situation
        # Calculate current score
        chase_df['current_score'] = chase_df.groupby('match_id')['total_runs'].cumsum()
        chase_df['runs_left'] = chase_df['target'] - chase_df['current_score']
        
        # Calculate balls bowled and balls left
        # Note: 'over' is 0-indexed, 'ball' is 1-indexed
        chase_df['balls_bowled'] = (chase_df['over'] * 6) + chase_df['ball']
        chase_df['balls_left'] = 120 - chase_df['balls_bowled']
        chase_df['balls_left'] = chase_df['balls_left'].apply(lambda x: 0 if x < 0 else x) # Handle extras going over 120 balls

        # Calculate wickets left
        chase_df['is_wicket'] = chase_df['is_wicket'].fillna(0)
        chase_df['cumulative_wickets'] = chase_df.groupby('match_id')['is_wicket'].cumsum()
        chase_df['wickets_left'] = 10 - chase_df['cumulative_wickets']

        # Calculate Run Rates
        chase_df['crr'] = (chase_df['current_score'] * 6) / chase_df['balls_bowled']
        # Prevent division by zero for RRR
        chase_df['rrr'] = np.where(chase_df['balls_left'] > 0, (chase_df['runs_left'] * 6) / chase_df['balls_left'], 0)

        # 5. Define the Target Variable (1 = Batting Team Wins, 0 = Bowling Team Wins)
        def result(row):
            return 1 if row['batting_team'] == row['winner'] else 0
            
        chase_df['result'] = chase_df.apply(result, axis=1)

        # 6. Final cleanup: Keep only the columns the ML model needs
        final_df = chase_df[['batting_team', 'bowling_team', 'venue', 'runs_left', 'balls_left', 'wickets_left', 'target', 'crr', 'rrr', 'result']]
        
        # Drop nulls and edge cases (like 0 balls left to avoid infinity errors)
        final_df = final_df.dropna()
        final_df = final_df[final_df['balls_left'] != 0]

        return final_df