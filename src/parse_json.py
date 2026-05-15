import os
import json
import pandas as pd
from tqdm import tqdm  # To show a nice progress bar

def parse_all_matches(json_folder, output_folder):
    match_list = []
    delivery_list = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    print(f"Parsing {len(json_files)} IPL match JSON files...")
    
    for file_name in tqdm(json_files):
        match_id = file_name.split('.')[0]
        file_path = os.path.join(json_folder, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 1. Parse Match-Level Info
        info = data.get('info', {})
        
        # Safely extract basic match metadata
        match_info = {
            'match_id': match_id,
            'season': info.get('season', None),
            'date': info.get('dates', [None])[0],
            'team1': info.get('teams', [None, None])[0],
            'team2': info.get('teams', [None, None])[1],
            'venue': info.get('venue', None),
            'toss_winner': info.get('toss', {}).get('winner', None),
            'toss_decision': info.get('toss', {}).get('decision', None),
            'winner': info.get('outcome', {}).get('winner', None),
            'result_type': 'normal' if 'winner' in info.get('outcome', {}) else list(info.get('outcome', {}).keys())[0] if info.get('outcome') else 'unknown',
            'win_by_runs': info.get('outcome', {}).get('by', {}).get('runs', 0),
            'win_by_wickets': info.get('outcome', {}).get('by', {}).get('wickets', 0),
            'player_of_match': info.get('player_of_match', [None])[0] if info.get('player_of_match') else None
        }
        match_list.append(match_info)
        
        # 2. Parse Ball-by-Ball Innings Data
        innings = data.get('innings', [])
        for inning_idx, inning in enumerate(innings):
            innings_num = inning_idx + 1
            batting_team = inning.get('team')
            bowling_team = match_info['team2'] if batting_team == match_info['team1'] else match_info['team1']
            
            overs_data = inning.get('overs', [])
            for over_data in overs_data:
                over_num = over_data.get('over') # 0-indexed (0 means 1st over)
                deliveries = over_data.get('deliveries', [])
                
                for ball_idx, delivery in enumerate(deliveries):
                    runs = delivery.get('runs', {})
                    wicket = delivery.get('wickets', [{}])[0] if delivery.get('wickets') else {}
                    
                    delivery_info = {
                        'match_id': match_id,
                        'innings': innings_num,
                        'batting_team': batting_team,
                        'bowling_team': bowling_team,
                        'over': over_num,
                        'ball': ball_idx + 1,
                        'batter': delivery.get('batter', delivery.get('batsman')),
                        'bowler': delivery.get('bowler'),
                        'non_striker': delivery.get('non_striker', delivery.get('non_striker')),
                        'batsman_runs': runs.get('batter', runs.get('batsman', 0)),
                        'extra_runs': runs.get('extras', 0),
                        'total_runs': runs.get('total', 0),
                        'is_wicket': 1 if wicket else 0,
                        'player_dismissed': wicket.get('player_out', None),
                        'dismissal_kind': wicket.get('kind', None)
                    }
                    delivery_list.append(delivery_info)

    # Convert to DataFrames and Save
    print("\nSaving compiled data to CSV...")
    df_matches = pd.DataFrame(match_list)
    df_deliveries = pd.DataFrame(delivery_list)
    
    os.makedirs(output_folder, exist_ok=True)
    df_matches.to_csv(os.path.join(output_folder, 'matches.csv'), index=False)
    df_deliveries.to_csv(os.path.join(output_folder, 'deliveries.csv'), index=False)
    
    print(f"Success! Generated:\n - {output_folder}/matches.csv ({df_matches.shape[0]} rows)\n - {output_folder}/deliveries.csv ({df_deliveries.shape[0]} rows)")

if __name__ == "__main__":
    parse_all_matches(json_folder="data/ipl_json", output_folder="data")