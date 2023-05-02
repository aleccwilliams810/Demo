import pandas as pd
import numpy as np


class MergeAndProcess:
    
    def __init__(self, player_data, team_data):
        self.player_data = player_data
        self.team_data = team_data

    def merge(self):
        # Create a dictionary to map team data to new columns in player data
        team_data_map = {
            col: col if col in self.player_data.columns else f'Team_{col}' 
            for col in self.team_data.columns 
            if col not in ['Year', 'Tm']
        }

        # Create new columns in player data for team data
        self.player_data = self.player_data.assign(**{col: np.nan for col in team_data_map.values()})

        # Map team data to new columns in player data based on team and year
        self.player_data = self.player_data.set_index(['Year', 'Tm'])
        self.team_data = self.team_data.set_index(['Year', 'Tm'])
        self.player_data.update(self.team_data.rename(columns=team_data_map))

        # Flatten the column index
        if isinstance(self.player_data.columns, pd.MultiIndex):
            self.player_data.columns = self.player_data.columns.map(lambda x: x[1] if x[0].startswith('Team_') else x[0])

        return self.player_data.reset_index()
    
    def process(self):
        # Merge player and team data
        merged_data = self.merge()
        
        # Replace non-alphanumeric characters in player names
        merged_data['Player'] = merged_data['Player'].str.replace(r'[^\w\s]+', '')

        # Add next year PPR
        merged_data['next_year_PPR'] = merged_data.groupby('Player')['PPR'].shift(-1)

        # Add PPR per game
        merged_data['PPR_per_game'] = merged_data['PPR'] / merged_data['G']
        merged_data['PPR_per_game'].fillna(np.nan, inplace=True)

        # Drop missing values
        merged_data = merged_data[~((merged_data.isna().any(axis=1) & (merged_data['Year'] < 2022)) | ((merged_data['Year'] == 2022) & merged_data.drop(columns=['next_year_PPR']).isna().any(axis=1)))]

        # Reset index
        merged_data.reset_index(drop=True, inplace=True)

        return merged_data
