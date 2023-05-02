import pandas as pd


class DataPreprocessor:
    
    _dictionary = {'New York Giants': 'NYG',
                    'Las Vegas Raiders': 'LVR',
                    'Los Angeles Chargers': 'LAC',
                    'Denver Broncos': 'DEN',
                    'Green Bay Packers': 'GNB',
                    'Jacksonville Jaguars': 'JAX',
                    'Washington Redskins': 'WAS',
                    'Los Angeles Rams': 'LAR',
                    'Arizona Cardinals': 'ARI',
                    'Carolina Panthers': 'CAR',
                    'Baltimore Ravens': 'BAL',
                    'New York Jets': 'NYJ',
                    'Miami Dolphins': 'MIA',
                    'Minnesota Vikings': 'MIN',
                    'Oakland Raiders': 'OAK',
                    'Chicago Bears': 'CHI',
                    'New England Patriots': 'NWE',
                    'Tennessee Titans': 'TEN',
                    'New Orleans Saints': 'NOR',
                    'Cleveland Browns': 'CLE',
                    'Tampa Bay Buccaneers': 'TAM',
                    'Buffalo Bills': 'BUF',
                    'Cincinnati Bengals': 'CIN',
                    'Houston Texans': 'HOU',
                    'San Francisco 49ers': 'SFO',
                    'Atlanta Falcons': 'ATL',
                    'Washington Football Team': 'WAS',
                    'Indianapolis Colts': 'IND',
                    'Seattle Seahawks': 'SEA',
                    'Pittsburgh Steelers': 'PIT',
                    'Dallas Cowboys': 'DAL',
                    'Detroit Lions': 'DET',
                    'Philadelphia Eagles': 'PHI',
                    'Kansas City Chiefs': 'KAN'}
    
    def __init__(self, data):
        self.data = data
        
    def preprocess_data(self):
        #self._print_data_header("Initial Data")
        self._flatten_multiindex_header()
        #self._print_data_header("After Flattening MultiIndex Header")
        self._convert_elements_to_float_or_string()  # Make sure this line is before _calculate_yards_per_play()
        self._handle_missing_values()
        #self._print_data_header("After Handling Missing Values")
        self._calculate_yards_per_play()
        #self._print_data_header("After Calculating Yards per Play")
        self._replace_team_names(self._dictionary)
        #self._print_data_header("After Renaming Columns")

        return self.data
        
    def _flatten_multiindex_header(self):
        if isinstance(self.data.columns, pd.MultiIndex):
            level1 = self.data.columns.get_level_values(0)
            level2 = self.data.columns.get_level_values(1)
            duplicates = level2.value_counts() > 1
            
            self.data.columns = [
                f'{col_level1}_{col_level2}' if duplicates[col_level2] else
                col_level1 if col_level2 == '' else
                col_level2
                for col_level1, col_level2 in zip(level1, level2)
            ]
        return self.data.reset_index(drop=True, inplace=True)
    
    def _convert_elements_to_float_or_string(self):
        for col in self.data.columns:
            for i in range(len(self.data[col])):
                try:
                    self.data.at[i, col] = float(self.data.at[i, col])
                except (ValueError, TypeError):
                    pass

    def _calculate_yards_per_play(self):
        # Calculate yards per play for rushing and receiving
        if 'Rushing_Att' in self.data.columns and 'Rushing_Yds' in self.data.columns:
            self.data['Y/A'] = self.data.apply(lambda x: 0 if (pd.isna(x['Rushing_Att']) or x['Rushing_Att'] <= 0) else x['Rushing_Yds'] / x['Rushing_Att'], axis=1)
        if 'Rec' in self.data.columns and 'Receiving_Yds' in self.data.columns:
            self.data['Y/R'] = self.data.apply(lambda x: 0 if (pd.isna(x['Rec']) or x['Rec'] <= 0) else x['Receiving_Yds'] / x['Rec'], axis=1)
            
    def _handle_missing_values(self, thresh=0.5):
        # Drop rows where most of the columns have string data or NaN
        str_cols = self.data.select_dtypes(include=['object']).columns
        str_counts = self.data[str_cols].apply(lambda x: sum(x.apply(lambda y: isinstance(y, str) or pd.isna(y))), axis=1)
        str_prop = str_counts / len(str_cols)
        self.data = self.data[str_prop <= thresh]

        # Fill applicable missing data
        should_fill_mask = (self.data.isnull() | (self.data == 0)).sum() / len(self.data) >= thresh
        cols_to_fill = should_fill_mask[should_fill_mask == True].index.tolist()
        self.data[cols_to_fill] = self.data[cols_to_fill].fillna(0)

        # Drop rows where most of the columns are null
        self.data.dropna(axis=0, thresh=len(self.data.columns) * thresh, inplace=True)

        # Drop rows where PPR data is null (if present in dataframe)
        if 'PPR' in self.data.columns:
            self.data.dropna(subset=['PPR'], inplace=True)

        # Drop rows where Rk > 275 if Rk is a column
        if 'Rk' in self.data.columns:
            self.data = self.data[self.data['Rk'] <= 275]
        
    def _replace_team_names(self, _dictionary):
        if 'Tm' not in self.data.columns:
            return self.data

        for key, value in _dictionary.items():
            mask = self.data['Tm'].str.startswith(key)
            if mask.any():
                self.data.loc[mask, 'Tm'] = value

        return self.data
    
    def _print_data_header(self, title):
        print(f"Data Header: {title}")
        print(self.data.head())
        
        

def wrapper(args):
    return create_and_evaluate_model(*args)