import requests
from typing import List
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class DataScraper:
    def __init__(self, years: List[int]):
        self.years = years
        self.player_urls = [f'https://www.pro-football-reference.com/years/{year}/fantasy.htm' for year in years]
        self.team_url = 'https://www.pro-football-reference.com/years/{}/'
        self.session = requests.Session()

    @lru_cache(maxsize=None)
    def scrape_data(self, url: str) -> pd.DataFrame:
        response = self.session.get(url)
        response_text = response.text

        parse_only = SoupStrainer('table')
        soup = BeautifulSoup(response_text, 'lxml', parse_only=parse_only)
        tables = soup.find_all('table')

        df_list = [pd.read_html(str(tables[i]))[0] for i in range(min(len(tables), 2))]
        if df_list:
            year = url.split('/')[-2]  # Extract the year from the URL
            df = pd.concat(df_list)
            df['Year'] = str(year)
            return df

            parse_only = SoupStrainer('table')
            soup = BeautifulSoup(response_text, 'lxml', parse_only=parse_only)
            tables = soup.find_all('table')

    def scrape_player_data(self) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            player_data_frames = list(executor.map(self.scrape_data, self.player_urls))
        return pd.concat([df for df in player_data_frames if df is not None], ignore_index=True)

    def scrape_team_data(self) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            team_urls = [self.team_url.format(year) for year in self.years]
            team_data_frames = list(executor.map(self.scrape_data, team_urls))
        return pd.concat([df for df in team_data_frames if df is not None], ignore_index=True)
