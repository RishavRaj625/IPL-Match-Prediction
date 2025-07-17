import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import feature modules
from Code.player_analysis import PlayerAnalysis
from Code.match_prediction import MatchPrediction
from Code.team_comparison import TeamComparison
from Code.venue_analysis import VenueAnalysis
from Code.player_comparison import PlayerComparison
from Code.profile_comparison import ProfileComparison
from Code.form_tracker import FormTracker

# Further added soon
# from powerplay_analysis import PowerplayAnalysis
# from top_performers import TopPerformers

# Page configuration
st.set_page_config(
    page_title="IPL Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
    }
    .player-profile-card {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class IPLDashboard:
    def __init__(self):
        self.load_data()
        self.initialize_modules()
    
    def load_data(self):
        """Load IPL datasets"""
        try:
            # Load your IPL datasets here
            self.matches_df_raw = pd.read_csv('CSV/matches.csv')  # Match data
            self.deliveries_df_raw = pd.read_csv('CSV/deliveries.csv')  # Ball-by-ball data
            self.players_details_df = pd.read_csv('CSV/2024_players_details.csv')  # Player details
            
            # Map column names to expected format
            self.map_column_names()
            
            # Data preprocessing
            self.preprocess_data()
            
        except FileNotFoundError as e:
            st.error(f"⚠️ Data files not found: {e}")
            st.info("📝 For demo purposes, creating sample data...")
            self.create_sample_data()
    
    def map_column_names(self):
        """Map actual column names to expected column names"""
        
        # Column mapping for matches.csv
        matches_column_mapping = {
            'match_number': 'id',
            'match_date': 'date',
            'toss_winner': 'toss_winner',
            'toss_decision': 'toss_decision',
            'winner': 'winner',
            'team1': 'team1',
            'team2': 'team2',
            'venue': 'venue',
            'city': 'city',
            'player_of_match': 'player_of_match'
        }
        
        # Column mapping for deliveries.csv
        deliveries_column_mapping = {
            'ID': 'match_id',
            'Innings': 'inning',
            'Overs': 'over',
            'BallNumber': 'ball',
            'Batter': 'batsman',
            'Bowler': 'bowler',
            'NonStriker': 'non_striker',
            'ExtraType': 'extra_type',
            'BatsmanRun': 'batsman_runs',
            'ExtrasRun': 'extras',
            'TotalRun': 'runs_scored',
            'IsWicketDelivery': 'is_wicket',
            'PlayerOut': 'player_dismissed',
            'Kind': 'dismissal_kind',
            'FieldersInvolved': 'fielders_involved',
            'BattingTeam': 'batting_team'
        }
        
        # Player details column mapping (keeping original names as they seem standard)
        players_column_mapping = {
            'ID': 'player_id',
            'Name': 'name',
            'longName': 'long_name',
            'battingName': 'batting_name',
            'fieldingName': 'fielding_name',
            'imgUrl': 'image_url',
            'dob': 'date_of_birth',
            'battingStyles': 'batting_style',
            'longBattingStyles': 'long_batting_style',
            'bowlingStyles': 'bowling_style',
            'longBowlingStyles': 'long_bowling_style',
            'playingRoles': 'playing_role',
            'espn_url': 'espn_url'
        }
        
        # Create mapped dataframes
        self.matches_df = self.matches_df_raw.copy()
        self.deliveries_df = self.deliveries_df_raw.copy()
        
        # Rename columns in matches dataframe
        available_matches_cols = {k: v for k, v in matches_column_mapping.items() 
                                 if k in self.matches_df.columns}
        self.matches_df = self.matches_df.rename(columns=available_matches_cols)
        
        # Rename columns in deliveries dataframe
        available_deliveries_cols = {k: v for k, v in deliveries_column_mapping.items() 
                                   if k in self.deliveries_df.columns}
        self.deliveries_df = self.deliveries_df.rename(columns=available_deliveries_cols)
        
        # Rename columns in players details dataframe
        available_players_cols = {k: v for k, v in players_column_mapping.items() 
                                if k in self.players_details_df.columns}
        self.players_details_df = self.players_details_df.rename(columns=available_players_cols)
        
        # Add missing columns with default values if needed
        self.add_missing_columns()
    
    def add_missing_columns(self):
        """Add missing columns with default values"""
        
        # For matches dataframe
        if 'season' not in self.matches_df.columns:
            # Extract season from date or match_number
            if 'date' in self.matches_df.columns:
                self.matches_df['date'] = pd.to_datetime(self.matches_df['date'], errors='coerce')
                self.matches_df['season'] = self.matches_df['date'].dt.year
            else:
                # Default season based on match number ranges (you may need to adjust this)
                self.matches_df['season'] = 2023
        
        # For deliveries dataframe
        if 'bowling_team' not in self.deliveries_df.columns:
            # You'll need to determine bowling team based on batting team and match info
            # This is a placeholder - you may need to implement proper logic
            self.deliveries_df['bowling_team'] = 'Unknown'
        
        # Convert data types
        if 'runs_scored' in self.deliveries_df.columns:
            self.deliveries_df['runs_scored'] = pd.to_numeric(self.deliveries_df['runs_scored'], errors='coerce').fillna(0)
        
        if 'extras' in self.deliveries_df.columns:
            self.deliveries_df['extras'] = pd.to_numeric(self.deliveries_df['extras'], errors='coerce').fillna(0)
        
        # Process player details
        if 'date_of_birth' in self.players_details_df.columns:
            self.players_details_df['date_of_birth'] = pd.to_datetime(self.players_details_df['date_of_birth'], errors='coerce')
            # Calculate age
            self.players_details_df['age'] = (datetime.now() - self.players_details_df['date_of_birth']).dt.days / 365.25
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        # Sample matches data
        teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
                'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
                'Rajasthan Royals', 'Sunrisers Hyderabad']
        
        venues = ['Wankhede Stadium', 'M. A. Chidambaram Stadium', 'M. Chinnaswamy Stadium',
                 'Eden Gardens', 'Arun Jaitley Stadium', 'PCA Stadium', 
                 'Sawai Mansingh Stadium', 'Rajiv Gandhi International Stadium']
        
        # Create sample matches
        np.random.seed(42)
        n_matches = 100
        
        self.matches_df = pd.DataFrame({
            'id': range(1, n_matches + 1),
            'season': np.random.choice([2020, 2021, 2022, 2023, 2024], n_matches),
            'team1': np.random.choice(teams, n_matches),
            'team2': np.random.choice(teams, n_matches),
            'venue': np.random.choice(venues, n_matches),
            'date': pd.date_range('2020-01-01', periods=n_matches, freq='5D'),
            'winner': np.random.choice(teams, n_matches),
            'win_by_runs': np.random.choice([0, 5, 10, 15, 20, 25, 30], n_matches),
            'win_by_wickets': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_matches),
            'toss_winner': np.random.choice(teams, n_matches),
            'toss_decision': np.random.choice(['bat', 'field'], n_matches)
        })
        
        # Create sample deliveries data
        players = ['V Kohli', 'MS Dhoni', 'R Sharma', 'KL Rahul', 'AB de Villiers', 
                  'D Warner', 'S Dhawan', 'H Pandya', 'J Bumrah', 'R Ashwin']
        
        n_deliveries = 2000
        self.deliveries_df = pd.DataFrame({
            'match_id': np.random.choice(range(1, n_matches + 1), n_deliveries),
            'inning': np.random.choice([1, 2], n_deliveries),
            'batting_team': np.random.choice(teams, n_deliveries),
            'bowling_team': np.random.choice(teams, n_deliveries),
            'over': np.random.choice(range(1, 21), n_deliveries),
            'ball': np.random.choice(range(1, 7), n_deliveries),
            'batsman': np.random.choice(players, n_deliveries),
            'bowler': np.random.choice(players, n_deliveries),
            'runs_scored': np.random.choice([0, 1, 2, 3, 4, 6], n_deliveries),
            'player_dismissed': np.random.choice([None] * 80 + players, n_deliveries),
            'dismissal_kind': np.random.choice(['caught', 'bowled', 'lbw', 'run out', None], n_deliveries),
            'extras': np.random.choice([0, 1, 2, 4], n_deliveries)
        })
        
        # Create sample player details data
        batting_styles = ['Right-hand bat', 'Left-hand bat']
        bowling_styles = ['Right-arm fast', 'Right-arm medium', 'Left-arm fast', 'Right-arm off-spin', 
                         'Left-arm orthodox', 'Right-arm leg-spin', 'Slow left-arm orthodox']
        playing_roles = ['Batsman', 'Bowler', 'All-rounder', 'Wicket-keeper batsman']
        
        self.players_details_df = pd.DataFrame({
            'player_id': range(1, len(players) + 1),
            'name': players,
            'long_name': [f"{name} Full Name" for name in players],
            'batting_name': players,
            'fielding_name': players,
            'image_url': [f"https://example.com/player_{i}.jpg" for i in range(len(players))],
            'date_of_birth': pd.date_range('1985-01-01', periods=len(players), freq='100D'),
            'batting_style': np.random.choice(batting_styles, len(players)),
            'long_batting_style': np.random.choice(batting_styles, len(players)),
            'bowling_style': np.random.choice(bowling_styles, len(players)),
            'long_bowling_style': np.random.choice(bowling_styles, len(players)),
            'playing_role': np.random.choice(playing_roles, len(players)),
            'espn_url': [f"https://espn.com/player/{name.replace(' ', '-').lower()}" for name in players]
        })
        
        # Calculate age for sample data
        self.players_details_df['age'] = (datetime.now() - self.players_details_df['date_of_birth']).dt.days / 365.25
        
        self.preprocess_data()
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        # Convert date column
        if 'date' in self.matches_df.columns:
            self.matches_df['date'] = pd.to_datetime(self.matches_df['date'], errors='coerce')
        
        # Add derived columns
        if 'date' in self.matches_df.columns:
            self.matches_df['match_year'] = self.matches_df['date'].dt.year
        elif 'season' in self.matches_df.columns:
            self.matches_df['match_year'] = self.matches_df['season']
        else:
            self.matches_df['match_year'] = 2023
        
        # Merge match and delivery data
        try:
            self.merged_df = self.deliveries_df.merge(
                self.matches_df[['id', 'season', 'date', 'venue', 'winner']], 
                left_on='match_id', 
                right_on='id', 
                how='left'
            )
        except KeyError as e:
            st.error(f"Error merging data: {e}")
            st.info("Available columns in matches_df: " + str(list(self.matches_df.columns)))
            st.info("Available columns in deliveries_df: " + str(list(self.deliveries_df.columns)))
    
    def initialize_modules(self):
        """Initialize all feature modules"""
        try:
            
            self.player_analysis = PlayerAnalysis(self.matches_df, self.deliveries_df)
            self.profile_comparison = ProfileComparison(self.matches_df, self.deliveries_df, self.players_details_df)
            self.player_comparison = PlayerComparison(self.matches_df, self.deliveries_df)
            self.match_prediction = MatchPrediction(self.matches_df, self.deliveries_df)
            self.team_comparison = TeamComparison(self.matches_df, self.deliveries_df)
            self.venue_analysis = VenueAnalysis(self.matches_df, self.deliveries_df)
            self.form_tracker = FormTracker(self.matches_df, self.deliveries_df)
            
        except Exception as e:
            st.error(f"Error initializing modules: {e}")
            # Create dummy modules
            self.player_analysis = None
            self.match_prediction = None
            self.profile_comparison = None
    
    def show_overview(self):
        """Display dashboard overview"""
        st.markdown("<h1 class='main-header'>🏏 IPL Analytics Dashboard</h1>", unsafe_allow_html=True)
        
        # Welcome message
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin: 20px 0; color: white;'>
            <h2>Welcome to the Ultimate IPL Analytics Experience!</h2>
            <p style='font-size: 18px; margin-top: 10px;'>
                Dive deep into Indian Premier League statistics, player performances, and match insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("### 📊 Database Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Matches", len(self.matches_df))
        
        with col2:
            st.metric("Total Deliveries", len(self.deliveries_df))
        
        with col3:
            st.metric("Players in Database", len(self.players_details_df))
        
        with col4:
            try:
                unique_players = len(set(self.deliveries_df['batsman'].dropna().unique().tolist() + 
                                       self.deliveries_df['bowler'].dropna().unique().tolist()))
                st.metric("Active Players", unique_players)
            except KeyError:
                st.metric("Active Players", "N/A")
        
        with col5:
            if 'season' in self.matches_df.columns:
                st.metric("Seasons", len(self.matches_df['season'].unique()))
            else:
                st.metric("Seasons", "N/A")
        
        # Features overview
        st.markdown("### 🎯 What You Can Explore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='feature-card'>
                <h4>📋 Player Profiles</h4>
                <p>Explore detailed player information including batting styles, bowling styles, playing roles, 
                and career statistics. Get comprehensive insights into your favorite players.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='feature-card'>
                <h4>🆚 Player Comparison</h4>
                <p>Compare two players side by side with detailed statistics including runs, wickets, 
                strike rates, and performance metrics across different seasons.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='feature-card'>
                <h4>🏟️ Venue Analysis</h4>
                <p>Discover how different venues affect match outcomes, batting averages, and bowling figures. 
                Understand the impact of home advantage and pitch conditions.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-card'>
                <h4>👤 Player Analysis</h4>
                <p>Deep dive into individual player performance with detailed batting and bowling statistics, 
                career progression, and match-winning contributions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='feature-card'>
                <h4>⚔️ Team vs Team</h4>
                <p>Analyze head-to-head records between teams, win-loss ratios, and performance trends. 
                Discover which teams dominate specific matchups.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='feature-card'>
                <h4>📈 Form Tracker</h4>
                <p>Track recent form of players and teams with performance trends over the last few matches. 
                Identify who's hot and who's struggling.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # How to use section
        st.markdown("### 🚀 How to Use This Dashboard")
        
        st.markdown("""
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #dee2e6;'>
            <h4 style='color: #495057; margin-bottom: 20px;'>Follow these simple steps to get started:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Step-by-step guide using columns
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("""
            <div style='background: #FF6B35; color: white; width: 40px; height: 40px; border-radius: 50%; 
                       display: flex; align-items: center; justify-content: center; margin: 10px auto;
                       font-size: 20px; font-weight: bold;'>1</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**Navigate:** Use the sidebar to select different analysis sections")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("""
            <div style='background: #FF6B35; color: white; width: 40px; height: 40px; border-radius: 50%; 
                       display: flex; align-items: center; justify-content: center; margin: 10px auto;
                       font-size: 20px; font-weight: bold;'>2</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**Filter:** Use dropdown menus and filters to customize your analysis")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("""
            <div style='background: #FF6B35; color: white; width: 40px; height: 40px; border-radius: 50%; 
                       display: flex; align-items: center; justify-content: center; margin: 10px auto;
                       font-size: 20px; font-weight: bold;'>3</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**Analyze:** View interactive charts and detailed statistics")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown("""
            <div style='background: #FF6B35; color: white; width: 40px; height: 40px; border-radius: 50%; 
                       display: flex; align-items: center; justify-content: center; margin: 10px auto;
                       font-size: 20px; font-weight: bold;'>4</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("**Insights:** Use the analysis to make informed decisions about players and teams")
        
        # Quick facts section
        st.markdown("### 💡 Quick Facts About This Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style='background: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid #FF6B35; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;'>
                <h4 style='color: #FF6B35; margin-bottom: 10px;'>🎯 Comprehensive Coverage</h4>
                <p style='color: #333; font-size: 14px; line-height: 1.5;'>
                    Covers all IPL seasons with detailed ball-by-ball data and player information
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid #FF6B35; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;'>
                <h4 style='color: #FF6B35; margin-bottom: 10px;'>📊 Interactive Visualizations</h4>
                <p style='color: #333; font-size: 14px; line-height: 1.5;'>
                    Dynamic charts and graphs that respond to your selections and filters
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background: #ffffff; padding: 20px; border-radius: 10px; border: 2px solid #FF6B35; 
                       box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0;'>
                <h4 style='color: #FF6B35; margin-bottom: 10px;'>🔍 Deep Analytics</h4>
                <p style='color: #333; font-size: 14px; line-height: 1.5;'>
                    Advanced statistics and insights beyond basic scorecards
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting started section
        st.markdown("### 🎉 Ready to Start?")
        
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
                    border-radius: 10px; margin: 20px 0; color: white;'>
            <h3>Choose any section from the sidebar to begin your cricket analytics journey!</h3>
            <p style='font-size: 16px; margin-top: 10px;'>
                Whether you're a fantasy cricket player, cricket analyst, or just a passionate fan, 
                this dashboard has everything you need to understand the game better.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Debug information (collapsible)
        with st.expander("🔧 Technical Information", expanded=False):
            st.write("Matches DataFrame columns:", list(self.matches_df.columns))
            st.write("Deliveries DataFrame columns:", list(self.deliveries_df.columns))
            st.write("Player Details DataFrame columns:", list(self.players_details_df.columns))
            st.write("Matches DataFrame shape:", self.matches_df.shape)
            st.write("Deliveries DataFrame shape:", self.deliveries_df.shape)
            st.write("Player Details DataFrame shape:", self.players_details_df.shape)
    
    def show_player_analysis(self):
        """Show player analysis or placeholder"""
        if self.player_analysis is not None:
            self.player_analysis.show_analysis()
        else:
            st.write("Player analysis module not available. Please check your data and modules.")
    
    def show_match_prediction(self):
        """Show match prediction or placeholder"""
        if self.match_prediction is not None:
            self.match_prediction.show_prediction()
        else:
            st.write("Match prediction module not available. Please check your data and modules.")
    
    def show_profile_comparison(self):
        """Show profile comparison or placeholder"""
        if self.profile_comparison is not None:
            self.profile_comparison.show_comparison()
        else:
            st.write("Profile comparison module not available. Please check your data and modules.")
    
    def run_dashboard(self):
        """Main dashboard runner"""
        # Sidebar navigation
        st.sidebar.title("🏏 Navigation")
        
        pages = {
            "📊 Overview": self.show_overview,
            "📋 Player Profiles": self.show_profile_comparison,
            "👤 Player Analysis": self.show_player_analysis,
            "🆚 Player Comparison": self.player_comparison.show_comparison,
            "🎯 Match Prediction": self.show_match_prediction,
            "⚔️ Team vs Team": self.team_comparison.show_comparison,
            "🏟️ Venue Analysis": self.venue_analysis.show_analysis,
            # "⚡ Powerplay Analysis": self.powerplay_analysis.show_analysis,
            # "🏆 Top Performers": self.top_performers.show_performers,
            "📈 Form Tracker": self.form_tracker.show_tracker
        }
        
        selected_page = st.sidebar.selectbox("Choose Analysis", list(pages.keys()))
        
        # Show selected page
        pages[selected_page]()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>IPL Analytics Dashboard | Built with Streamlit & Plotly | Designed by Rishav</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    dashboard = IPLDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
