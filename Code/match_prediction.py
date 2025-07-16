import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class MatchPrediction:
    def __init__(self, matches_df, deliveries_df):
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.prepare_prediction_data()
        self.train_model()
    
    def prepare_prediction_data(self):
        """Prepare data for match prediction"""
        # Calculate comprehensive team statistics
        self.team_stats = self.calculate_team_stats()
        self.venue_stats = self.calculate_venue_stats()
        
        # Prepare feature matrix
        self.features_df = self.create_features()
    
    def calculate_team_stats(self):
        """Calculate comprehensive team performance statistics"""
        team_stats = {}
        
        # Get all unique teams
        all_teams = set()
        if 'team1' in self.matches_df.columns and 'team2' in self.matches_df.columns:
            all_teams.update(self.matches_df['team1'].unique())
            all_teams.update(self.matches_df['team2'].unique())
        
        for team in all_teams:
            # Matches played
            team1_matches = self.matches_df[self.matches_df['team1'] == team]
            team2_matches = self.matches_df[self.matches_df['team2'] == team]
            total_matches = len(team1_matches) + len(team2_matches)
            
            # Wins
            wins = len(self.matches_df[self.matches_df['winner'] == team])
            win_rate = wins / total_matches if total_matches > 0 else 0
            
            # Batting statistics
            batting_data = self.deliveries_df[self.deliveries_df['batting_team'] == team]
            if len(batting_data) > 0:
                avg_runs_per_match = batting_data.groupby('match_id')['runs_scored'].sum().mean()
                avg_runs_per_over = batting_data['runs_scored'].mean()
                strike_rate = (batting_data['runs_scored'].sum() / len(batting_data)) * 100
            else:
                avg_runs_per_match = 0
                avg_runs_per_over = 0
                strike_rate = 0
            
            # Bowling statistics
            bowling_data = self.deliveries_df[self.deliveries_df['bowling_team'] == team]
            if len(bowling_data) > 0:
                avg_runs_conceded = bowling_data.groupby('match_id')['runs_scored'].sum().mean()
                economy_rate = bowling_data['runs_scored'].mean() * 6  # runs per over
                wickets_per_match = bowling_data.groupby('match_id')['player_dismissed'].apply(
                    lambda x: x.notna().sum()
                ).mean()
            else:
                avg_runs_conceded = 0
                economy_rate = 0
                wickets_per_match = 0
            
            # Head-to-head performance (recent form)
            recent_matches = self.matches_df.tail(20)  # Last 20 matches
            recent_wins = len(recent_matches[recent_matches['winner'] == team])
            recent_form = recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0
            
            team_stats[team] = {
                'team': team,
                'matches_played': total_matches,
                'wins': wins,
                'win_rate': win_rate,
                'avg_runs_per_match': avg_runs_per_match,
                'avg_runs_per_over': avg_runs_per_over,
                'strike_rate': strike_rate,
                'avg_runs_conceded': avg_runs_conceded,
                'economy_rate': economy_rate,
                'wickets_per_match': wickets_per_match,
                'recent_form': recent_form
            }
        
        return pd.DataFrame(list(team_stats.values()))
    
    def calculate_venue_stats(self):
        """Calculate venue-specific statistics"""
        venue_stats = {}
        
        if 'venue' in self.matches_df.columns:
            for venue in self.matches_df['venue'].unique():
                venue_matches = self.matches_df[self.matches_df['venue'] == venue]
                
                # Average scores at venue
                venue_deliveries = self.deliveries_df[
                    self.deliveries_df['match_id'].isin(venue_matches['id'])
                ]
                
                if len(venue_deliveries) > 0:
                    avg_score = venue_deliveries.groupby('match_id')['runs_scored'].sum().mean()
                    high_scoring = 1 if avg_score > 160 else 0
                else:
                    avg_score = 0
                    high_scoring = 0
                
                # Toss advantage
                if 'toss_decision' in self.matches_df.columns and 'toss_winner' in self.matches_df.columns:
                    toss_win_match_win = len(venue_matches[
                        venue_matches['toss_winner'] == venue_matches['winner']
                    ])
                    toss_advantage = toss_win_match_win / len(venue_matches) if len(venue_matches) > 0 else 0.5
                else:
                    toss_advantage = 0.5
                
                venue_stats[venue] = {
                    'venue': venue,
                    'avg_score': avg_score,
                    'high_scoring': high_scoring,
                    'toss_advantage': toss_advantage
                }
        
        return pd.DataFrame(list(venue_stats.values())) if venue_stats else pd.DataFrame()
    
    def get_head_to_head_stats(self, team1, team2):
        """Get head-to-head statistics between two teams"""
        h2h_matches = self.matches_df[
            ((self.matches_df['team1'] == team1) & (self.matches_df['team2'] == team2)) |
            ((self.matches_df['team1'] == team2) & (self.matches_df['team2'] == team1))
        ]
        
        if len(h2h_matches) == 0:
            return 0.5  # Equal probability if no history
        
        team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        team1_h2h_rate = team1_wins / len(h2h_matches)
        
        return team1_h2h_rate
    
    def create_features(self):
        """Create comprehensive feature matrix for prediction"""
        features = []
        
        for _, match in self.matches_df.iterrows():
            if pd.isna(match['winner']):
                continue
                
            team1 = match['team1']
            team2 = match['team2']
            winner = match['winner']
            venue = match.get('venue', 'Unknown')
            
            # Get team statistics
            team1_stats = self.team_stats[self.team_stats['team'] == team1]
            team2_stats = self.team_stats[self.team_stats['team'] == team2]
            
            if len(team1_stats) > 0 and len(team2_stats) > 0:
                team1_stats = team1_stats.iloc[0]
                team2_stats = team2_stats.iloc[0]
                
                # Get venue statistics
                venue_info = self.venue_stats[self.venue_stats['venue'] == venue] if len(self.venue_stats) > 0 else None
                venue_avg_score = venue_info.iloc[0]['avg_score'] if venue_info is not None and len(venue_info) > 0 else 160
                venue_high_scoring = venue_info.iloc[0]['high_scoring'] if venue_info is not None and len(venue_info) > 0 else 0
                venue_toss_advantage = venue_info.iloc[0]['toss_advantage'] if venue_info is not None and len(venue_info) > 0 else 0.5
                
                # Head-to-head
                h2h_rate = self.get_head_to_head_stats(team1, team2)
                
                # Toss factor
                toss_winner = match.get('toss_winner', team1)
                toss_decision = match.get('toss_decision', 'bat')
                toss_factor = 1 if toss_winner == team1 else 0
                batting_first = 1 if toss_decision == 'bat' else 0
                
                feature_row = {
                    # Team 1 features
                    'team1_win_rate': team1_stats['win_rate'],
                    'team1_avg_runs': team1_stats['avg_runs_per_match'],
                    'team1_strike_rate': team1_stats['strike_rate'],
                    'team1_avg_conceded': team1_stats['avg_runs_conceded'],
                    'team1_economy': team1_stats['economy_rate'],
                    'team1_wickets_per_match': team1_stats['wickets_per_match'],
                    'team1_recent_form': team1_stats['recent_form'],
                    
                    # Team 2 features
                    'team2_win_rate': team2_stats['win_rate'],
                    'team2_avg_runs': team2_stats['avg_runs_per_match'],
                    'team2_strike_rate': team2_stats['strike_rate'],
                    'team2_avg_conceded': team2_stats['avg_runs_conceded'],
                    'team2_economy': team2_stats['economy_rate'],
                    'team2_wickets_per_match': team2_stats['wickets_per_match'],
                    'team2_recent_form': team2_stats['recent_form'],
                    
                    # Comparative features
                    'win_rate_diff': team1_stats['win_rate'] - team2_stats['win_rate'],
                    'runs_diff': team1_stats['avg_runs_per_match'] - team2_stats['avg_runs_per_match'],
                    'economy_diff': team2_stats['economy_rate'] - team1_stats['economy_rate'],
                    
                    # Venue features
                    'venue_avg_score': venue_avg_score,
                    'venue_high_scoring': venue_high_scoring,
                    'venue_toss_advantage': venue_toss_advantage,
                    
                    # Match context features
                    'h2h_rate': h2h_rate,
                    'toss_factor': toss_factor,
                    'batting_first': batting_first,
                    
                    # Target and metadata
                    'winner': winner,
                    'team1': team1,
                    'team2': team2,
                    'venue': venue
                }
                features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def train_model(self):
        """Train the prediction model with improved features"""
        if len(self.features_df) == 0:
            st.warning("Not enough data to train the model")
            self.model = None
            self.accuracy = 0
            return
        
        # Define feature columns
        feature_columns = [
            'team1_win_rate', 'team2_win_rate', 'team1_avg_runs', 'team2_avg_runs',
            'team1_strike_rate', 'team2_strike_rate', 'team1_avg_conceded', 'team2_avg_conceded',
            'team1_economy', 'team2_economy', 'team1_wickets_per_match', 'team2_wickets_per_match',
            'team1_recent_form', 'team2_recent_form', 'win_rate_diff', 'runs_diff', 'economy_diff',
            'venue_avg_score', 'venue_high_scoring', 'venue_toss_advantage',
            'h2h_rate', 'toss_factor', 'batting_first'
        ]
        
        # Prepare features and target
        X = self.features_df[feature_columns].fillna(0)
        y = self.features_df['winner']
        
        # Remove rows where winner is not team1 or team2
        valid_indices = []
        for idx, row in self.features_df.iterrows():
            if row['winner'] in [row['team1'], row['team2']]:
                valid_indices.append(idx)
        
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        if len(X) == 0:
            st.warning("No valid training data available")
            self.model = None
            self.accuracy = 0
            return
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store feature columns for prediction
        self.feature_columns = feature_columns
    
    def predict_match(self, team1, team2, venue='Unknown'):
        """Predict match outcome between two specific teams"""
        if self.model is None:
            return None, "Model not trained properly"
        
        try:
            # Get team statistics
            team1_stats = self.team_stats[self.team_stats['team'] == team1]
            team2_stats = self.team_stats[self.team_stats['team'] == team2]
            
            if len(team1_stats) == 0 or len(team2_stats) == 0:
                return None, "Team statistics not available"
            
            team1_stats = team1_stats.iloc[0]
            team2_stats = team2_stats.iloc[0]
            
            # Get venue statistics
            venue_info = self.venue_stats[self.venue_stats['venue'] == venue] if len(self.venue_stats) > 0 else None
            venue_avg_score = venue_info.iloc[0]['avg_score'] if venue_info is not None and len(venue_info) > 0 else 160
            venue_high_scoring = venue_info.iloc[0]['high_scoring'] if venue_info is not None and len(venue_info) > 0 else 0
            venue_toss_advantage = venue_info.iloc[0]['toss_advantage'] if venue_info is not None and len(venue_info) > 0 else 0.5
            
            # Head-to-head
            h2h_rate = self.get_head_to_head_stats(team1, team2)
            
            # Prepare features
            features = np.array([
                team1_stats['win_rate'],
                team2_stats['win_rate'],
                team1_stats['avg_runs_per_match'],
                team2_stats['avg_runs_per_match'],
                team1_stats['strike_rate'],
                team2_stats['strike_rate'],
                team1_stats['avg_runs_conceded'],
                team2_stats['avg_runs_conceded'],
                team1_stats['economy_rate'],
                team2_stats['economy_rate'],
                team1_stats['wickets_per_match'],
                team2_stats['wickets_per_match'],
                team1_stats['recent_form'],
                team2_stats['recent_form'],
                team1_stats['win_rate'] - team2_stats['win_rate'],
                team1_stats['avg_runs_per_match'] - team2_stats['avg_runs_per_match'],
                team2_stats['economy_rate'] - team1_stats['economy_rate'],
                venue_avg_score,
                venue_high_scoring,
                venue_toss_advantage,
                h2h_rate,
                0.5,  # toss_factor (neutral)
                0.5   # batting_first (neutral)
            ]).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Decode prediction
            predicted_winner = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get probability for the two teams only
            team_prob = {}
            for i, team in enumerate(self.label_encoder.classes_):
                if team in [team1, team2]:
                    team_prob[team] = probabilities[i]
            
            # Normalize probabilities to sum to 1 for the two teams
            total_prob = sum(team_prob.values())
            if total_prob > 0:
                team_prob = {team: prob/total_prob for team, prob in team_prob.items()}
            
            return predicted_winner, team_prob
        
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def show_prediction(self):
        """Streamlit UI for predicting match result"""
        st.subheader("🎯 Match Prediction")
        
        if self.model is None:
            st.error("Model not trained properly. Please check your data.")
            return
        
        available_teams = self.team_stats['team'].tolist()
        available_venues = ['Unknown'] + (self.venue_stats['venue'].tolist() if len(self.venue_stats) > 0 else [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            team1 = st.selectbox("Select Team 1", available_teams, key="team1_prediction")
        
        with col2:
            team2 = st.selectbox("Select Team 2", available_teams, key="team2_prediction")
        
        with col3:
            venue = st.selectbox("Select Venue", available_venues, key="venue_prediction")
        
        if st.button("Predict Match Result", type="primary"):
            if team1 != team2:
                predicted_winner, result = self.predict_match(team1, team2, venue)
                
                if predicted_winner:
                    # Display prediction result
                    st.success(f"🏆 Predicted Winner: **{predicted_winner}**")
                    
                    # Show probability with progress bars
                    st.write("### 📊 Win Probabilities:")
                    
                    col1, col2 = st.columns(2)
                    for i, (team, prob) in enumerate(result.items()):
                        with col1 if i == 0 else col2:
                            st.metric(
                                label=team,
                                value=f"{prob:.1%}",
                                delta=f"{prob - 0.5:.1%}" if prob != 0.5 else None
                            )
                            st.progress(prob)
                    
                    # Model performance info
                    st.info(f"📈 Model Accuracy: {self.accuracy:.1%}")
                    
                    # Show team comparison
                    st.write("### ⚖️ Team Comparison")
                    
                    team1_stats = self.team_stats[self.team_stats['team'] == team1].iloc[0]
                    team2_stats = self.team_stats[self.team_stats['team'] == team2].iloc[0]
                    
                    comparison_df = pd.DataFrame({
                        'Metric': ['Win Rate', 'Avg Runs/Match', 'Economy Rate', 'Recent Form'],
                        team1: [
                            f"{team1_stats['win_rate']:.1%}",
                            f"{team1_stats['avg_runs_per_match']:.1f}",
                            f"{team1_stats['economy_rate']:.1f}",
                            f"{team1_stats['recent_form']:.1%}"
                        ],
                        team2: [
                            f"{team2_stats['win_rate']:.1%}",
                            f"{team2_stats['avg_runs_per_match']:.1f}",
                            f"{team2_stats['economy_rate']:.1f}",
                            f"{team2_stats['recent_form']:.1%}"
                        ]
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Head-to-head history
                    h2h_rate = self.get_head_to_head_stats(team1, team2)
                    st.write(f"### 🎯 Head-to-Head: {team1} wins {h2h_rate:.1%} of matches against {team2}")
                    
                    # Feature importance
                    with st.expander("🔍 Feature Importance"):
                        fig = px.bar(
                            self.feature_importance.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(result)
            else:
                st.warning("⚠️ Please select different teams.")
        
        # Show model statistics
        with st.expander("📊 Model Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Training Samples", len(self.features_df))
                st.metric("Model Accuracy", f"{self.accuracy:.1%}")
            
            with col2:
                st.metric("Available Teams", len(self.team_stats))
                st.metric("Available Venues", len(self.venue_stats) if len(self.venue_stats) > 0 else 1)