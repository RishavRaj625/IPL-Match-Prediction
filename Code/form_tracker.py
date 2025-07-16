import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FormTracker:
    def __init__(self, matches_df, deliveries_df):
        """
        Initialize Form Tracker with match and delivery data
        
        Args:
            matches_df (pd.DataFrame): DataFrame containing match information
            deliveries_df (pd.DataFrame): DataFrame containing ball-by-ball data
        """
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.merged_df = self._merge_data()
        
    def _merge_data(self):
        """Merge matches and deliveries data"""
        try:
            # Ensure we have the required columns
            match_cols = ['id', 'season', 'date', 'team1', 'team2', 'winner', 'venue']
            available_match_cols = [col for col in match_cols if col in self.matches_df.columns]
            
            merged = self.deliveries_df.merge(
                self.matches_df[available_match_cols], 
                left_on='match_id', 
                right_on='id', 
                how='left'
            )
            
            # Convert date to datetime if it exists
            if 'date' in merged.columns:
                merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
            
            return merged
        except Exception as e:
            st.error(f"Error merging data: {e}")
            return pd.DataFrame()
    
    def get_team_recent_form(self, team_name, num_matches=10):
        """
        Get recent form for a specific team
        
        Args:
            team_name (str): Name of the team
            num_matches (int): Number of recent matches to consider
            
        Returns:
            pd.DataFrame: Recent form data
        """
        # Get matches where the team played
        team_matches = self.matches_df[
            (self.matches_df['team1'] == team_name) | 
            (self.matches_df['team2'] == team_name)
        ].copy()
        
        # Sort by date if available, otherwise by match id
        if 'date' in team_matches.columns:
            team_matches = team_matches.sort_values('date', ascending=False)
        else:
            team_matches = team_matches.sort_values('id', ascending=False)
        
        # Get recent matches
        recent_matches = team_matches.head(num_matches)
        
        # Add result column
        recent_matches['result'] = recent_matches['winner'].apply(
            lambda x: 'W' if x == team_name else 'L'
        )
        
        return recent_matches
    
    def calculate_team_form_score(self, team_name, num_matches=10):
        """
        Calculate a form score for a team based on recent performances
        
        Args:
            team_name (str): Name of the team
            num_matches (int): Number of recent matches to consider
            
        Returns:
            dict: Form score and details
        """
        recent_form = self.get_team_recent_form(team_name, num_matches)
        
        if recent_form.empty:
            return {'score': 0, 'wins': 0, 'losses': 0, 'win_rate': 0}
        
        wins = len(recent_form[recent_form['result'] == 'W'])
        losses = len(recent_form[recent_form['result'] == 'L'])
        win_rate = wins / len(recent_form) if len(recent_form) > 0 else 0
        
        # Calculate weighted form score (recent matches have more weight)
        weights = np.linspace(1, 0.5, len(recent_form))
        weighted_wins = sum(weights[i] for i, result in enumerate(recent_form['result']) if result == 'W')
        max_possible_score = sum(weights)
        
        form_score = (weighted_wins / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        return {
            'score': form_score,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'matches_played': len(recent_form)
        }
    
    def get_player_recent_form(self, player_name, role='batsman', num_matches=10):
        """
        Get recent form for a specific player
        
        Args:
            player_name (str): Name of the player
            role (str): Role of the player ('batsman' or 'bowler')
            num_matches (int): Number of recent matches to consider
            
        Returns:
            pd.DataFrame: Recent form data
        """
        if role == 'batsman':
            player_data = self.merged_df[self.merged_df['batsman'] == player_name]
        else:
            player_data = self.merged_df[self.merged_df['bowler'] == player_name]
        
        if player_data.empty:
            return pd.DataFrame()
        
        # Get recent matches
        recent_matches = player_data.drop_duplicates('match_id').tail(num_matches)
        
        if role == 'batsman':
            # Calculate batting stats per match
            batting_stats = []
            for match_id in recent_matches['match_id'].unique():
                match_data = player_data[player_data['match_id'] == match_id]
                
                runs = match_data['runs_scored'].sum() if 'runs_scored' in match_data.columns else 0
                balls = len(match_data)
                fours = len(match_data[match_data['runs_scored'] == 4]) if 'runs_scored' in match_data.columns else 0
                sixes = len(match_data[match_data['runs_scored'] == 6]) if 'runs_scored' in match_data.columns else 0
                
                strike_rate = (runs / balls * 100) if balls > 0 else 0
                
                batting_stats.append({
                    'match_id': match_id,
                    'runs': runs,
                    'balls': balls,
                    'fours': fours,
                    'sixes': sixes,
                    'strike_rate': strike_rate,
                    'date': match_data['date'].iloc[0] if 'date' in match_data.columns else None
                })
            
            return pd.DataFrame(batting_stats)
        
        else:  # bowler
            # Calculate bowling stats per match
            bowling_stats = []
            for match_id in recent_matches['match_id'].unique():
                match_data = player_data[player_data['match_id'] == match_id]
                
                runs_conceded = match_data['runs_scored'].sum() if 'runs_scored' in match_data.columns else 0
                balls_bowled = len(match_data)
                wickets = len(match_data[match_data['player_dismissed'].notna()]) if 'player_dismissed' in match_data.columns else 0
                
                economy = (runs_conceded / balls_bowled * 6) if balls_bowled > 0 else 0
                overs = balls_bowled / 6
                
                bowling_stats.append({
                    'match_id': match_id,
                    'runs_conceded': runs_conceded,
                    'wickets': wickets,
                    'balls_bowled': balls_bowled,
                    'overs': overs,
                    'economy': economy,
                    'date': match_data['date'].iloc[0] if 'date' in match_data.columns else None
                })
            
            return pd.DataFrame(bowling_stats)
    
    def show_tracker(self):
        """Display the form tracker interface"""
        st.title("📈 Form Tracker")
        st.write("Track recent form and performance trends for teams and players")
        
        # Sidebar for analysis type
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            # ["Team Form", "Player Form", "Head-to-Head Form"]
            ["Team Form", "Head-to-Head Form"]
        )
        
        if analysis_type == "Team Form":
            self.show_team_form_analysis()
        elif analysis_type == "Player Form":
            self.show_player_form_analysis()
        else:
            self.show_head_to_head_form()
    
    def show_team_form_analysis(self):
        """Show team form analysis"""
        st.subheader("🏏 Team Form Analysis")
        
        # Get available teams
        available_teams = sorted(set(self.matches_df['team1'].unique().tolist() + 
                                   self.matches_df['team2'].unique().tolist()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_team = st.selectbox("Select Team", available_teams)
            
        with col2:
            num_matches = st.slider("Number of Recent Matches", 5, 20, 10)
        
        if selected_team:
            # Get form data
            form_data = self.calculate_team_form_score(selected_team, num_matches)
            recent_matches = self.get_team_recent_form(selected_team, num_matches)
            
            # Display form metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Form Score", f"{form_data['score']:.1f}/100")
            
            with col2:
                st.metric("Wins", form_data['wins'])
            
            with col3:
                st.metric("Losses", form_data['losses'])
            
            with col4:
                st.metric("Win Rate", f"{form_data['win_rate']:.1%}")
            
            # Recent form visualization
            if not recent_matches.empty:
                st.subheader("Recent Match Results")
                
                # Create form chart
                fig = go.Figure()
                
                colors = ['green' if result == 'W' else 'red' for result in recent_matches['result']]
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(recent_matches))),
                    y=[1 if result == 'W' else 0 for result in recent_matches['result']],
                    mode='markers+lines',
                    marker=dict(color=colors, size=12),
                    line=dict(color='blue', width=2),
                    name='Form'
                ))
                
                fig.update_layout(
                    title=f"{selected_team} - Recent Form",
                    xaxis_title="Match (Most Recent → Oldest)",
                    yaxis_title="Result",
                    yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Loss', 'Win']),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent matches table
                st.subheader("Recent Matches Details")
                display_cols = ['date', 'team1', 'team2', 'winner', 'venue', 'result']
                available_cols = [col for col in display_cols if col in recent_matches.columns]
                
                if available_cols:
                    st.dataframe(recent_matches[available_cols].head(10))
        
        # All teams comparison
        st.subheader("All Teams Form Comparison")
        
        form_comparison = []
        for team in available_teams:
            form_data = self.calculate_team_form_score(team, num_matches)
            form_comparison.append({
                'Team': team,
                'Form Score': form_data['score'],
                'Wins': form_data['wins'],
                'Losses': form_data['losses'],
                'Win Rate': form_data['win_rate']
            })
        
        form_df = pd.DataFrame(form_comparison)
        form_df = form_df.sort_values('Form Score', ascending=False)
        
        # Form comparison chart
        fig = px.bar(
            form_df,
            x='Team',
            y='Form Score',
            color='Form Score',
            color_continuous_scale='RdYlGn',
            title=f"Team Form Comparison (Last {num_matches} matches)"
        )
        
        fig.update_layout(
            xaxis_title="Teams",
            yaxis_title="Form Score",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Form table
        st.dataframe(form_df.round(2))
    
    def show_player_form_analysis(self):
        """Show player form analysis"""
        st.subheader("👤 Player Form Analysis")
        
        # Get available players
        batsmen = self.deliveries_df['batsman'].dropna().unique().tolist()
        bowlers = self.deliveries_df['bowler'].dropna().unique().tolist()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            player_role = st.selectbox("Select Role", ["Batsman", "Bowler"])
        
        with col2:
            if player_role == "Batsman":
                selected_player = st.selectbox("Select Player", sorted(batsmen))
            else:
                selected_player = st.selectbox("Select Player", sorted(bowlers))
        
        with col3:
            num_matches = st.slider("Recent Matches", 5, 15, 10)
        
        if selected_player:
            role = player_role.lower()
            player_form = self.get_player_recent_form(selected_player, role, num_matches)
            
            if not player_form.empty:
                if role == 'batsman':
                    self.show_batsman_form(selected_player, player_form)
                else:
                    self.show_bowler_form(selected_player, player_form)
            else:
                st.warning(f"No recent data found for {selected_player}")
    
    def show_batsman_form(self, player_name, form_data):
        """Show batsman form analysis"""
        st.subheader(f"🏏 {player_name} - Batting Form")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_runs = form_data['runs'].mean()
            st.metric("Average Runs", f"{avg_runs:.1f}")
        
        with col2:
            avg_sr = form_data['strike_rate'].mean()
            st.metric("Average Strike Rate", f"{avg_sr:.1f}")
        
        with col3:
            total_fours = form_data['fours'].sum()
            st.metric("Total Fours", total_fours)
        
        with col4:
            total_sixes = form_data['sixes'].sum()
            st.metric("Total Sixes", total_sixes)
        
        # Runs trend
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Runs per Match', 'Strike Rate Trend', 'Boundaries Distribution', 'Form Consistency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Runs per match
        fig.add_trace(
            go.Scatter(x=list(range(len(form_data))), y=form_data['runs'],
                      mode='lines+markers', name='Runs', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Strike rate trend
        fig.add_trace(
            go.Scatter(x=list(range(len(form_data))), y=form_data['strike_rate'],
                      mode='lines+markers', name='Strike Rate', line=dict(color='green')),
            row=1, col=2
        )
        
        # Boundaries distribution
        fig.add_trace(
            go.Bar(x=['Fours', 'Sixes'], y=[form_data['fours'].sum(), form_data['sixes'].sum()],
                   name='Boundaries', marker_color=['orange', 'red']),
            row=2, col=1
        )
        
        # Form consistency (coefficient of variation)
        runs_cv = form_data['runs'].std() / form_data['runs'].mean() if form_data['runs'].mean() > 0 else 0
        consistency_score = max(0, 100 - runs_cv * 100)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=consistency_score,
                title={'text': "Consistency Score"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent performances table
        st.subheader("Recent Performances")
        display_data = form_data.copy()
        if 'date' in display_data.columns:
            display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_data.round(2))
    
    def show_bowler_form(self, player_name, form_data):
        """Show bowler form analysis"""
        st.subheader(f"⚾ {player_name} - Bowling Form")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_wickets = form_data['wickets'].mean()
            st.metric("Average Wickets", f"{avg_wickets:.1f}")
        
        with col2:
            avg_economy = form_data['economy'].mean()
            st.metric("Average Economy", f"{avg_economy:.1f}")
        
        with col3:
            total_wickets = form_data['wickets'].sum()
            st.metric("Total Wickets", total_wickets)
        
        with col4:
            total_overs = form_data['overs'].sum()
            st.metric("Total Overs", f"{total_overs:.1f}")
        
        # Bowling trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Wickets per Match', 'Economy Rate Trend', 'Overs Bowled', 'Performance Radar'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Wickets per match
        fig.add_trace(
            go.Scatter(x=list(range(len(form_data))), y=form_data['wickets'],
                      mode='lines+markers', name='Wickets', line=dict(color='red')),
            row=1, col=1
        )
        
        # Economy rate trend
        fig.add_trace(
            go.Scatter(x=list(range(len(form_data))), y=form_data['economy'],
                      mode='lines+markers', name='Economy Rate', line=dict(color='orange')),
            row=1, col=2
        )
        
        # Overs bowled
        fig.add_trace(
            go.Bar(x=list(range(len(form_data))), y=form_data['overs'],
                   name='Overs', marker_color='blue'),
            row=2, col=1
        )
        
        # Performance radar chart
        categories = ['Wickets', 'Economy', 'Consistency']
        values = [
            (form_data['wickets'].mean() / 3) * 100,  # Normalized to 0-100
            max(0, 100 - (form_data['economy'].mean() / 10) * 100),  # Inverted economy
            max(0, 100 - (form_data['wickets'].std() / form_data['wickets'].mean() * 100)) if form_data['wickets'].mean() > 0 else 0
        ]
        
        fig.add_trace(
            go.Scatterpolar(r=values, theta=categories, fill='toself', name='Performance'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent performances table
        st.subheader("Recent Performances")
        display_data = form_data.copy()
        if 'date' in display_data.columns:
            display_data['date'] = pd.to_datetime(display_data['date']).dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_data.round(2))
    
    def show_head_to_head_form(self):
        """Show head-to-head form analysis"""
        st.subheader("⚔️ Head-to-Head Form")
        
        # Get available teams
        available_teams = sorted(set(self.matches_df['team1'].unique().tolist() + 
                                   self.matches_df['team2'].unique().tolist()))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            team1 = st.selectbox("Select Team 1", available_teams, key="h2h_team1")
        
        with col2:
            team2 = st.selectbox("Select Team 2", available_teams, key="h2h_team2")
        
        with col3:
            num_matches = st.slider("Recent H2H Matches", 5, 15, 10)
        
        if team1 and team2 and team1 != team2:
            # Get head-to-head matches
            h2h_matches = self.matches_df[
                ((self.matches_df['team1'] == team1) & (self.matches_df['team2'] == team2)) |
                ((self.matches_df['team1'] == team2) & (self.matches_df['team2'] == team1))
            ]
            
            if not h2h_matches.empty:
                # Sort by date if available
                if 'date' in h2h_matches.columns:
                    h2h_matches = h2h_matches.sort_values('date', ascending=False)
                else:
                    h2h_matches = h2h_matches.sort_values('id', ascending=False)
                
                recent_h2h = h2h_matches.head(num_matches)
                
                # Calculate head-to-head stats
                team1_wins = len(recent_h2h[recent_h2h['winner'] == team1])
                team2_wins = len(recent_h2h[recent_h2h['winner'] == team2])
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(f"{team1} Wins", team1_wins)
                
                with col2:
                    st.metric(f"{team2} Wins", team2_wins)
                
                with col3:
                    st.metric("Total Matches", len(recent_h2h))
                
                # Head-to-head visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Head-to-Head Record', 'Recent Form'),
                    specs=[[{"type": "pie"}, {"type": "scatter"}]]
                )
                
                # Pie chart for overall record
                fig.add_trace(
                    go.Pie(labels=[team1, team2], values=[team1_wins, team2_wins],
                           name="H2H Record"),
                    row=1, col=1
                )
                
                # Recent form timeline
                recent_h2h['match_num'] = range(len(recent_h2h))
                colors = ['blue' if winner == team1 else 'red' for winner in recent_h2h['winner']]
                
                fig.add_trace(
                    go.Scatter(
                        x=recent_h2h['match_num'],
                        y=[1 if winner == team1 else 0 for winner in recent_h2h['winner']],
                        mode='markers+lines',
                        marker=dict(color=colors, size=10),
                        name='Recent Form'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recent matches table
                st.subheader("Recent Head-to-Head Matches")
                display_cols = ['date', 'team1', 'team2', 'winner', 'venue']
                available_cols = [col for col in display_cols if col in recent_h2h.columns]
                
                if available_cols:
                    st.dataframe(recent_h2h[available_cols])
                
            else:
                st.warning(f"No head-to-head data found between {team1} and {team2}")
        
        elif team1 == team2:
            st.warning("Please select different teams for head-to-head analysis")