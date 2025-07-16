import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

class ProfileComparison:
    def __init__(self, matches_df, deliveries_df, players_details_df):
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.players_details_df = players_details_df
        
    def show_comparison(self):
        """Main function to display profile comparison"""
        st.markdown("# 📋 Player Profile Comparison")
        st.markdown("Compare detailed player profiles, statistics, and career information (Based on 2024 dataset)")
        
        # Sidebar for player selection
        st.sidebar.markdown("## Player Selection")
        
        # Get available players
        available_players = self.players_details_df['name'].unique() if 'name' in self.players_details_df.columns else []
        
        if len(available_players) == 0:
            st.error("No player data available")
            return
            
        # Select players for comparison
        selected_players = st.sidebar.multiselect(
            "Select Players (up to 4)",
            available_players,
            default=available_players[:2] if len(available_players) >= 2 else available_players[:1],
            max_selections=4
        )
        
        if not selected_players:
            st.warning("Please select at least one player to view profile")
            return
        
        # Display player profiles
        self.display_player_profiles(selected_players)
        
        # Comparison charts
        if len(selected_players) > 1:
            self.display_comparison_charts(selected_players)
        
        # Performance statistics
        self.display_performance_stats(selected_players)
    
    def display_player_profiles(self, selected_players):
        """Display detailed player profiles"""
        st.markdown("## 👤 Player Profiles")
        
        for player in selected_players:
            player_info = self.players_details_df[self.players_details_df['name'] == player]
            
            if player_info.empty:
                st.warning(f"No profile data found for {player}")
                continue
            
            player_info = player_info.iloc[0]
            
            with st.expander(f"🏏 {player} - Player Profile", expanded=True):
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    # Player image (placeholder if not available)
                    if 'image_url' in player_info and pd.notna(player_info['image_url']):
                        st.image(player_info['image_url'], width=150)
                    else:
                        st.image("https://via.placeholder.com/150x150.png?text=Player", width=150)
                
                with col2:
                    # Basic information
                    st.markdown(f"### {player}")
                    
                    if 'long_name' in player_info and pd.notna(player_info['long_name']):
                        st.markdown(f"**Full Name:** {player_info['long_name']}")
                    
                    if 'date_of_birth' in player_info and pd.notna(player_info['date_of_birth']):
                        st.markdown(f"**Date of Birth:** {player_info['date_of_birth'].strftime('%B %d, %Y')}")
                    
                    if 'age' in player_info and pd.notna(player_info['age']):
                        st.markdown(f"**Age:** {int(player_info['age'])} years")
                    
                    if 'playing_role' in player_info and pd.notna(player_info['playing_role']):
                        st.markdown(f"**Playing Role:** {player_info['playing_role']}")
                
                with col3:
                    # Links
                    if 'espn_url' in player_info and pd.notna(player_info['espn_url']):
                        st.markdown(f"[ESPN Profile]({player_info['espn_url']})")
                
                # Playing styles
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏏 Batting Style")
                    if 'batting_style' in player_info and pd.notna(player_info['batting_style']):
                        st.write(f"**Style:** {player_info['batting_style']}")
                    if 'long_batting_style' in player_info and pd.notna(player_info['long_batting_style']):
                        st.write(f"**Details:** {player_info['long_batting_style']}")
                
                with col2:
                    st.markdown("#### ⚾ Bowling Style")
                    if 'bowling_style' in player_info and pd.notna(player_info['bowling_style']):
                        st.write(f"**Style:** {player_info['bowling_style']}")
                    if 'long_bowling_style' in player_info and pd.notna(player_info['long_bowling_style']):
                        st.write(f"**Details:** {player_info['long_bowling_style']}")
    
    def display_comparison_charts(self, selected_players):
        """Display comparison charts for selected players"""
        st.markdown("## 📊 Player Comparison Charts")
        
        # Age comparison
        self.create_age_comparison(selected_players)
        
        # Style distribution
        self.create_style_comparison(selected_players)
        
        # Role comparison
        self.create_role_comparison(selected_players)
    
    def create_age_comparison(self, selected_players):
        """Create age comparison chart"""
        st.markdown("### 📈 Age Comparison")
        
        ages = []
        names = []
        
        for player in selected_players:
            player_info = self.players_details_df[self.players_details_df['name'] == player]
            if not player_info.empty and 'age' in player_info.columns:
                age = player_info.iloc[0]['age']
                if pd.notna(age):
                    ages.append(int(age))
                    names.append(player)
        
        if ages:
            fig = px.bar(x=names, y=ages, title="Age Comparison")
            fig.update_layout(xaxis_title="Players", yaxis_title="Age (years)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Age data not available for selected players")
    
    def create_style_comparison(self, selected_players):
        """Create batting and bowling style comparison"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏏 Batting Styles")
            batting_styles = []
            player_names = []
            
            for player in selected_players:
                player_info = self.players_details_df[self.players_details_df['name'] == player]
                if not player_info.empty and 'batting_style' in player_info.columns:
                    style = player_info.iloc[0]['batting_style']
                    if pd.notna(style):
                        batting_styles.append(style)
                        player_names.append(player)
            
            if batting_styles:
                batting_df = pd.DataFrame({'Player': player_names, 'Batting Style': batting_styles})
                fig = px.pie(batting_df, names='Batting Style', title="Batting Styles Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Batting style data not available")
        
        with col2:
            st.markdown("### ⚾ Bowling Styles")
            bowling_styles = []
            player_names = []
            
            for player in selected_players:
                player_info = self.players_details_df[self.players_details_df['name'] == player]
                if not player_info.empty and 'bowling_style' in player_info.columns:
                    style = player_info.iloc[0]['bowling_style']
                    if pd.notna(style):
                        bowling_styles.append(style)
                        player_names.append(player)
            
            if bowling_styles:
                bowling_df = pd.DataFrame({'Player': player_names, 'Bowling Style': bowling_styles})
                fig = px.pie(bowling_df, names='Bowling Style', title="Bowling Styles Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Bowling style data not available")
    
    def create_role_comparison(self, selected_players):
        """Create playing role comparison"""
        st.markdown("### 🎯 Playing Roles")
        
        roles = []
        names = []
        
        for player in selected_players:
            player_info = self.players_details_df[self.players_details_df['name'] == player]
            if not player_info.empty and 'playing_role' in player_info.columns:
                role = player_info.iloc[0]['playing_role']
                if pd.notna(role):
                    roles.append(role)
                    names.append(player)
        
        if roles:
            roles_df = pd.DataFrame({'Player': names, 'Playing Role': roles})
            fig = px.bar(roles_df, x='Player', y='Playing Role', title="Playing Roles Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Playing role data not available for selected players")
    
    def display_performance_stats(self, selected_players):
        """Display performance statistics for selected players"""
        st.markdown("## 📈 Performance Statistics")
        
        # Get batting statistics
        batting_stats = self.get_batting_statistics(selected_players)
        
        # Get bowling statistics
        bowling_stats = self.get_bowling_statistics(selected_players)
        
        # Display statistics
        if not batting_stats.empty:
            st.markdown("### 🏏 Batting Statistics")
            st.dataframe(batting_stats, use_container_width=True)
            
            # Create batting performance chart
            self.create_batting_performance_chart(batting_stats)
        
        if not bowling_stats.empty:
            st.markdown("### ⚾ Bowling Statistics")
            st.dataframe(bowling_stats, use_container_width=True)
            
            # Create bowling performance chart
            self.create_bowling_performance_chart(bowling_stats)
    
    def get_batting_statistics(self, selected_players):
        """Calculate batting statistics for selected players"""
        batting_stats = []
        
        for player in selected_players:
            # Filter deliveries for this batsman
            if 'batsman' in self.deliveries_df.columns:
                player_balls = self.deliveries_df[self.deliveries_df['batsman'] == player]
            else:
                continue
            
            if player_balls.empty:
                continue
            
            # Calculate statistics with proper column names
            if 'batsman_runs' in player_balls.columns:
                total_runs = player_balls['batsman_runs'].sum()
                boundaries = len(player_balls[player_balls['batsman_runs'] == 4])
                sixes = len(player_balls[player_balls['batsman_runs'] == 6])
            else:
                total_runs = 0
                boundaries = 0
                sixes = 0
            
            total_balls = len(player_balls)
            
            # Calculate dismissals
            if 'player_dismissed' in player_balls.columns:
                dismissals = len(player_balls[player_balls['player_dismissed'] == player])
            else:
                dismissals = 0
            
            # Calculate strike rate
            strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0
            
            # Calculate average
            average = total_runs / dismissals if dismissals > 0 else total_runs
            
            # Get matches played
            if 'match_id' in player_balls.columns:
                matches_played = player_balls['match_id'].nunique()
            else:
                matches_played = 0
            
            batting_stats.append({
                'Player': player,
                'Matches': matches_played,
                'Runs': total_runs,
                'Balls': total_balls,
                'Average': round(average, 2),
                'Strike Rate': round(strike_rate, 2),
                'Boundaries': boundaries,
                'Sixes': sixes,
                'Dismissals': dismissals
            })
        
        return pd.DataFrame(batting_stats)
    
    def get_bowling_statistics(self, selected_players):
        """Calculate bowling statistics for selected players"""
        bowling_stats = []
        
        for player in selected_players:
            # Filter deliveries for this bowler
            if 'bowler' in self.deliveries_df.columns:
                player_balls = self.deliveries_df[self.deliveries_df['bowler'] == player]
            else:
                continue
            
            if player_balls.empty:
                continue
            
            # Calculate statistics with proper column names
            total_balls = len(player_balls)
            
            # Use the correct column name for runs
            if 'runs_scored' in player_balls.columns:
                total_runs = player_balls['runs_scored'].sum()
            elif 'batsman_runs' in player_balls.columns:
                # If runs_scored is not available, use batsman_runs + extras
                batsman_runs = player_balls['batsman_runs'].sum()
                extras = player_balls['extras'].sum() if 'extras' in player_balls.columns else 0
                total_runs = batsman_runs + extras
            else:
                total_runs = 0
            
            # Calculate wickets
            if 'player_dismissed' in player_balls.columns:
                wickets = len(player_balls[player_balls['player_dismissed'].notna()])
            else:
                wickets = 0
            
            # Calculate economy rate (runs per over)
            overs = total_balls / 6
            economy = total_runs / overs if overs > 0 else 0
            
            # Calculate average
            average = total_runs / wickets if wickets > 0 else 0
            
            # Calculate strike rate (balls per wicket)
            strike_rate = total_balls / wickets if wickets > 0 else 0
            
            # Get matches played
            if 'match_id' in player_balls.columns:
                matches_played = player_balls['match_id'].nunique()
            else:
                matches_played = 0
            
            bowling_stats.append({
                'Player': player,
                'Matches': matches_played,
                'Balls': total_balls,
                'Overs': round(overs, 1),
                'Runs': total_runs,
                'Wickets': wickets,
                'Economy': round(economy, 2),
                'Average': round(average, 2),
                'Strike Rate': round(strike_rate, 2)
            })
        
        return pd.DataFrame(bowling_stats)
    
    def create_batting_performance_chart(self, batting_stats):
        """Create batting performance comparison chart"""
        if batting_stats.empty:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Runs comparison
            fig = px.bar(batting_stats, x='Player', y='Runs', title="Total Runs Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strike rate comparison
            fig = px.bar(batting_stats, x='Player', y='Strike Rate', title="Strike Rate Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Average and boundaries
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(batting_stats, x='Player', y='Average', title="Batting Average Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boundaries comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Boundaries', x=batting_stats['Player'], y=batting_stats['Boundaries']))
            fig.add_trace(go.Bar(name='Sixes', x=batting_stats['Player'], y=batting_stats['Sixes']))
            fig.update_layout(barmode='group', title="Boundaries vs Sixes")
            st.plotly_chart(fig, use_container_width=True)
    
    def create_bowling_performance_chart(self, bowling_stats):
        """Create bowling performance comparison chart"""
        if bowling_stats.empty:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Wickets comparison
            fig = px.bar(bowling_stats, x='Player', y='Wickets', title="Total Wickets Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Economy rate comparison
            fig = px.bar(bowling_stats, x='Player', y='Economy', title="Economy Rate Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Average and strike rate
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(bowling_stats, x='Player', y='Average', title="Bowling Average Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(bowling_stats, x='Player', y='Strike Rate', title="Bowling Strike Rate Comparison")
            st.plotly_chart(fig, use_container_width=True)