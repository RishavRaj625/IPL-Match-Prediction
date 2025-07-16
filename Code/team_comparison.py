import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class TeamComparison:
    def __init__(self, matches_df, deliveries_df):
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.merged_df = self.deliveries_df.merge(
            self.matches_df[['id', 'season', 'date', 'venue', 'winner']], 
            left_on='match_id', 
            right_on='id', 
            how='left'
        )
        self.teams = self.get_unique_teams()
    
    def get_unique_teams(self):
        """Get list of unique teams"""
        teams = set()
        if 'team1' in self.matches_df.columns:
            teams.update(self.matches_df['team1'].dropna().unique())
        if 'team2' in self.matches_df.columns:
            teams.update(self.matches_df['team2'].dropna().unique())
        if 'batting_team' in self.deliveries_df.columns:
            teams.update(self.deliveries_df['batting_team'].dropna().unique())
        return sorted(list(teams))
    
    def get_head_to_head_stats(self, team1, team2):
        """Get head-to-head statistics between two teams"""
        if 'team1' not in self.matches_df.columns or 'team2' not in self.matches_df.columns:
            return None
        
        # Filter matches between the two teams
        h2h_matches = self.matches_df[
            ((self.matches_df['team1'] == team1) & (self.matches_df['team2'] == team2)) |
            ((self.matches_df['team1'] == team2) & (self.matches_df['team2'] == team1))
        ].copy()
        
        if len(h2h_matches) == 0:
            return None
        
        # Calculate wins
        team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])
        
        # Calculate recent form (last 5 matches)
        recent_matches = h2h_matches.tail(5) if len(h2h_matches) >= 5 else h2h_matches
        team1_recent_wins = len(recent_matches[recent_matches['winner'] == team1])
        team2_recent_wins = len(recent_matches[recent_matches['winner'] == team2])
        
        return {
            'total_matches': len(h2h_matches),
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'team1_recent_wins': team1_recent_wins,
            'team2_recent_wins': team2_recent_wins,
            'recent_matches': len(recent_matches)
        }
    
    def get_team_batting_stats(self, team, season=None):
        """Get batting statistics for a team"""
        if 'batting_team' not in self.deliveries_df.columns:
            return None
        
        team_data = self.merged_df[self.merged_df['batting_team'] == team].copy()
        
        if season:
            team_data = team_data[team_data['season'] == season]
        
        if len(team_data) == 0:
            return None
        
        # Calculate batting stats
        if 'runs_scored' in team_data.columns:
            total_runs = team_data['runs_scored'].sum()
            avg_runs_per_match = total_runs / len(team_data['match_id'].unique()) if len(team_data['match_id'].unique()) > 0 else 0
            strike_rate = (total_runs / len(team_data)) * 100 if len(team_data) > 0 else 0
            
            # Boundaries
            fours = len(team_data[team_data['runs_scored'] == 4])
            sixes = len(team_data[team_data['runs_scored'] == 6])
            
            return {
                'total_runs': total_runs,
                'avg_runs_per_match': avg_runs_per_match,
                'strike_rate': strike_rate,
                'fours': fours,
                'sixes': sixes,
                'total_balls': len(team_data)
            }
        
        return None
    
    def get_team_bowling_stats(self, team, season=None):
        """Get bowling statistics for a team"""
        if 'bowling_team' not in self.deliveries_df.columns:
            return None
        
        team_data = self.merged_df[self.merged_df['bowling_team'] == team].copy()
        
        if season:
            team_data = team_data[team_data['season'] == season]
        
        if len(team_data) == 0:
            return None
        
        # Calculate bowling stats
        if 'runs_scored' in team_data.columns:
            runs_conceded = team_data['runs_scored'].sum()
            avg_runs_conceded_per_match = runs_conceded / len(team_data['match_id'].unique()) if len(team_data['match_id'].unique()) > 0 else 0
            economy_rate = (runs_conceded / len(team_data)) * 6 if len(team_data) > 0 else 0
            
            # Wickets
            if 'player_dismissed' in team_data.columns:
                wickets = len(team_data[team_data['player_dismissed'].notna()])
            else:
                wickets = 0
            
            return {
                'runs_conceded': runs_conceded,
                'avg_runs_conceded_per_match': avg_runs_conceded_per_match,
                'economy_rate': economy_rate,
                'wickets': wickets,
                'total_balls': len(team_data)
            }
        
        return None
    
    def plot_head_to_head_comparison(self, team1, team2, h2h_stats):
        """Create head-to-head comparison visualization"""
        if not h2h_stats:
            return None
        
        # Overall wins comparison
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Overall Head-to-Head', 'Recent Form (Last 5)'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Overall H2H pie chart
        fig.add_trace(
            go.Pie(
                labels=[team1, team2],
                values=[h2h_stats['team1_wins'], h2h_stats['team2_wins']],
                name="Overall"
            ),
            row=1, col=1
        )
        
        # Recent form pie chart
        fig.add_trace(
            go.Pie(
                labels=[team1, team2],
                values=[h2h_stats['team1_recent_wins'], h2h_stats['team2_recent_wins']],
                name="Recent"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"{team1} vs {team2} - Head to Head Analysis",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def plot_team_performance_comparison(self, team1, team2, season=None):
        """Compare team performance metrics"""
        team1_batting = self.get_team_batting_stats(team1, season)
        team2_batting = self.get_team_batting_stats(team2, season)
        team1_bowling = self.get_team_bowling_stats(team1, season)
        team2_bowling = self.get_team_bowling_stats(team2, season)
        
        if not all([team1_batting, team2_batting, team1_bowling, team2_bowling]):
            return None
        
        # Create comparison metrics
        metrics = ['Avg Runs/Match', 'Strike Rate', 'Fours', 'Sixes', 'Economy Rate']
        team1_values = [
            team1_batting['avg_runs_per_match'],
            team1_batting['strike_rate'],
            team1_batting['fours'],
            team1_batting['sixes'],
            team1_bowling['economy_rate']
        ]
        team2_values = [
            team2_batting['avg_runs_per_match'],
            team2_batting['strike_rate'],
            team2_batting['fours'],
            team2_batting['sixes'],
            team2_bowling['economy_rate']
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=team1_values,
            theta=metrics,
            fill='toself',
            name=team1,
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=team2_values,
            theta=metrics,
            fill='toself',
            name=team2,
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(max(team1_values), max(team2_values)) * 1.1])
            ),
            showlegend=True,
            title=f"Performance Comparison: {team1} vs {team2}",
            height=500
        )
        
        return fig
    
    def plot_season_wise_comparison(self, team1, team2):
        """Plot season-wise performance comparison"""
        if 'season' not in self.merged_df.columns:
            return None
        
        seasons = sorted(self.merged_df['season'].unique())
        
        team1_runs = []
        team2_runs = []
        
        for season in seasons:
            team1_stats = self.get_team_batting_stats(team1, season)
            team2_stats = self.get_team_batting_stats(team2, season)
            
            team1_runs.append(team1_stats['avg_runs_per_match'] if team1_stats else 0)
            team2_runs.append(team2_stats['avg_runs_per_match'] if team2_stats else 0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=team1_runs,
            mode='lines+markers',
            name=team1,
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=seasons,
            y=team2_runs,
            mode='lines+markers',
            name=team2,
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title=f"Season-wise Average Runs Comparison: {team1} vs {team2}",
            xaxis_title="Season",
            yaxis_title="Average Runs per Match",
            height=400
        )
        
        return fig
    
    def show_comparison(self):
        """Main function to display team comparison interface"""
        st.title("⚔️ Team vs Team Comparison")
        
        if len(self.teams) < 2:
            st.error("Not enough teams available for comparison")
            return
        
        # Team selection
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Team 1", self.teams, key="team1")
        
        with col2:
            team2 = st.selectbox("Select Team 2", [t for t in self.teams if t != team1], key="team2")
        
        # Season filter
        if 'season' in self.merged_df.columns:
            seasons = ['All Seasons'] + sorted(self.merged_df['season'].unique().tolist())
            selected_season = st.selectbox("Select Season", seasons)
            season_filter = None if selected_season == 'All Seasons' else selected_season
        else:
            season_filter = None
        
        if team1 == team2:
            st.warning("Please select two different teams")
            return
        
        # Get head-to-head stats
        h2h_stats = self.get_head_to_head_stats(team1, team2)
        
        if h2h_stats:
            # Display head-to-head metrics
            st.subheader("🏆 Head-to-Head Record")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Matches", h2h_stats['total_matches'])
            
            with col2:
                st.metric(f"{team1} Wins", h2h_stats['team1_wins'])
            
            with col3:
                st.metric(f"{team2} Wins", h2h_stats['team2_wins'])
            
            # Head-to-head visualization
            h2h_fig = self.plot_head_to_head_comparison(team1, team2, h2h_stats)
            if h2h_fig:
                st.plotly_chart(h2h_fig, use_container_width=True)
        
        # Performance comparison
        st.subheader("📊 Performance Comparison")
        
        # Get team stats
        team1_batting = self.get_team_batting_stats(team1, season_filter)
        team2_batting = self.get_team_batting_stats(team2, season_filter)
        team1_bowling = self.get_team_bowling_stats(team1, season_filter)
        team2_bowling = self.get_team_bowling_stats(team2, season_filter)
        
        if team1_batting and team2_batting:
            # Batting comparison
            st.subheader("🏏 Batting Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{team1}**")
                st.metric("Total Runs", f"{team1_batting['total_runs']:,}")
                st.metric("Avg Runs/Match", f"{team1_batting['avg_runs_per_match']:.1f}")
                st.metric("Strike Rate", f"{team1_batting['strike_rate']:.2f}")
                st.metric("Fours", team1_batting['fours'])
                st.metric("Sixes", team1_batting['sixes'])
            
            with col2:
                st.markdown(f"**{team2}**")
                st.metric("Total Runs", f"{team2_batting['total_runs']:,}")
                st.metric("Avg Runs/Match", f"{team2_batting['avg_runs_per_match']:.1f}")
                st.metric("Strike Rate", f"{team2_batting['strike_rate']:.2f}")
                st.metric("Fours", team2_batting['fours'])
                st.metric("Sixes", team2_batting['sixes'])
        
        if team1_bowling and team2_bowling:
            # Bowling comparison
            st.subheader("⚾ Bowling Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{team1}**")
                st.metric("Runs Conceded", f"{team1_bowling['runs_conceded']:,}")
                st.metric("Avg Runs Conceded/Match", f"{team1_bowling['avg_runs_conceded_per_match']:.1f}")
                st.metric("Economy Rate", f"{team1_bowling['economy_rate']:.2f}")
                st.metric("Wickets", team1_bowling['wickets'])
            
            with col2:
                st.markdown(f"**{team2}**")
                st.metric("Runs Conceded", f"{team2_bowling['runs_conceded']:,}")
                st.metric("Avg Runs Conceded/Match", f"{team2_bowling['avg_runs_conceded_per_match']:.1f}")
                st.metric("Economy Rate", f"{team2_bowling['economy_rate']:.2f}")
                st.metric("Wickets", team2_bowling['wickets'])
        
        # Performance radar chart
        if all([team1_batting, team2_batting, team1_bowling, team2_bowling]):
            performance_fig = self.plot_team_performance_comparison(team1, team2, season_filter)
            if performance_fig:
                st.plotly_chart(performance_fig, use_container_width=True)
        
        # Season-wise comparison
        if season_filter is None and 'season' in self.merged_df.columns:
            st.subheader("📈 Season-wise Performance")
            season_fig = self.plot_season_wise_comparison(team1, team2)
            if season_fig:
                st.plotly_chart(season_fig, use_container_width=True)
        
        # Additional insights
        st.subheader("💡 Key Insights")
        
        insights = []
        
        if h2h_stats:
            if h2h_stats['team1_wins'] > h2h_stats['team2_wins']:
                insights.append(f"📊 {team1} has a better head-to-head record with {h2h_stats['team1_wins']} wins vs {h2h_stats['team2_wins']} wins")
            elif h2h_stats['team2_wins'] > h2h_stats['team1_wins']:
                insights.append(f"📊 {team2} has a better head-to-head record with {h2h_stats['team2_wins']} wins vs {h2h_stats['team1_wins']} wins")
            else:
                insights.append(f"📊 Both teams are evenly matched with {h2h_stats['team1_wins']} wins each")
        
        if team1_batting and team2_batting:
            if team1_batting['avg_runs_per_match'] > team2_batting['avg_runs_per_match']:
                insights.append(f"🏏 {team1} has a stronger batting average ({team1_batting['avg_runs_per_match']:.1f} vs {team2_batting['avg_runs_per_match']:.1f})")
            else:
                insights.append(f"🏏 {team2} has a stronger batting average ({team2_batting['avg_runs_per_match']:.1f} vs {team1_batting['avg_runs_per_match']:.1f})")
        
        if team1_bowling and team2_bowling:
            if team1_bowling['economy_rate'] < team2_bowling['economy_rate']:
                insights.append(f"⚾ {team1} has a better bowling economy rate ({team1_bowling['economy_rate']:.2f} vs {team2_bowling['economy_rate']:.2f})")
            else:
                insights.append(f"⚾ {team2} has a better bowling economy rate ({team2_bowling['economy_rate']:.2f} vs {team1_bowling['economy_rate']:.2f})")
        
        for insight in insights:
            st.write(f"• {insight}")
        
        if not insights:
            st.info("Select teams to see detailed comparison insights")