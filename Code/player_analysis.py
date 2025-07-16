import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

class PlayerAnalysis:
    def __init__(self, matches_df, deliveries_df):
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.prepare_player_data()
    
    def prepare_player_data(self):
        """Prepare player statistics data"""
        # Batting statistics
        self.batting_stats = self.deliveries_df.groupby('batsman').agg({
            'runs_scored': ['sum', 'count'],
            'match_id': 'nunique'
        }).reset_index()
        
        self.batting_stats.columns = ['player', 'total_runs', 'balls_faced', 'matches']
        self.batting_stats['strike_rate'] = (self.batting_stats['total_runs'] / 
                                           self.batting_stats['balls_faced'] * 100).round(2)
        self.batting_stats['average'] = (self.batting_stats['total_runs'] / 
                                       self.batting_stats['matches']).round(2)
        
        # Bowling statistics
        bowling_data = self.deliveries_df[self.deliveries_df['player_dismissed'].notna()]
        self.bowling_stats = bowling_data.groupby('bowler').agg({
            'player_dismissed': 'count',
            'match_id': 'nunique'
        }).reset_index()
        
        self.bowling_stats.columns = ['player', 'wickets', 'matches']
        self.bowling_stats['avg_wickets_per_match'] = (self.bowling_stats['wickets'] / 
                                                      self.bowling_stats['matches']).round(2)
        
        # Economy rate calculation
        bowler_runs = self.deliveries_df.groupby('bowler')['runs_scored'].sum().reset_index()
        bowler_balls = self.deliveries_df.groupby('bowler').size().reset_index(name='balls')
        
        economy_df = bowler_runs.merge(bowler_balls, on='bowler')
        economy_df['economy_rate'] = (economy_df['runs_scored'] / 
                                     (economy_df['balls'] / 6)).round(2)
        
        self.bowling_stats = self.bowling_stats.merge(
            economy_df[['bowler', 'economy_rate']], 
            left_on='player', 
            right_on='bowler', 
            how='left'
        ).drop('bowler', axis=1)
    
    def show_analysis(self):
        """Display player analysis interface"""
        st.title("👤 Player Performance Analysis")
        
        # Player selection
        all_players = list(set(self.deliveries_df['batsman'].unique().tolist() + 
                              self.deliveries_df['bowler'].unique().tolist()))
        
        selected_player = st.selectbox("Select Player", sorted(all_players))
        
        if selected_player:
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overview", 
                "🏏 Batting Analysis", 
                "🎯 Bowling Analysis", 
                "📈 Performance Trends"
            ])
            
            with tab1:
                self.show_player_overview(selected_player)
            
            with tab2:
                self.show_batting_analysis(selected_player)
            
            with tab3:
                self.show_bowling_analysis(selected_player)
            
            with tab4:
                self.show_performance_trends(selected_player)
    
    def show_player_overview(self, player):
        """Show player overview statistics"""
        st.subheader(f"📋 {player} - Overview")
        
        # Get player stats
        batting_stat = self.batting_stats[self.batting_stats['player'] == player]
        bowling_stat = self.bowling_stats[self.bowling_stats['player'] == player]
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not batting_stat.empty:
                st.metric("Total Runs", int(batting_stat['total_runs'].iloc[0]))
            else:
                st.metric("Total Runs", 0)
        
        with col2:
            if not batting_stat.empty:
                st.metric("Strike Rate", f"{batting_stat['strike_rate'].iloc[0]}%")
            else:
                st.metric("Strike Rate", "N/A")
        
        with col3:
            if not bowling_stat.empty:
                st.metric("Total Wickets", int(bowling_stat['wickets'].iloc[0]))
            else:
                st.metric("Total Wickets", 0)
        
        with col4:
            if not bowling_stat.empty:
                st.metric("Economy Rate", f"{bowling_stat['economy_rate'].iloc[0]}")
            else:
                st.metric("Economy Rate", "N/A")
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            if not batting_stat.empty:
                # Batting rank
                batting_rank = (self.batting_stats.sort_values('total_runs', ascending=False)
                               .reset_index(drop=True)
                               .index[self.batting_stats.sort_values('total_runs', ascending=False)['player'] == player].tolist())
                
                if batting_rank:
                    st.info(f"🏏 Batting Rank: #{batting_rank[0] + 1} (by total runs)")
        
        with col2:
            if not bowling_stat.empty:
                # Bowling rank
                bowling_rank = (self.bowling_stats.sort_values('wickets', ascending=False)
                               .reset_index(drop=True)
                               .index[self.bowling_stats.sort_values('wickets', ascending=False)['player'] == player].tolist())
                
                if bowling_rank:
                    st.info(f"🎯 Bowling Rank: #{bowling_rank[0] + 1} (by total wickets)")
    
    def show_batting_analysis(self, player):
        """Show detailed batting analysis"""
        st.subheader(f"🏏 {player} - Batting Analysis")
        
        # Filter batting data
        batting_data = self.deliveries_df[self.deliveries_df['batsman'] == player]
        
        if batting_data.empty:
            st.warning(f"No batting data found for {player}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Runs by over
            runs_by_over = batting_data.groupby('over')['runs_scored'].sum().reset_index()
            fig = px.bar(runs_by_over, x='over', y='runs_scored', 
                        title=f"{player} - Runs by Over")
            fig.update_layout(xaxis_title="Over", yaxis_title="Runs")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Runs distribution
            runs_dist = batting_data['runs_scored'].value_counts().sort_index()
            fig = px.pie(values=runs_dist.values, names=runs_dist.index, 
                        title=f"{player} - Runs Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Match-wise performance
        st.subheader("📊 Match-wise Performance")
        match_performance = batting_data.groupby('match_id')['runs_scored'].sum().reset_index()
        match_performance = match_performance.merge(
            self.matches_df[['id', 'date', 'venue']], 
            left_on='match_id', 
            right_on='id'
        )
        
        fig = px.line(match_performance, x='date', y='runs_scored', 
                     title=f"{player} - Match-wise Runs",
                     hover_data=['venue'])
        fig.update_layout(xaxis_title="Date", yaxis_title="Runs")
        st.plotly_chart(fig, use_container_width=True)
        
        # Venue performance
        st.subheader("🏟️ Venue Performance")
        venue_performance = batting_data.merge(
            self.matches_df[['id', 'venue']], 
            left_on='match_id', 
            right_on='id'
        ).groupby('venue')['runs_scored'].agg(['sum', 'mean']).reset_index()
        
        venue_performance.columns = ['venue', 'total_runs', 'avg_runs']
        venue_performance = venue_performance.sort_values('total_runs', ascending=False)
        
        fig = px.bar(venue_performance, x='venue', y='total_runs', 
                    title=f"{player} - Runs by Venue")
        fig.update_layout(xaxis_title="Venue", yaxis_title="Total Runs")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_bowling_analysis(self, player):
        """Show detailed bowling analysis"""
        st.subheader(f"🎯 {player} - Bowling Analysis")
        
        # Filter bowling data
        bowling_data = self.deliveries_df[self.deliveries_df['bowler'] == player]
        
        if bowling_data.empty:
            st.warning(f"No bowling data found for {player}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Wickets by over
            wickets_by_over = bowling_data[bowling_data['player_dismissed'].notna()].groupby('over').size().reset_index(name='wickets')
            fig = px.bar(wickets_by_over, x='over', y='wickets', 
                        title=f"{player} - Wickets by Over")
            fig.update_layout(xaxis_title="Over", yaxis_title="Wickets")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dismissal types
            dismissal_types = bowling_data[bowling_data['player_dismissed'].notna()]['dismissal_kind'].value_counts()
            fig = px.pie(values=dismissal_types.values, names=dismissal_types.index, 
                        title=f"{player} - Dismissal Types")
            st.plotly_chart(fig, use_container_width=True)
        
        # Economy rate by over
        st.subheader("📈 Economy Rate Analysis")
        economy_by_over = bowling_data.groupby('over')['runs_scored'].sum().reset_index()
        economy_by_over['balls'] = bowling_data.groupby('over').size().reset_index(name='balls')['balls']
        economy_by_over['economy_rate'] = (economy_by_over['runs_scored'] / 
                                         (economy_by_over['balls'] / 6)).round(2)
        
        fig = px.line(economy_by_over, x='over', y='economy_rate', 
                     title=f"{player} - Economy Rate by Over")
        fig.update_layout(xaxis_title="Over", yaxis_title="Economy Rate")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance against teams
        st.subheader("⚔️ Performance Against Teams")
        team_performance = bowling_data.merge(
            self.matches_df[['id', 'team1', 'team2']], 
            left_on='match_id', 
            right_on='id'
        )
        
        # Determine opposition team
        team_performance['opposition'] = team_performance.apply(
            lambda x: x['team1'] if x['bowling_team'] == x['team2'] else x['team2'], 
            axis=1
        )
        
        team_stats = team_performance.groupby('opposition').agg({
            'runs_scored': 'sum',
            'player_dismissed': lambda x: x.notna().sum()
        }).reset_index()
        
        team_stats.columns = ['opposition', 'runs_conceded', 'wickets']
        team_stats = team_stats.sort_values('wickets', ascending=False)
        
        fig = px.bar(team_stats, x='opposition', y='wickets', 
                    title=f"{player} - Wickets Against Teams")
        fig.update_layout(xaxis_title="Opposition", yaxis_title="Wickets")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_performance_trends(self, player):
        """Show performance trends over time"""
        st.subheader(f"📈 {player} - Performance Trends")
        
        # Get player data
        player_data = self.deliveries_df[
            (self.deliveries_df['batsman'] == player) | 
            (self.deliveries_df['bowler'] == player)
        ]
        
        if player_data.empty:
            st.warning(f"No data found for {player}")
            return
        
        # Merge with match data for dates
        player_data = player_data.merge(
            self.matches_df[['id', 'date', 'season']], 
            left_on='match_id', 
            right_on='id'
        )
        
        # Season-wise performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Batting trend
            batting_trend = player_data[player_data['batsman'] == player].groupby('season')['runs_scored'].sum().reset_index()
            if not batting_trend.empty:
                fig = px.line(batting_trend, x='season', y='runs_scored', 
                             title=f"{player} - Runs by Season")
                fig.update_layout(xaxis_title="Season", yaxis_title="Runs")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bowling trend
            bowling_trend = player_data[
                (player_data['bowler'] == player) & 
                (player_data['player_dismissed'].notna())
            ].groupby('season').size().reset_index(name='wickets')
            
            if not bowling_trend.empty:
                fig = px.line(bowling_trend, x='season', y='wickets', 
                             title=f"{player} - Wickets by Season")
                fig.update_layout(xaxis_title="Season", yaxis_title="Wickets")
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance consistency
        st.subheader("🎯 Performance Consistency")
        
        # Calculate match-wise consistency
        match_runs = player_data[player_data['batsman'] == player].groupby('match_id')['runs_scored'].sum()
        match_wickets = player_data[
            (player_data['bowler'] == player) & 
            (player_data['player_dismissed'].notna())
        ].groupby('match_id').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not match_runs.empty:
                st.metric("Batting Consistency (Std Dev)", f"{match_runs.std():.2f}")
                st.metric("Average Runs per Match", f"{match_runs.mean():.2f}")
        
        with col2:
            if not match_wickets.empty:
                st.metric("Bowling Consistency (Std Dev)", f"{match_wickets.std():.2f}")
                st.metric("Average Wickets per Match", f"{match_wickets.mean():.2f}")