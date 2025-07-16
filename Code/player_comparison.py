import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

class PlayerComparison:
    def __init__(self, matches_df, deliveries_df):
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data for player comparison"""
        try:
            # Get batting statistics
            self.batting_stats = self.get_batting_stats()
            self.bowling_stats = self.get_bowling_stats()
            
            # Get list of players
            batsmen = set(self.deliveries_df['batsman'].dropna().unique()) if 'batsman' in self.deliveries_df.columns else set()
            bowlers = set(self.deliveries_df['bowler'].dropna().unique()) if 'bowler' in self.deliveries_df.columns else set()
            
            self.all_players = sorted(list(batsmen.union(bowlers)))
            
        except Exception as e:
            st.error(f"Error preparing data: {e}")
            self.all_players = []
            self.batting_stats = pd.DataFrame()
            self.bowling_stats = pd.DataFrame()
    
    def get_batting_stats(self):
        """Calculate batting statistics for all players"""
        try:
            if 'batsman' not in self.deliveries_df.columns or 'runs_scored' not in self.deliveries_df.columns:
                return pd.DataFrame()
            
            # Group by batsman and calculate stats
            batting_stats = self.deliveries_df.groupby('batsman').agg({
                'runs_scored': ['sum', 'count', 'mean'],
                'match_id': 'nunique'
            }).reset_index()
            
            # Flatten column names
            batting_stats.columns = ['player', 'total_runs', 'balls_faced', 'avg_runs_per_ball', 'matches_played']
            
            # Calculate additional metrics
            batting_stats['strike_rate'] = (batting_stats['total_runs'] / batting_stats['balls_faced'] * 100).round(2)
            batting_stats['average'] = (batting_stats['total_runs'] / batting_stats['matches_played']).round(2)
            
            # Calculate boundaries
            if 'runs_scored' in self.deliveries_df.columns:
                # Count fours and sixes separately
                fours = self.deliveries_df[self.deliveries_df['runs_scored'] == 4].groupby('batsman').size()
                sixes = self.deliveries_df[self.deliveries_df['runs_scored'] == 6].groupby('batsman').size()
                
                # Create boundaries dataframe
                all_batsmen = fours.index.union(sixes.index)
                boundaries = pd.DataFrame({
                    'player': all_batsmen,
                    'fours': fours.reindex(all_batsmen, fill_value=0),
                    'sixes': sixes.reindex(all_batsmen, fill_value=0)
                }).reset_index(drop=True)
                
                batting_stats = batting_stats.merge(boundaries, on='player', how='left')
                batting_stats['fours'] = batting_stats['fours'].fillna(0)
                batting_stats['sixes'] = batting_stats['sixes'].fillna(0)
                        
            # Filter players with minimum criteria
            batting_stats = batting_stats[batting_stats['balls_faced'] >= 30]  # Minimum 30 balls
            
            return batting_stats.sort_values('total_runs', ascending=False)
            
        except Exception as e:
            st.error(f"Error calculating batting stats: {e}")
            return pd.DataFrame()
    
    def get_bowling_stats(self):
        """Calculate bowling statistics for all players"""
        try:
            if 'bowler' not in self.deliveries_df.columns:
                return pd.DataFrame()
            
            # Group by bowler and calculate stats
            bowling_stats = self.deliveries_df.groupby('bowler').agg({
                'runs_scored': 'sum',
                'match_id': 'nunique',
                'over': 'count'  # Using over as proxy for balls bowled
            }).reset_index()
            
            bowling_stats.columns = ['player', 'runs_conceded', 'matches_played', 'balls_bowled']
            
            # Calculate wickets if dismissal data is available
            if 'player_dismissed' in self.deliveries_df.columns:
                wickets = self.deliveries_df[self.deliveries_df['player_dismissed'].notna()].groupby('bowler').size().reset_index()
                wickets.columns = ['player', 'wickets']
                bowling_stats = bowling_stats.merge(wickets, on='player', how='left')
                bowling_stats['wickets'] = bowling_stats['wickets'].fillna(0)
            else:
                bowling_stats['wickets'] = 0
            
            # Calculate bowling metrics
            bowling_stats['overs_bowled'] = (bowling_stats['balls_bowled'] / 6).round(1)
            bowling_stats['economy'] = (bowling_stats['runs_conceded'] / bowling_stats['overs_bowled']).round(2)
            bowling_stats['average'] = np.where(bowling_stats['wickets'] > 0, 
                                               (bowling_stats['runs_conceded'] / bowling_stats['wickets']).round(2), 
                                               np.inf)
            bowling_stats['strike_rate'] = np.where(bowling_stats['wickets'] > 0, 
                                                   (bowling_stats['balls_bowled'] / bowling_stats['wickets']).round(2), 
                                                   np.inf)
            
            # Filter players with minimum criteria
            bowling_stats = bowling_stats[bowling_stats['overs_bowled'] >= 10]  # Minimum 10 overs
            
            return bowling_stats.sort_values('wickets', ascending=False)
            
        except Exception as e:
            st.error(f"Error calculating bowling stats: {e}")
            return pd.DataFrame()
    
    def get_head_to_head_stats(self, batsman, bowler):
        """Get head-to-head statistics between batsman and bowler"""
        try:
            if 'batsman' not in self.deliveries_df.columns or 'bowler' not in self.deliveries_df.columns:
                return None
            
            h2h_data = self.deliveries_df[
                (self.deliveries_df['batsman'] == batsman) & 
                (self.deliveries_df['bowler'] == bowler)
            ]
            
            if h2h_data.empty:
                return None
            
            stats = {
                'balls_faced': len(h2h_data),
                'runs_scored': h2h_data['runs_scored'].sum() if 'runs_scored' in h2h_data.columns else 0,
                'dismissals': 0,
                'dot_balls': 0,
                'boundaries': 0
            }
            
            if 'runs_scored' in h2h_data.columns:
                stats['dot_balls'] = len(h2h_data[h2h_data['runs_scored'] == 0])
                stats['boundaries'] = len(h2h_data[h2h_data['runs_scored'].isin([4, 6])])
            
            if 'player_dismissed' in h2h_data.columns:
                stats['dismissals'] = h2h_data['player_dismissed'].notna().sum()
            
            stats['strike_rate'] = (stats['runs_scored'] / stats['balls_faced'] * 100).round(2) if stats['balls_faced'] > 0 else 0
            
            return stats
            
        except Exception as e:
            st.error(f"Error calculating head-to-head stats: {e}")
            return None
    
    def create_radar_chart(self, players_data, metrics, title):
        """Create radar chart for player comparison"""
        fig = go.Figure()
        
        colors = ['#FF6B35', '#004E98', '#7B68EE', '#FF1493', '#32CD32']
        
        for i, (player, data) in enumerate(players_data.items()):
            values = [data.get(metric, 0) for metric in metrics]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=player,
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                opacity=0.3
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max([max(data.values()) for data in players_data.values()])]
                )
            ),
            showlegend=True,
            title=title,
            height=500
        )
        
        return fig
    
    def show_comparison(self):
        """Main function to display player comparison"""
        st.title("🆚 Player Comparison")
        
        if not self.all_players:
            st.error("No player data available for comparison.")
            return
        
        # Sidebar for player selection
        st.sidebar.subheader("Select Players")
        
        # Multi-select for players
        selected_players = st.sidebar.multiselect(
            "Choose players to compare (max 5)",
            self.all_players,
            max_selections=5
        )
        
        if not selected_players:
            st.info("Please select at least 2 players to compare.")
            return
        
        if len(selected_players) < 2:
            st.warning("Please select at least 2 players for comparison.")
            return
        
        # Comparison type
        comparison_type = st.sidebar.radio(
            "Comparison Type",
            ["Batting Stats", "Bowling Stats", "Head-to-Head", "Overall Performance"]
        )
        
        # Display comparison based on type
        if comparison_type == "Batting Stats":
            self.show_batting_comparison(selected_players)
        elif comparison_type == "Bowling Stats":
            self.show_bowling_comparison(selected_players)
        elif comparison_type == "Head-to-Head":
            self.show_head_to_head_comparison(selected_players)
        else:
            self.show_overall_comparison(selected_players)
    
    def show_batting_comparison(self, players):
        """Show batting statistics comparison"""
        st.subheader("🏏 Batting Statistics Comparison")
        
        if self.batting_stats.empty:
            st.error("No batting statistics available.")
            return
        
        # Filter data for selected players
        player_batting_stats = self.batting_stats[self.batting_stats['player'].isin(players)]
        
        if player_batting_stats.empty:
            st.warning("No batting data available for selected players.")
            return
        
        # Display stats table
        st.subheader("📊 Batting Statistics Table")
        display_cols = ['player', 'total_runs', 'balls_faced', 'average', 'strike_rate', 'fours', 'sixes', 'matches_played']
        available_cols = [col for col in display_cols if col in player_batting_stats.columns]
        st.dataframe(player_batting_stats[available_cols])
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Runs comparison
            fig = px.bar(
                player_batting_stats,
                x='player',
                y='total_runs',
                title="Total Runs Comparison",
                color='player'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Strike rate comparison
            fig = px.bar(
                player_batting_stats,
                x='player',
                y='strike_rate',
                title="Strike Rate Comparison",
                color='player'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart for batting metrics
        if len(player_batting_stats) >= 2:
            st.subheader("🎯 Batting Performance Radar")
            
            # Prepare data for radar chart
            metrics = ['total_runs', 'average', 'strike_rate', 'fours', 'sixes']
            available_metrics = [m for m in metrics if m in player_batting_stats.columns]
            
            if available_metrics:
                # Normalize metrics for radar chart
                normalized_data = {}
                for _, row in player_batting_stats.iterrows():
                    normalized_data[row['player']] = {}
                    for metric in available_metrics:
                        max_val = player_batting_stats[metric].max()
                        if max_val > 0:
                            normalized_data[row['player']][metric] = (row[metric] / max_val) * 100
                        else:
                            normalized_data[row['player']][metric] = 0
                
                radar_fig = self.create_radar_chart(normalized_data, available_metrics, "Batting Performance Comparison")
                st.plotly_chart(radar_fig, use_container_width=True)
    
    def show_bowling_comparison(self, players):
        """Show bowling statistics comparison"""
        st.subheader("🎳 Bowling Statistics Comparison")
        
        if self.bowling_stats.empty:
            st.error("No bowling statistics available.")
            return
        
        # Filter data for selected players
        player_bowling_stats = self.bowling_stats[self.bowling_stats['player'].isin(players)]
        
        if player_bowling_stats.empty:
            st.warning("No bowling data available for selected players.")
            return
        
        # Display stats table
        st.subheader("📊 Bowling Statistics Table")
        display_cols = ['player', 'wickets', 'overs_bowled', 'runs_conceded', 'economy', 'average', 'strike_rate', 'matches_played']
        available_cols = [col for col in display_cols if col in player_bowling_stats.columns]
        st.dataframe(player_bowling_stats[available_cols])
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Wickets comparison
            fig = px.bar(
                player_bowling_stats,
                x='player',
                y='wickets',
                title="Wickets Comparison",
                color='player'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Economy rate comparison
            fig = px.bar(
                player_bowling_stats,
                x='player',
                y='economy',
                title="Economy Rate Comparison",
                color='player'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def show_head_to_head_comparison(self, players):
        """Show head-to-head comparison between players"""
        st.subheader("⚔️ Head-to-Head Analysis")
        
        if len(players) != 2:
            st.warning("Please select exactly 2 players for head-to-head comparison.")
            return
        
        player1, player2 = players[0], players[1]
        
        # Check if one is primarily batsman and other is bowler
        batsman_in_data = player1 in self.batting_stats['player'].values if not self.batting_stats.empty else False
        bowler_in_data = player2 in self.bowling_stats['player'].values if not self.bowling_stats.empty else False
        
        if batsman_in_data and bowler_in_data:
            # Get head-to-head stats
            h2h_stats = self.get_head_to_head_stats(player1, player2)
            
            if h2h_stats:
                st.subheader(f"📊 {player1} vs {player2}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Balls Faced", h2h_stats['balls_faced'])
                
                with col2:
                    st.metric("Runs Scored", h2h_stats['runs_scored'])
                
                with col3:
                    st.metric("Strike Rate", f"{h2h_stats['strike_rate']:.2f}")
                
                with col4:
                    st.metric("Dismissals", h2h_stats['dismissals'])
                
                # Visualization
                fig = go.Figure()
                
                categories = ['Dot Balls', 'Boundaries', 'Dismissals', 'Other Runs']
                values = [
                    h2h_stats['dot_balls'],
                    h2h_stats['boundaries'],
                    h2h_stats['dismissals'],
                    h2h_stats['balls_faced'] - h2h_stats['dot_balls'] - h2h_stats['boundaries'] - h2h_stats['dismissals']
                ]
                
                fig.add_trace(go.Pie(
                    labels=categories,
                    values=values,
                    title=f"{player1} vs {player2} - Ball Outcome Distribution"
                ))
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No head-to-head data available between {player1} and {player2}.")
        else:
            st.info("Head-to-head analysis requires one batsman and one bowler.")
    
    def show_overall_comparison(self, players):
        """Show overall performance comparison"""
        st.subheader("🏆 Overall Performance Comparison")
        
        # Create tabs for different aspects
        tab1, tab2, tab3 = st.tabs(["Performance Summary", "Match Impact", "Consistency"])
        
        with tab1:
            self.show_performance_summary(players)
        
        with tab2:
            self.show_match_impact(players)
        
        with tab3:
            self.show_consistency_analysis(players)
    
    def show_performance_summary(self, players):
        """Show performance summary for selected players"""
        st.subheader("📈 Performance Summary")
        
        summary_data = []
        
        for player in players:
            player_data = {'Player': player}
            
            # Batting stats
            if not self.batting_stats.empty and player in self.batting_stats['player'].values:
                batting_row = self.batting_stats[self.batting_stats['player'] == player].iloc[0]
                player_data.update({
                    'Batting_Runs': batting_row.get('total_runs', 0),
                    'Batting_SR': batting_row.get('strike_rate', 0),
                    'Batting_Avg': batting_row.get('average', 0)
                })
            else:
                player_data.update({
                    'Batting_Runs': 0,
                    'Batting_SR': 0,
                    'Batting_Avg': 0
                })
            
            # Bowling stats
            if not self.bowling_stats.empty and player in self.bowling_stats['player'].values:
                bowling_row = self.bowling_stats[self.bowling_stats['player'] == player].iloc[0]
                player_data.update({
                    'Bowling_Wickets': bowling_row.get('wickets', 0),
                    'Bowling_Economy': bowling_row.get('economy', 0),
                    'Bowling_Avg': bowling_row.get('average', 0)
                })
            else:
                player_data.update({
                    'Bowling_Wickets': 0,
                    'Bowling_Economy': 0,
                    'Bowling_Avg': 0
                })
            
            summary_data.append(player_data)
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
    
    def show_match_impact(self, players):
        """Show match impact analysis"""
        st.subheader("🎯 Match Impact Analysis")
        
        # This would require more detailed match-by-match analysis
        # For now, showing basic impact metrics
        
        impact_data = []
        
        for player in players:
            if not self.batting_stats.empty and player in self.batting_stats['player'].values:
                batting_row = self.batting_stats[self.batting_stats['player'] == player].iloc[0]
                matches = batting_row.get('matches_played', 0)
                runs_per_match = batting_row.get('total_runs', 0) / matches if matches > 0 else 0
                
                impact_data.append({
                    'Player': player,
                    'Matches': matches,
                    'Runs_Per_Match': runs_per_match,
                    'Type': 'Batting'
                })
        
        if impact_data:
            impact_df = pd.DataFrame(impact_data)
            
            fig = px.scatter(
                impact_df,
                x='Matches',
                y='Runs_Per_Match',
                color='Player',
                size='Runs_Per_Match',
                title="Match Impact: Runs per Match vs Matches Played"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def show_consistency_analysis(self, players):
        """Show consistency analysis"""
        st.subheader("📊 Consistency Analysis")
        
        st.info("Consistency analysis would require match-by-match performance data.")
        st.write("This feature can be enhanced with more detailed match-wise statistics.")