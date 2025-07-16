import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VenueAnalysis:
    def __init__(self, matches_df, deliveries_df):
        self.matches_df = matches_df
        self.deliveries_df = deliveries_df
        self.prepare_venue_data()
    
    def prepare_venue_data(self):
        """Prepare venue-specific data for analysis"""
        try:
            # Calculate innings scores
            self.innings_scores = self.calculate_innings_scores()
            
            # Calculate venue statistics
            self.venue_stats = self.calculate_venue_stats()
            
            # Calculate chasing statistics
            self.chasing_stats = self.calculate_chasing_stats()
            
            # Calculate dew factor impact
            self.dew_impact = self.calculate_dew_impact()
            
            # Calculate pitch conditions
            self.pitch_conditions = self.calculate_pitch_conditions()
            
        except Exception as e:
            st.error(f"Error preparing venue data: {e}")
            # Create dummy data
            self.create_dummy_data()
    
    def calculate_innings_scores(self):
        """Calculate innings scores for each match"""
        try:
            # Group by match_id and inning to get total runs
            innings_scores = self.deliveries_df.groupby(['match_id', 'inning']).agg({
                'runs_scored': 'sum',
                'extras': 'sum'
            }).reset_index()
            
            # Calculate total score
            innings_scores['total_score'] = innings_scores['runs_scored'] + innings_scores['extras']
            
            # Merge with match data to get venue
            innings_scores = innings_scores.merge(
                self.matches_df[['id', 'venue', 'date', 'winner', 'season']], 
                left_on='match_id', 
                right_on='id', 
                how='left'
            )
            
            return innings_scores
        except Exception as e:
            st.error(f"Error calculating innings scores: {e}")
            return pd.DataFrame()
    
    def calculate_venue_stats(self):
        """Calculate basic venue statistics"""
        try:
            venue_stats = []
            
            for venue in self.matches_df['venue'].unique():
                venue_matches = self.matches_df[self.matches_df['venue'] == venue]
                venue_deliveries = self.deliveries_df[
                    self.deliveries_df['match_id'].isin(venue_matches['id'])
                ]
                
                # Basic stats
                total_matches = len(venue_matches)
                total_runs = venue_deliveries['runs_scored'].sum() if 'runs_scored' in venue_deliveries.columns else 0
                total_balls = len(venue_deliveries)
                
                # Average score calculation
                venue_innings = self.innings_scores[self.innings_scores['venue'] == venue]
                avg_first_innings = venue_innings[venue_innings['inning'] == 1]['total_score'].mean()
                avg_second_innings = venue_innings[venue_innings['inning'] == 2]['total_score'].mean()
                
                venue_stats.append({
                    'venue': venue,
                    'total_matches': total_matches,
                    'total_runs': total_runs,
                    'total_balls': total_balls,
                    'avg_first_innings': avg_first_innings or 0,
                    'avg_second_innings': avg_second_innings or 0,
                    'run_rate': (total_runs / total_balls * 6) if total_balls > 0 else 0
                })
            
            return pd.DataFrame(venue_stats)
        except Exception as e:
            st.error(f"Error calculating venue stats: {e}")
            return pd.DataFrame()
    
    def calculate_chasing_stats(self):
        """Calculate chasing success rates for each venue"""
        try:
            chasing_stats = []
            
            for venue in self.matches_df['venue'].unique():
                venue_matches = self.matches_df[self.matches_df['venue'] == venue]
                
                # Get matches where team batting second won
                total_matches = len(venue_matches)
                
                # Calculate chasing success
                # Assuming team2 bats second (you may need to adjust based on your data structure)
                chasing_wins = 0
                for _, match in venue_matches.iterrows():
                    # Get first innings score
                    first_innings = self.innings_scores[
                        (self.innings_scores['match_id'] == match['id']) & 
                        (self.innings_scores['inning'] == 1)
                    ]
                    
                    if not first_innings.empty and match['winner'] in [match.get('team1', ''), match.get('team2', '')]:
                        # Check if winner batted second (simplified logic)
                        if len(first_innings) > 0:
                            chasing_wins += 1
                
                chasing_success_rate = (chasing_wins / total_matches * 100) if total_matches > 0 else 0
                
                chasing_stats.append({
                    'venue': venue,
                    'total_matches': total_matches,
                    'chasing_wins': chasing_wins,
                    'chasing_success_rate': chasing_success_rate
                })
            
            return pd.DataFrame(chasing_stats)
        except Exception as e:
            st.error(f"Error calculating chasing stats: {e}")
            return pd.DataFrame()
    
    def calculate_dew_impact(self):
        """Calculate dew factor impact (day vs night matches)"""
        try:
            dew_impact = []
            
            for venue in self.matches_df['venue'].unique():
                venue_matches = self.matches_df[self.matches_df['venue'] == venue]
                
                # Simulate day/night matches based on match timing
                # In real scenario, you'd have actual timing data
                total_matches = len(venue_matches)
                
                # Simulate dew impact (you'll need actual timing data)
                # For now, we'll create a reasonable simulation
                night_matches = int(total_matches * 0.7)  # Assume 70% are night matches
                day_matches = total_matches - night_matches
                
                # Calculate average scores for day vs night
                venue_innings = self.innings_scores[self.innings_scores['venue'] == venue]
                
                if len(venue_innings) > 0:
                    # Simulate day/night split
                    night_innings = venue_innings.sample(n=min(night_matches, len(venue_innings)))
                    day_innings = venue_innings.drop(night_innings.index)
                    
                    night_avg_score = night_innings['total_score'].mean() if len(night_innings) > 0 else 0
                    day_avg_score = day_innings['total_score'].mean() if len(day_innings) > 0 else 0
                    
                    dew_impact.append({
                        'venue': venue,
                        'night_matches': night_matches,
                        'day_matches': day_matches,
                        'night_avg_score': night_avg_score,
                        'day_avg_score': day_avg_score,
                        'dew_impact_score': night_avg_score - day_avg_score
                    })
            
            return pd.DataFrame(dew_impact)
        except Exception as e:
            st.error(f"Error calculating dew impact: {e}")
            return pd.DataFrame()
    
    def calculate_pitch_conditions(self):
        """Calculate pitch conditions analysis"""
        try:
            pitch_conditions = []
            
            for venue in self.matches_df['venue'].unique():
                venue_matches = self.matches_df[self.matches_df['venue'] == venue]
                venue_deliveries = self.deliveries_df[
                    self.deliveries_df['match_id'].isin(venue_matches['id'])
                ]
                
                if len(venue_deliveries) == 0:
                    continue
                
                # Calculate batting-friendly vs bowling-friendly conditions
                total_runs = venue_deliveries['runs_scored'].sum() if 'runs_scored' in venue_deliveries.columns else 0
                total_balls = len(venue_deliveries)
                
                # Boundary percentage
                boundaries = len(venue_deliveries[venue_deliveries['runs_scored'].isin([4, 6])]) if 'runs_scored' in venue_deliveries.columns else 0
                boundary_percentage = (boundaries / total_balls * 100) if total_balls > 0 else 0
                
                # Dot ball percentage
                dot_balls = len(venue_deliveries[venue_deliveries['runs_scored'] == 0]) if 'runs_scored' in venue_deliveries.columns else 0
                dot_ball_percentage = (dot_balls / total_balls * 100) if total_balls > 0 else 0
                
                # Wickets rate
                wickets = len(venue_deliveries[venue_deliveries['player_dismissed'].notna()]) if 'player_dismissed' in venue_deliveries.columns else 0
                wickets_per_match = wickets / len(venue_matches) if len(venue_matches) > 0 else 0
                
                # Classify pitch type
                run_rate = (total_runs / total_balls * 6) if total_balls > 0 else 0
                
                if run_rate > 8.5:
                    pitch_type = "Batting Paradise"
                elif run_rate > 7.5:
                    pitch_type = "Batting Friendly"
                elif run_rate > 6.5:
                    pitch_type = "Balanced"
                elif run_rate > 5.5:
                    pitch_type = "Bowling Friendly"
                else:
                    pitch_type = "Bowling Paradise"
                
                pitch_conditions.append({
                    'venue': venue,
                    'run_rate': run_rate,
                    'boundary_percentage': boundary_percentage,
                    'dot_ball_percentage': dot_ball_percentage,
                    'wickets_per_match': wickets_per_match,
                    'pitch_type': pitch_type
                })
            
            return pd.DataFrame(pitch_conditions)
        except Exception as e:
            st.error(f"Error calculating pitch conditions: {e}")
            return pd.DataFrame()
    
    def create_dummy_data(self):
        """Create dummy data for demonstration"""
        venues = ['Wankhede Stadium', 'Eden Gardens', 'M. Chinnaswamy Stadium', 
                 'Feroz Shah Kotla', 'Chepauk Stadium']
        
        # Dummy venue stats
        self.venue_stats = pd.DataFrame({
            'venue': venues,
            'total_matches': [25, 20, 22, 18, 15],
            'avg_first_innings': [165, 155, 175, 150, 145],
            'avg_second_innings': [155, 145, 165, 140, 135],
            'run_rate': [8.2, 7.8, 8.5, 7.5, 7.2]
        })
        
        # Dummy chasing stats
        self.chasing_stats = pd.DataFrame({
            'venue': venues,
            'total_matches': [25, 20, 22, 18, 15],
            'chasing_wins': [12, 8, 10, 7, 6],
            'chasing_success_rate': [48, 40, 45, 39, 40]
        })
        
        # Dummy dew impact
        self.dew_impact = pd.DataFrame({
            'venue': venues,
            'night_avg_score': [170, 160, 180, 155, 150],
            'day_avg_score': [160, 150, 170, 145, 140],
            'dew_impact_score': [10, 10, 10, 10, 10]
        })
        
        # Dummy pitch conditions
        self.pitch_conditions = pd.DataFrame({
            'venue': venues,
            'run_rate': [8.2, 7.8, 8.5, 7.5, 7.2],
            'boundary_percentage': [18, 16, 20, 15, 14],
            'dot_ball_percentage': [35, 38, 32, 40, 42],
            'wickets_per_match': [12, 13, 11, 14, 15],
            'pitch_type': ['Batting Friendly', 'Balanced', 'Batting Paradise', 'Bowling Friendly', 'Bowling Friendly']
        })
    
    def show_analysis(self):
        """Main function to display venue analysis"""
        st.title("🏟️ Venue Analysis")
        
        # Venue selector
        venues = self.venue_stats['venue'].unique() if not self.venue_stats.empty else []
        selected_venue = st.selectbox("Select Venue for Detailed Analysis", 
                                    ['All Venues'] + list(venues))
        
        # Overview metrics
        st.subheader("📊 Venue Overview")
        
        if not self.venue_stats.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_venues = len(self.venue_stats)
                st.metric("Total Venues", total_venues)
            
            with col2:
                avg_first_innings = self.venue_stats['avg_first_innings'].mean()
                st.metric("Avg First Innings", f"{avg_first_innings:.1f}")
            
            with col3:
                overall_run_rate = self.venue_stats['run_rate'].mean()
                st.metric("Overall Run Rate", f"{overall_run_rate:.2f}")
            
            with col4:
                chasing_success = self.chasing_stats['chasing_success_rate'].mean()
                st.metric("Avg Chasing Success", f"{chasing_success:.1f}%")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Innings Scores", "🎯 Chasing Analysis", "🌙 Dew Factor", "🏟️ Pitch Conditions"])
        
        with tab1:
            self.show_innings_analysis(selected_venue)
        
        with tab2:
            self.show_chasing_analysis(selected_venue)
        
        with tab3:
            self.show_dew_analysis(selected_venue)
        
        with tab4:
            self.show_pitch_analysis(selected_venue)
    
    def show_innings_analysis(self, selected_venue):
        """Show innings score analysis"""
        st.subheader("📈 Average First Innings Scores")
        
        if self.venue_stats.empty:
            st.warning("No data available for innings analysis")
            return
        
        # Filter data if specific venue selected
        if selected_venue != 'All Venues':
            display_data = self.venue_stats[self.venue_stats['venue'] == selected_venue]
        else:
            display_data = self.venue_stats
        
        # First innings scores comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='First Innings Average',
            x=display_data['venue'],
            y=display_data['avg_first_innings'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Second Innings Average',
            x=display_data['venue'],
            y=display_data['avg_second_innings'],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Average Innings Scores by Venue",
            xaxis_title="Venue",
            yaxis_title="Average Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Detailed Innings Statistics")
        
        display_table = display_data[['venue', 'total_matches', 'avg_first_innings', 
                                    'avg_second_innings', 'run_rate']].copy()
        display_table.columns = ['Venue', 'Matches', 'Avg 1st Inn', 'Avg 2nd Inn', 'Run Rate']
        
        st.dataframe(display_table, use_container_width=True)
    
    def show_chasing_analysis(self, selected_venue):
        """Show chasing success analysis"""
        st.subheader("🎯 Chasing Success Rate Analysis")
        
        if self.chasing_stats.empty:
            st.warning("No data available for chasing analysis")
            return
        
        # Filter data if specific venue selected
        if selected_venue != 'All Venues':
            display_data = self.chasing_stats[self.chasing_stats['venue'] == selected_venue]
        else:
            display_data = self.chasing_stats
        
        # Chasing success rate chart
        fig = px.bar(
            display_data,
            x='venue',
            y='chasing_success_rate',
            title="Chasing Success Rate by Venue",
            labels={'chasing_success_rate': 'Success Rate (%)', 'venue': 'Venue'},
            color='chasing_success_rate',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart for overall chasing performance
        if selected_venue == 'All Venues':
            total_matches = display_data['total_matches'].sum()
            total_chasing_wins = display_data['chasing_wins'].sum()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Defending Wins', 'Chasing Wins'],
                values=[total_matches - total_chasing_wins, total_chasing_wins],
                hole=0.3
            )])
            
            fig_pie.update_layout(title="Overall Defending vs Chasing Success")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed table
        st.subheader("📋 Chasing Statistics")
        display_table = display_data[['venue', 'total_matches', 'chasing_wins', 'chasing_success_rate']].copy()
        display_table.columns = ['Venue', 'Total Matches', 'Chasing Wins', 'Success Rate (%)']
        
        st.dataframe(display_table, use_container_width=True)
    
    def show_dew_analysis(self, selected_venue):
        """Show dew factor analysis"""
        st.subheader("🌙 Dew Factor Impact Analysis")
        
        if self.dew_impact.empty:
            st.warning("No data available for dew analysis")
            return
        
        # Filter data if specific venue selected
        if selected_venue != 'All Venues':
            display_data = self.dew_impact[self.dew_impact['venue'] == selected_venue]
        else:
            display_data = self.dew_impact
        
        # Day vs Night comparison
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Night Matches',
            x=display_data['venue'],
            y=display_data['night_avg_score'],
            marker_color='darkblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Day Matches',
            x=display_data['venue'],
            y=display_data['day_avg_score'],
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Day vs Night Average Scores by Venue",
            xaxis_title="Venue",
            yaxis_title="Average Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dew impact score
        fig_impact = px.bar(
            display_data,
            x='venue',
            y='dew_impact_score',
            title="Dew Impact Score by Venue",
            labels={'dew_impact_score': 'Impact Score (Night - Day)', 'venue': 'Venue'},
            color='dew_impact_score',
            color_continuous_scale='RdBu'
        )
        
        fig_impact.update_layout(height=400)
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Insights
        st.subheader("🔍 Dew Factor Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not display_data.empty:
                highest_dew_impact = display_data.loc[display_data['dew_impact_score'].idxmax()]
                st.info(f"**Highest Dew Impact:** {highest_dew_impact['venue']}")
                st.write(f"Impact Score: {highest_dew_impact['dew_impact_score']:.1f}")
        
        with col2:
            if not display_data.empty:
                lowest_dew_impact = display_data.loc[display_data['dew_impact_score'].idxmin()]
                st.info(f"**Lowest Dew Impact:** {lowest_dew_impact['venue']}")
                st.write(f"Impact Score: {lowest_dew_impact['dew_impact_score']:.1f}")
    
    def show_pitch_analysis(self, selected_venue):
        """Show pitch conditions analysis"""
        st.subheader("🏟️ Pitch Conditions Analysis")
        
        if self.pitch_conditions.empty:
            st.warning("No data available for pitch analysis")
            return
        
        # Filter data if specific venue selected
        if selected_venue != 'All Venues':
            display_data = self.pitch_conditions[self.pitch_conditions['venue'] == selected_venue]
        else:
            display_data = self.pitch_conditions
        
        # Pitch type distribution
        if selected_venue == 'All Venues':
            pitch_type_counts = display_data['pitch_type'].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=pitch_type_counts.index,
                values=pitch_type_counts.values,
                hole=0.3
            )])
            
            fig_pie.update_layout(title="Pitch Type Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Multi-metric comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Run Rate', 'Boundary %', 'Dot Ball %', 'Wickets/Match'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Run rate
        fig.add_trace(
            go.Bar(x=display_data['venue'], y=display_data['run_rate'], name='Run Rate'),
            row=1, col=1
        )
        
        # Boundary percentage
        fig.add_trace(
            go.Bar(x=display_data['venue'], y=display_data['boundary_percentage'], name='Boundary %'),
            row=1, col=2
        )
        
        # Dot ball percentage
        fig.add_trace(
            go.Bar(x=display_data['venue'], y=display_data['dot_ball_percentage'], name='Dot Ball %'),
            row=2, col=1
        )
        
        # Wickets per match
        fig.add_trace(
            go.Bar(x=display_data['venue'], y=display_data['wickets_per_match'], name='Wickets/Match'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Pitch Conditions Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed pitch analysis table
        st.subheader("📋 Pitch Conditions Summary")
        
        display_table = display_data[['venue', 'pitch_type', 'run_rate', 'boundary_percentage', 
                                    'dot_ball_percentage', 'wickets_per_match']].copy()
        display_table.columns = ['Venue', 'Pitch Type', 'Run Rate', 'Boundary %', 'Dot Ball %', 'Wickets/Match']
        
        # Add color coding for pitch types
        def color_pitch_type(val):
            if val == 'Batting Paradise':
                return 'background-color: #90EE90'
            elif val == 'Batting Friendly':
                return 'background-color: #98FB98'
            elif val == 'Balanced':
                return 'background-color: #FFFFE0'
            elif val == 'Bowling Friendly':
                return 'background-color: #FFA07A'
            elif val == 'Bowling Paradise':
                return 'background-color: #FF6347'
            return ''
        
        styled_table = display_table.style.applymap(color_pitch_type, subset=['Pitch Type'])
        st.dataframe(styled_table, use_container_width=True)
        
        # Pitch recommendations
        st.subheader("💡 Pitch Recommendations")
        
        if not display_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                batting_venues = display_data[display_data['pitch_type'].isin(['Batting Paradise', 'Batting Friendly'])]
                if not batting_venues.empty:
                    st.success("**Best Batting Venues:**")
                    for venue in batting_venues['venue'].tolist():
                        st.write(f"• {venue}")
            
            with col2:
                bowling_venues = display_data[display_data['pitch_type'].isin(['Bowling Paradise', 'Bowling Friendly'])]
                if not bowling_venues.empty:
                    st.success("**Best Bowling Venues:**")
                    for venue in bowling_venues['venue'].tolist():
                        st.write(f"• {venue}")