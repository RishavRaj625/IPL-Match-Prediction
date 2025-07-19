# 🏏 IPL Analytics Dashboard

A comprehensive web application for analyzing Indian Premier League (IPL) cricket data, built with Streamlit and featuring interactive visualizations and deep statistical insights.

## 🌟 Features

### 📊 Core Analytics
- **Player Profiles**: Detailed player information including batting styles, bowling styles, and career statistics
- **Player Analysis**: Deep dive into individual player performance with batting and bowling metrics
- **Player Comparison**: Side-by-side comparison of two players with comprehensive statistics
- **Team vs Team**: Head-to-head records and performance analysis between teams
- **Venue Analysis**: Impact of different venues on match outcomes and player performance
- **Form Tracker**: Recent performance trends for players and teams
- **Match Prediction**: Predictive analytics for upcoming matches

### 🎯 Interactive Features
- Real-time filtering and data selection
- Interactive charts and visualizations
- Responsive design for all devices
- Dynamic metric calculations
- Comprehensive statistical breakdowns

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ipl-analytics-dashboard.git
   cd ipl-analytics-dashboard
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data files**
   - Create a `CSV` folder in the project root
   - Place your IPL data files in the `CSV` folder:
     - `matches.csv` - Match information
     - `deliveries.csv` - Ball-by-ball data
     - `2024_players_details.csv` - Player details

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 📁 Project Structure

```
ipl-analytics-dashboard/
│
├── main.py                     # Main dashboard application
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
│
├── Code/                      # Feature modules
│   ├── player_analysis.py     # Player performance analysis
│   ├── match_prediction.py    # Match prediction algorithms
│   ├── team_comparison.py     # Team vs team analysis
│   ├── venue_analysis.py      # Venue impact analysis
│   ├── player_comparison.py   # Player comparison tools
│   ├── profile_comparison.py  # Player profile management
│   └── form_tracker.py        # Form and trend tracking
│
├── CSV/                       # Data files
│   ├── matches.csv            # Match data
│   ├── deliveries.csv         # Ball-by-ball data
│   └── 2024_players_details.csv # Player information
│
├── assets/                    # Static assets (optional)
│   ├── images/               # Screenshots and logos
│   └── styles/               # Additional CSS files
│
└── docs/                     # Documentation
    ├── data_schema.md        # Data structure documentation
    ├── api_reference.md      # Module API documentation
    └── user_guide.md         # User guide
```

## 📊 Data Requirements

### Expected Data Format

#### matches.csv
| Column | Description | Type |
|--------|-------------|------|
| match_number | Unique match identifier | Integer |
| match_date | Date of the match | Date |
| team1 | First team name | String |
| team2 | Second team name | String |
| venue | Match venue | String |
| city | City where match was played | String |
| winner | Winning team | String |
| toss_winner | Toss winning team | String |
| toss_decision | Toss decision (bat/field) | String |
| player_of_match | Player of the match | String |

#### deliveries.csv
| Column | Description | Type |
|--------|-------------|------|
| ID | Match ID | Integer |
| Innings | Innings number | Integer |
| Overs | Over number | Integer |
| BallNumber | Ball number in over | Integer |
| Batter | Batsman name | String |
| Bowler | Bowler name | String |
| NonStriker | Non-striker batsman | String |
| BattingTeam | Batting team name | String |
| BatsmanRun | Runs scored by batsman | Integer |
| ExtrasRun | Extra runs | Integer |
| TotalRun | Total runs on delivery | Integer |
| IsWicketDelivery | Whether wicket fell | Boolean |
| PlayerOut | Player dismissed | String |
| Kind | Type of dismissal | String |

#### 2024_players_details.csv
| Column | Description | Type |
|--------|-------------|------|
| ID | Player ID | Integer |
| Name | Player name | String |
| longName | Full player name | String |
| dob | Date of birth | Date |
| battingStyles | Batting style | String |
| bowlingStyles | Bowling style | String |
| playingRoles | Playing role | String |
| espn_url | ESPN profile URL | String |

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly Express, Plotly Graph Objects
- **Styling**: Custom CSS with Streamlit
- **Data Analysis**: Python statistical libraries

## 🎨 Features Overview

### 📋 Player Profiles
- Comprehensive player information
- Batting and bowling statistics
- Career progression analysis
- Performance metrics across seasons

### 🆚 Player Comparison
- Side-by-side statistical comparison
- Performance trend analysis
- Head-to-head records
- Visualization of key metrics

### 🏟️ Venue Analysis
- Venue-specific performance metrics
- Home advantage analysis
- Pitch condition impact
- Historical venue statistics

### 📈 Form Tracker
- Recent performance trends
- Player form indicators
- Team momentum analysis
- Match-by-match progression

## 🚀 Usage Examples

### Running Player Analysis
```python
# Navigate to Player Analysis section
# Select player from dropdown
# View comprehensive statistics and visualizations
```

### Comparing Players
```python
# Go to Player Comparison section
# Select two players for comparison
# Analyze side-by-side statistics
```

### Venue Analysis
```python
# Access Venue Analysis section
# Select venue from dropdown
# Explore venue-specific insights
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for any changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IPL for providing the data
- Streamlit community for the excellent framework
- Plotly team for interactive visualizations
- Cricket analytics community for insights and inspiration

## 📧 Contact

- **Developer**: Rishav
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Project Link**: https://ipl-match-prediction-dash.streamlit.app/

## 🐛 Bug Reports & Feature Requests

Please use the [GitHub Issues](https://github.com/yourusername/ipl-analytics-dashboard/issues) page to report bugs or request new features.


## 🔄 Version History

- **v1.0.0** - Initial release with core features
- **v1.1.0** - Added player comparison functionality
- **v1.2.0** - Venue analysis and form tracker
- **v1.3.0** - Enhanced visualizations and UI improvements

## 📈 Roadmap

- [ ] Add powerplay analysis
- [ ] Implement top performers section
- [ ] Add fantasy cricket insights
- [ ] Mobile app version
- [ ] Real-time data integration
- [ ] Advanced machine learning predictions

## 🎯 Quick Start Guide

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Add your data**: Place CSV files in the `CSV` folder
3. **Run the app**: `streamlit run main.py`
4. **Open browser**: Navigate to `http://localhost:8501`
5. **Start analyzing**: Use the sidebar to navigate between sections

---

**Made with ❤️ for cricket analytics enthusiasts**
