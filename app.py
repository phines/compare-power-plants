import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import time

# Set page config
st.set_page_config(
    page_title="Solar vs Nuclear Power Cost Comparison",
    page_icon="⚡",
    layout="wide"
)

# Add custom CSS for tabs
st.markdown("""
    <style>
        .stTabs {
            margin-top: 32px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            padding-left: 24px;
            padding-right: 24px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-bottom: 2px solid #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
FACILITY_SIZE_MW = 600
YEARS = 40
HOURS_PER_YEAR = 8760

# Optimization parameters
SOLAR_CAPACITY_STEP = 100  # MW
BATTERY_CAPACITY_STEP = 100  # MWh
MAX_ITERATIONS = 100000  # Not sure how many needed.

# Location mapping for solar sites
LOCATION_NAMES = {
    (35.85, -115.15): "Las Vegas",
    (36.85, -116.15): "Beatty",
    (37.85, -117.15): "Tonopah",
    (38.85, -118.15): "Hawthorne",
    (39.85, -119.15): "Reno"
}

# Default parameters with detailed citations
DEFAULT_PARAMETERS = {
    # Common parameters
    'discount_rate_pct': {
        'value': 6.0,
        'min': 3.0,
        'max': 15.0,
        'step': 0.5,
        'citation': "U.S. Energy Information Administration (EIA) Annual Energy Outlook 2023, Table 8.2. Electricity Generating Capacity, Reference case",
        'url': "https://www.eia.gov/outlooks/aeo/data/browser/#/?id=8-EIA"
    },
    'target_power_mw': {
        'value': 600,
        'min': 100,
        'max': 1000,
        'step': 100,
        'citation': "Typical size for a large industrial facility or data center",
        'url': None
    },
    
    # Solar system parameters
    'solar_construction_time': {
        'value': 2.0,
        'min': 0.5,
        'max': 5.0,
        'step': 0.5,
        'citation': "Typical construction time for utility-scale solar projects",
        'url': None
    },
    'solar_capacity_mw': {
        'value': 2400,
        'min': 600,
        'max': 4800,
        'step': 100,
        'citation': "Based on typical solar farm sizes and capacity factors needed for 24/7 operation",
        'url': None
    },
    'battery_capacity_mwh': {
        'value': 9600,
        'min': 1200,
        'max': 12000,
        'step': 100,
        'citation': "Enough batteries to run a 600 MW facility for about 16 hours",
        'url': None
    },
    'solar_replacement_cost_factor': {
        'value': 0.7,
        'min': 0.5,
        'max': 0.9,
        'step': 0.05,
        'citation': "Industry standard assumption for solar panel replacement costs as a fraction of initial cost",
        'url': None
    },
    'round_trip_efficiency_pct': {
        'value': 94.0,
        'min': 85.0,
        'max': 99.0,
        'step': 0.5,
        'citation': "Lazard Levelized Cost of Storage 2023, Lithium-ion battery round-trip efficiency",
        'url': "https://www.lazard.com/research-insights/levelized-cost-of-storage-version-9-0/"
    },
    'solar_and_inverter_cost': {
        'value': 1000.0,
        'min': 500.0,
        'max': 2000.0,
        'step': 50.0,
        'citation': "Lazard Levelized Cost of Energy 2023, Utility-scale solar PV capital cost including power conversion system",
        'url': "https://www.lazard.com/research-insights/levelized-cost-of-energy-version-17-0/"
    },
    'battery_cost': {
        'value': 250.0,
        'min': 100.0,
        'max': 500.0,
        'step': 10.0,
        'citation': "Lazard Levelized Cost of Storage 2023, Lithium-ion battery capital cost",
        'url': "https://www.lazard.com/research-insights/levelized-cost-of-storage-version-9-0/"
    },
    'land_cost': {
        'value': 10000.0,
        'min': 1000.0,
        'max': 20000.0,
        'step': 500.0,
        'citation': "USDA Land Values 2023, Nevada agricultural land value",
        'url': "https://www.nass.usda.gov/Publications/Todays_Reports/reports/land0823.pdf"
    },
    'solar_maintenance_cost': {
        'value': 12.0,
        'min': 5.0,
        'max': 100.0,
        'step': 1.0,
        'citation': "Lazard Levelized Cost of Energy 2023, Utility-scale solar PV O&M cost",
        'url': "https://www.lazard.com/research-insights/levelized-cost-of-energy-version-17-0/"
    },
    
    # Nuclear system parameters
    'nuclear_construction_cost': {
        'value': 9000.0,
        'min': 3000.0,
        'max': 12000.0,
        'step': 100.0,
        'citation': "Lazard Levelized Cost of Energy 2023, Nuclear power plant capital cost, and the Vogtle plant construction costs.",
        'url': "https://www.lazard.com/research-insights/levelized-cost-of-energy-version-17-0/"
    },
    'nuclear_construction_time': {
        'value': 7,
        'min': 3,
        'max': 15,
        'step': 1,
        'citation': "IAEA Construction Time Database, Average construction time for nuclear power plants",
        'url': "https://www.iaea.org/resources/databases/construction-time-database"
    },
    'nuclear_capacity_factor': {
        'value': 92.0,
        'min': 70.0,
        'max': 98.0,
        'step': 1.0,
        'citation': "IAEA Power Reactor Information System, Average capacity factor for nuclear power plants",
        'url': "https://pris.iaea.org/PRIS/WorldStatistics/OperationalReactorsByCountry.aspx"
    },
    'nuclear_fuel_cost': {
        'value': 20.0,
        'min': 5.0,
        'max': 50.0,
        'step': 1.0,
        'citation': "World Nuclear Association, Nuclear Fuel Cost Report 2023",
        'url': "https://www.world-nuclear.org/information-library/economic-aspects/economics-of-nuclear-power.aspx"
    },
    'nuclear_maintenance_cost': {
        'value': 100.0,
        'min': 50.0,
        'max': 200.0,
        'step': 10.0,
        'citation': "Lazard Levelized Cost of Energy 2023, Nuclear power plant O&M cost",
        'url': "https://www.lazard.com/research-insights/levelized-cost-of-energy-version-17-0/"
    },
    'nuclear_decommissioning_cost': {
        'value': 500.0,
        'min': 100.0,
        'max': 1000.0,
        'step': 50.0,
        'citation': "IAEA Decommissioning Costs Database, Average decommissioning cost per kW",
        'url': "https://www.iaea.org/resources/databases/decommissioning-costs-database"
    }
}

# Color palette for charts
COLORS = {
    'Solar upfront': '#FFA500',  # Orange
    'Battery upfront': '#FFD700',  # Gold
    'Solar land': '#90EE90',  # Light Green
    'Solar O&M': '#FF69B4',  # Hot Pink
    'Solar replacement': '#FF6347',  # Tomato
    'Nuclear upfront': '#1E90FF',  # Dodger Blue
    'Nuclear fuel': '#4169E1',  # Royal Blue
    'Nuclear O&M': '#87CEEB',  # Sky Blue
    'Decommissioning (Nuc)': '#000080'  # Navy
}

# Tooltip definitions with full citations
TOOLTIPS = {
    'solar_efficiency': "The percentage of sunlight that solar panels convert into electricity. Higher efficiency means less land needed but higher costs. Source: National Renewable Energy Laboratory (NREL) Best Research-Cell Efficiencies Chart, 2023.",
    'degradation_rate': "The annual rate at which solar panel efficiency decreases over time. Lower degradation means longer-lasting panels. Source: NREL PV Degradation Rates, 2023.",
    'panel_lifetime': "The expected operational lifetime of solar panels before replacement is needed. Source: NREL PV Lifetime Project, 2023.",
    'battery_efficiency': "The percentage of energy that can be recovered from battery storage. Higher efficiency means less energy loss during storage. Source: Lazard Levelized Cost of Storage 2023.",
    'battery_lifetime': "The expected operational lifetime of battery storage systems before replacement is needed. Source: Lazard Levelized Cost of Storage 2023.",
    'solar_module_cost': "The cost per kilowatt of solar panel capacity, including installation. Source: Lazard Levelized Cost of Energy 2023.",
    'battery_cost': "The cost per kilowatt-hour of battery storage capacity. Source: Lazard Levelized Cost of Storage 2023.",
    'construction_time': "The expected time to build and commission the nuclear power plant. Source: IAEA Construction Time Database.",
    'capacity_factor': "The percentage of time the power plant operates at full capacity due to refueling or mainteance. Source: IAEA Power Reactor Information System.",
    'fuel_cost': "The cost of nuclear fuel per megawatt-hour of electricity generated. Source: World Nuclear Association.",
    'discount_rate': "The annual rate used to calculate the present value of future costs. Source: U.S. Energy Information Administration (EIA) Annual Energy Outlook 2023."
}

def load_solar_data(file_path):
    """Load and process solar data from CSV file."""
    df = pd.read_csv(file_path)
    # Specify the format for datetime parsing (MM/DD/YY HH:MM)
    df['LocalTime'] = pd.to_datetime(df['LocalTime'], format='%m/%d/%y %H:%M')
    return df

def get_location_name(lat, lon):
    """Get a human-readable location name for given coordinates."""
    # Find the closest predefined location
    min_dist = float('inf')
    closest_name = None
    
    for (ref_lat, ref_lon), name in LOCATION_NAMES.items():
        dist = ((lat - ref_lat) ** 2 + (lon - ref_lon) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    
    return closest_name or f"Site near {lat:.2f}°N, {lon:.2f}°W"

def get_solar_sites():
    """Get list of available solar sites from data directory."""
    data_dir = "data"
    sites = []
    for file in os.listdir(data_dir):
        if file.endswith("_60_Min.csv") and "HA4" in file:  # Only include HA4 files
            # Extract coordinates and capacity from filename
            # Format: HA4_35.85_-115.15_2006_UPV_50MW_60_Min.csv
            parts = file.split('_')
            if len(parts) >= 6:  # Ensure we have enough parts
                try:
                    lat = float(parts[1])
                    lon = float(parts[2])
                    # Find the part that contains MW
                    capacity_part = next((p for p in parts if 'MW' in p), None)
                    if capacity_part:
                        capacity = float(capacity_part.replace('MW', ''))
                        location_name = get_location_name(lat, lon)
                        sites.append({
                            'file': file,
                            'lat': lat,
                            'lon': lon,
                            'capacity': capacity,
                            'location_name': location_name
                        })
                except (ValueError, IndexError):
                    continue  # Skip files that don't match the expected format
    return sites

def simulate_battery_storage(solar_data, solar_capacity_mw, battery_capacity_mwh, target_power_mw, round_trip_efficiency_pct=95.0):
    """Simulate battery storage operation over the year.
    
    Args:
        solar_data (pd.DataFrame): DataFrame with columns ['LocalTime', 'Power(MW)']
        solar_capacity_mw (float): Rated capacity of solar plant in megawatts (MW)
        battery_capacity_mwh (float): Battery storage capacity in megawatt-hours (MWh)
        target_power_mw (float): Target power load provided to the facility (MW)
        round_trip_efficiency_pct (float): Battery round-trip efficiency as percentage (80-98%)
    
    Returns:
        tuple: Contains arrays for each hour of the year:
            - battery_soc (np.array): Battery state of charge in megawatt-hours (MWh)
            - power_output (np.array): Power output in megawatts (MW)
            - battery_charge (np.array): Battery charging power in megawatts (MW)
            - battery_discharge (np.array): Battery discharging power in megawatts (MW)
    """
    # Initialize arrays for results
    battery_soc = np.zeros(len(solar_data))  # State of charge in MWh
    power_output = np.zeros(len(solar_data))  # Power output in MW
    battery_charge = np.zeros(len(solar_data))  # Battery charging power in MW
    battery_discharge = np.zeros(len(solar_data))  # Battery discharging power in MW
    battery_soc[0] = battery_capacity_mwh * 1.0  # Start at full charge
    
    # Calculate efficiency factors (both values are < 1)
    round_trip_efficiency = round_trip_efficiency_pct / 100  # Convert to decimal
    charge_efficiency = np.sqrt(round_trip_efficiency)  # Efficiency when charging
    discharge_efficiency = charge_efficiency  # Equal charge and discharge efficiencies
    
    # Calculate scaling factor for solar data (unitless)
    scale_factor = solar_capacity_mw / solar_data['Power(MW)'].max()
    scaled_solar = solar_data['Power(MW)'] * scale_factor  # Solar power timeseries in MW
    
    # Simulate each hour
    for k in range(1, len(solar_data)):
        # Calculate available power from solar (MW)
        available_solar_power = scaled_solar.iloc[k]
        delta_t = 1 # time step in hours
        
        # Calculate power surplus (MW)
        # Positive means we have extra power to store
        power_surplus = available_solar_power - target_power_mw
        
        if power_surplus > 0:  # Can charge battery
            # Calculate the battery charging charge (MW)
            max_charge = min(battery_capacity_mwh - battery_soc[k-1], power_surplus)
            battery_charge[k] = max_charge  # MW
            # Update battery state (MWh = MWh + MW * efficiency)
            battery_soc[k] = battery_soc[k-1] +  battery_charge[k] * charge_efficiency * delta_t
            # Power output is available power minus what went to charging (MW)
            power_output[k] = min(available_solar_power - battery_charge[k],target_power_mw)
        else:  # Need to discharge battery
            # Calculate maximum discharge (MW)
            max_discharge = min(battery_soc[k-1] * discharge_efficiency, -power_surplus)
            battery_discharge[k] = max_discharge  # MW
            # Update battery state (MWh = MWh - MW / efficiency)
            battery_soc[k] = battery_soc[k-1] - battery_discharge[k] / discharge_efficiency * delta_t
            # Power output is available power plus discharge (MW)
            power_output[k] = min(available_solar_power + max_discharge,target_power_mw)
    
    return battery_soc, power_output, battery_charge, battery_discharge

def calculate_reliability(power_output_mw, target_power_mw):
    """Calculate the reliability of the system."""
    total_energy_target = target_power_mw * len(power_output_mw)
    total_energy_actual = np.sum(power_output_mw)
    reliability = total_energy_actual / total_energy_target
    return reliability

def calculate_solar_costs(
    solar_capacity_mw,
    battery_capacity_mwh,
    solar_module_cost_usd_per_kw=DEFAULT_PARAMETERS['solar_and_inverter_cost']['value'],
    battery_cost_usd_per_kwh=DEFAULT_PARAMETERS['battery_cost']['value'],
    land_cost_usd_per_acre=DEFAULT_PARAMETERS['land_cost']['value'],
    maintenance_cost_usd_per_kw_year=DEFAULT_PARAMETERS['solar_maintenance_cost']['value'],
    installation_cost_usd_per_mw=1000000  # Source: Lazard LCOE 2023
):
    """Calculate solar power system costs."""
    # Land requirements (assuming 1 MW requires 5 acres)
    required_land_acres = solar_capacity_mw * 5
    
    # Initial costs
    solar_panel_cost_usd = solar_capacity_mw * 1000 * solar_module_cost_usd_per_kw  # Convert MW to kW
    battery_cost_usd = battery_capacity_mwh * 1000 * battery_cost_usd_per_kwh  # Convert MWh to kWh
    land_cost_usd = required_land_acres * land_cost_usd_per_acre
    installation_cost_usd = solar_capacity_mw * installation_cost_usd_per_mw
    
    # Combine solar panel and installation costs
    solar_upfront_cost_usd = solar_panel_cost_usd + installation_cost_usd
    
    # Annual costs
    annual_maintenance_usd = solar_capacity_mw * 1000 * maintenance_cost_usd_per_kw_year  # Convert MW to kW
    
    return {
        'solar_upfront_cost_usd': solar_upfront_cost_usd,
        'battery_cost_usd': battery_cost_usd,
        'land_cost_usd': land_cost_usd,
        'annual_maintenance_usd': annual_maintenance_usd,
        'total_initial_cost_usd': solar_upfront_cost_usd + battery_cost_usd + land_cost_usd
    }

def calculate_nuclear_costs(
    nuclear_construction_time_years=DEFAULT_PARAMETERS['nuclear_construction_time']['value'],
    nuclear_capacity_factor_pct=DEFAULT_PARAMETERS['nuclear_capacity_factor']['value'],
    nuclear_fuel_cost_usd_per_mwh=DEFAULT_PARAMETERS['nuclear_fuel_cost']['value'],
    nuclear_maintenance_cost_usd_per_kw_year=DEFAULT_PARAMETERS['nuclear_maintenance_cost']['value'],
    nuclear_decommissioning_cost_usd_per_kw=DEFAULT_PARAMETERS['nuclear_decommissioning_cost']['value'],
    nuclear_construction_cost_usd_per_kw=DEFAULT_PARAMETERS['nuclear_construction_cost']['value']
):
    """Calculate nuclear power system costs over the project lifetime."""
    # Convert percentage to decimal
    nuclear_capacity_factor = nuclear_capacity_factor_pct / 100
    
    # Initial costs
    nuclear_construction_cost_usd = FACILITY_SIZE_MW * 1000 * nuclear_construction_cost_usd_per_kw  # Convert MW to kW
    
    # Annual costs
    # Calculate annual energy production in MWh
    annual_energy_mwh = FACILITY_SIZE_MW * HOURS_PER_YEAR * nuclear_capacity_factor
    
    # Calculate fuel cost based on energy production
    nuclear_annual_fuel_cost_usd = annual_energy_mwh * nuclear_fuel_cost_usd_per_mwh
    
    # Calculate O&M cost based on capacity
    nuclear_annual_maintenance_usd = FACILITY_SIZE_MW * 1000 * nuclear_maintenance_cost_usd_per_kw_year  # Convert MW to kW
    
    # Decommissioning cost
    nuclear_decommissioning_cost_usd = FACILITY_SIZE_MW * 1000 * nuclear_decommissioning_cost_usd_per_kw  # Convert MW to kW
    
    return {
        'nuclear_construction_cost_usd': nuclear_construction_cost_usd,
        'nuclear_annual_fuel_cost_usd': nuclear_annual_fuel_cost_usd,
        'nuclear_annual_maintenance_usd': nuclear_annual_maintenance_usd,
        'nuclear_decommissioning_cost_usd': nuclear_decommissioning_cost_usd,
        'nuclear_capacity_factor': nuclear_capacity_factor,
        'nuclear_construction_time_years': nuclear_construction_time_years
    }

def create_cost_comparison_chart(solar_costs, nuclear_costs, discount_rate_pct=7.0):
    """Create a cost comparison chart using Plotly."""
    years = list(range(YEARS + 1))
    discount_rate = discount_rate_pct / 100
    
    # Calculate present value of all costs
    solar_pv_usd = solar_costs['total_initial_cost_usd']
    nuclear_pv_usd = nuclear_costs['nuclear_construction_cost_usd']
    
    # Calculate present value of annual costs
    solar_annual_pv = sum(solar_costs['annual_maintenance_usd'] / (1 + discount_rate) ** year for year in range(1, YEARS + 1))
    nuclear_annual_pv = sum((nuclear_costs['nuclear_annual_fuel_cost_usd'] + nuclear_costs['nuclear_annual_maintenance_usd']) / (1 + discount_rate) ** year for year in range(1, YEARS + 1))
    
    # Calculate present value of replacement costs
    solar_replacement_pv = sum(
        (solar_costs['total_initial_cost_usd'] * 0.7 if year in [20, 40] else 0)
        / (1 + discount_rate) ** year for year in range(1, YEARS + 1)
    )
    
    # Calculate present value of decommissioning
    nuclear_decommissioning_pv = nuclear_costs['nuclear_decommissioning_cost_usd'] / (1 + discount_rate) ** YEARS
    
    # Create stacked bar chart data
    solar_components = {
        'Solar upfront': solar_costs['solar_upfront_cost_usd'],
        'Battery upfront': solar_costs['battery_cost_usd'],
        'Solar land': solar_costs['land_cost_usd'],
        'Solar O&M': solar_annual_pv,
        'Solar replacement': solar_replacement_pv
    }
    
    nuclear_components = {
        'Nuclear upfront': nuclear_costs['nuclear_construction_cost_usd'],
        'Nuclear fuel': nuclear_costs['nuclear_annual_fuel_cost_usd'] * YEARS / (1 + discount_rate) ** (YEARS/2),
        'Nuclear O&M': nuclear_annual_pv,
        'Decommissioning (Nuc)': nuclear_decommissioning_pv
    }
    
    # Create figure with improved layout
    fig = go.Figure()
    
    # Add solar components
    solar_y = 0
    for component, value in solar_components.items():
        # Format value in billions or millions
        if value >= 1e9:
            formatted_value = f"${value/1e9:.2f}B"
        else:
            formatted_value = f"${value/1e6:.0f}M"
        fig.add_trace(go.Bar(
            name=component,
            x=['Solar'],
            y=[value],
            marker_color=COLORS[component],
            text=[f"{component}: {formatted_value}"],
            textposition='auto',
            textfont=dict(size=10, color='black'),
            hovertemplate=f"{component}: {formatted_value}<extra></extra>"
        ))
        solar_y += value
    
    # Add nuclear components
    nuclear_y = 0
    for component, value in nuclear_components.items():
        # Format value in billions or millions
        if value >= 1e9:
            formatted_value = f"${value/1e9:.2f}B"
        else:
            formatted_value = f"${value/1e6:.0f}M"
        fig.add_trace(go.Bar(
            name=component,
            x=['Nuclear'],
            y=[value],
            marker_color=COLORS[component],
            text=[f"{component}: {formatted_value}"],
            textposition='auto',
            textfont=dict(size=10, color='black'),
            hovertemplate=f"{component}: {formatted_value}<extra></extra>"
        ))
        nuclear_y += value
    
    # Update layout with improved styling
    fig.update_layout(
        yaxis_title='Present value cost (USD)',
        barmode='stack',
        showlegend=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        height=600,
        xaxis=dict(
            tickmode='array',
            ticktext=['Solar', 'Nuclear'],
            tickvals=[0, 1],
            tickangle=0,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            tickformat='$,.0f',
            tickfont=dict(size=12)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def plot_weekly_data(solar_data, power_output, battery_soc, battery_charge, battery_discharge, week_start_date, solar_capacity_mw):
    """Create a plot showing one week of data with power and battery state of charge in separate subplots."""
    # Convert week_start_date to datetime if it's a string or date
    if isinstance(week_start_date, str):
        week_start_date = pd.to_datetime(week_start_date)
    elif isinstance(week_start_date, date):
        week_start_date = pd.to_datetime(week_start_date)
    
    # Calculate start and end indices for the week
    start_idx = solar_data[solar_data['LocalTime'].dt.date >= week_start_date.date()].index[0]
    end_idx = min(start_idx + 168, len(solar_data))  # 168 hours = 1 week
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Power output and battery activity', 'Battery state of charge'),
        row_heights=[0.7, 0.3]
    )
    
    # Calculate scaling factor for solar data
    scale_factor = solar_capacity_mw / solar_data['Power(MW)'].max()
    scaled_solar = solar_data['Power(MW)'].iloc[start_idx:end_idx] * scale_factor
    
    # Add scaled solar power trace
    fig.add_trace(
        go.Scatter(
            x=solar_data['LocalTime'].iloc[start_idx:end_idx],
            y=scaled_solar,
            name='Solar power',
            line=dict(color='#FF8C00')  # Dark orange color
        ),
        row=1, col=1
    )
    
    # Add power output trace
    fig.add_trace(
        go.Scatter(
            x=solar_data['LocalTime'].iloc[start_idx:end_idx],
            y=power_output[start_idx:end_idx],
            name='Power output',
            line=dict(color='#1E90FF')  # Dodger blue color
        ),
        row=1, col=1
    )
    
    # Add battery charge/discharge traces
    fig.add_trace(
        go.Scatter(
            x=solar_data['LocalTime'].iloc[start_idx:end_idx],
            y=battery_charge[start_idx:end_idx],
            name='Battery charging',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=solar_data['LocalTime'].iloc[start_idx:end_idx],
            y=-battery_discharge[start_idx:end_idx],
            name='Battery discharging',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Calculate battery SOC percentage
    battery_capacity = max(battery_soc)  # Use max SOC as capacity
    battery_soc_pct = (battery_soc / battery_capacity) * 100
    
    # Add battery state of charge trace
    fig.add_trace(
        go.Scatter(
            x=solar_data['LocalTime'].iloc[start_idx:end_idx],
            y=battery_soc_pct[start_idx:end_idx],
            name='Battery state of charge',
            line=dict(color='blue')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        #title=f'Weekly power and battery data (Starting {week_start_date.strftime("%Y-%m-%d")})',
        height=800,  # Make the chart taller to accommodate subplots
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="State of charge (%)", row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)  # Set SOC y-axis range to 0-100%
    
    return fig

def create_comparison_table(solar_costs, nuclear_costs, solar_capacity_mw, battery_capacity_mwh, 
                            power_output, reliability, discount_rate_pct, solar_lcoe, nuclear_lcoe,
                            solar_construction_time):
    """Create a comprehensive comparison table of all metrics."""
    # Calculate total costs
    solar_total = (
        solar_costs['total_initial_cost_usd'] +
        sum(solar_costs['annual_maintenance_usd'] / (1 + discount_rate_pct/100) ** year 
            for year in range(1, YEARS + 1))
    )
    
    nuclear_total = (
        nuclear_costs['nuclear_construction_cost_usd'] +
        sum((nuclear_costs['nuclear_annual_fuel_cost_usd'] + nuclear_costs['nuclear_annual_maintenance_usd']) 
            / (1 + discount_rate_pct/100) ** year for year in range(1, YEARS + 1)) +
        nuclear_costs['nuclear_decommissioning_cost_usd'] / (1 + discount_rate_pct/100) ** YEARS
    )
    
    # Calculate annual energy output
    solar_annual_energy = np.sum(power_output)  # MWh
    nuclear_annual_energy = FACILITY_SIZE_MW * HOURS_PER_YEAR * nuclear_costs['nuclear_capacity_factor']
    
    # Create list of metrics and their values
    metrics = [
        ('Peak power capacity', 
         f"{solar_capacity_mw:,} MW",
         f"{FACILITY_SIZE_MW:,} MW"),
        ('Annual energy output',
         f"{solar_annual_energy:,.0f} MWh",
         f"{nuclear_annual_energy:,.0f} MWh"),
        ('System reliability',
         f"{reliability:.1%}",
         f"{nuclear_costs['nuclear_capacity_factor']:.1%}"),
        ('Upfront cost',
         f"${solar_costs['total_initial_cost_usd']:,.0f}",
         f"${nuclear_costs['nuclear_construction_cost_usd']:,.0f}"),
        ('Annual O&M cost',
         f"${solar_costs['annual_maintenance_usd']:,.0f}/year",
         f"${nuclear_costs['nuclear_annual_fuel_cost_usd'] + nuclear_costs['nuclear_annual_maintenance_usd']:,.0f}/year"),
        ('Total present value cost',
         f"${solar_total:,.0f}",
         f"${nuclear_total:,.0f}"),
        ('Levelized cost of energy (LCOE)',
         f"${solar_lcoe:.2f}/MWh",
         f"${nuclear_lcoe:.2f}/MWh"),
        ('Land required',
         f"{solar_capacity_mw * 5:,.0f} acres",
         "1,000 acres"),
        ('Construction time',
         f"{solar_construction_time:.1f} years",
         f"{nuclear_costs['nuclear_construction_time_years']:.1f} years"),
        ('Operational lifetime',
         f"{YEARS - solar_construction_time:.1f} years",
         f"{YEARS - nuclear_costs['nuclear_construction_time_years']:.1f} years"),
        ('Environmental impact',
         "Low (no emissions during operation)",
         "Low (no CO2 emissions during operation)"),
        ('Decommissioning cost',
         "Minimal (panels can be recycled)",
         f"${nuclear_costs['nuclear_decommissioning_cost_usd']:,.0f}")
    ]
    
    # Create DataFrame directly from the list of tuples
    df = pd.DataFrame(metrics, columns=['Metric', 'Solar PV + Storage', 'Nuclear'])
    
    return df

def calculate_capacity_factor(solar_data, solar_capacity_mw):
    """Calculate the capacity factor for a solar site."""
    # Scale the power data to the solar capacity
    scale_factor = solar_capacity_mw / solar_data['Power(MW)'].max()
    scaled_power = solar_data['Power(MW)'] * scale_factor
    
    # Calculate capacity factor (average power output / rated capacity)
    capacity_factor = scaled_power.mean() / solar_capacity_mw
    return capacity_factor

def create_solar_sites_map(sites, solar_data_dict):
    """Create a map showing solar sites and their capacity factors."""
    # Create figure
    fig = go.Figure()
    
    # Calculate center point and bounds
    lats = [site['lat'] for site in sites]
    lons = [site['lon'] for site in sites]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Calculate bounds for Nevada
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Add some padding to the bounds
    lat_padding = (lat_max - lat_min) * 0.2
    lon_padding = (lon_max - lon_min) * 0.2
    
    # Calculate all capacity factors first to determine the range
    capacity_factors = []
    for site in sites:
        cf = calculate_capacity_factor(solar_data_dict[site['file']], site['capacity'])
        capacity_factors.append(cf * 100)  # Convert to percentage
    
    # Define the color range based on actual data
    cf_min = min(capacity_factors)
    cf_max = max(capacity_factors)
    
    # Create custom colormap (100 points from blue to dark gray to red)
    colormap = []
    for i in range(100):
        if i < 50:  # Blue to dark gray
            r = int(37 * (i / 50))
            g = int(24 * (i / 50))
            b = int(255 * (1 - i / 50))
        else:  # Dark gray to red
            r = int(37 + (255 - 37) * ((i - 50) / 50))
            g = int(24 * (1 - (i - 50) / 50))
            b = int(24 * (1 - (i - 50) / 50))
        colormap.append(f'rgb({r},{g},{b})')
    
    # Add scatter map trace for each site
    for site in sites:
        capacity_factor = calculate_capacity_factor(
            solar_data_dict[site['file']], 
            site['capacity']
        ) * 100  # Convert to percentage
        
        # Normalize capacity factor to 0-1 range and get color index
        color_index = int(99 * (capacity_factor - cf_min) / (cf_max - cf_min))
        color = colormap[color_index]
        
        fig.add_trace(go.Scattermapbox(
            lat=[site['lat']],
            lon=[site['lon']],
            mode='markers+text',
            marker=dict(
                size=15,
                color=color
            ),
            text=[f"{site['location_name']}<br>CF: {capacity_factor:.1f}%"],
            textposition='top center',
            showlegend=False  # Hide the legend
        ))
    
    # Create colorbar ticks
    tick_values = np.linspace(cf_min, cf_max, 6)  # 6 ticks for the colorbar
    
    # Add colorbar as a separate trace
    fig.add_trace(go.Scattermapbox(
        lat=[None],
        lon=[None],
        mode='markers',
        marker=dict(
            size=0,
            showscale=True,
            colorscale=[[i/99, color] for i, color in enumerate(colormap)],
            colorbar=dict(
                title=dict(
                    text='Capacity Factor (%)',
                    side='right'
                ),
                x=1.02,
                len=0.75,
                thickness=20,
                tickmode='array',
                ticktext=[f'{v:.1f}' for v in tick_values],
                tickvals=tick_values
            ),
            cmin=cf_min,
            cmax=cf_max
        ),
        showlegend=False
    ))
    
    # Update layout with calculated center and zoom
    fig.update_layout(
        title='Solar Sites and Capacity Factors in Nevada',
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=39.5, lon=-116.5),
            zoom=6.5
        ),
        height=600,
        margin=dict(l=0, r=50, t=30, b=50),  # Increased bottom margin
        showlegend=False  # Hide the legend
    )
    
    return fig

def calculate_present_value_costs(solar_costs, nuclear_costs, discount_rate_pct, solar_replacement_cost_factor):
    """Calculate present value of all costs for both systems."""
    discount_rate = discount_rate_pct / 100
    
    # Get construction times
    nuclear_construction_years = nuclear_costs.get('nuclear_construction_time_years', 5)
    
    # Solar components
    solar_initial_pv = solar_costs['total_initial_cost_usd']
    solar_annual_pv = sum(
        solar_costs['annual_maintenance_usd'] / (1 + discount_rate) ** year 
        for year in range(1, YEARS)  # Solar starts operation after 1 year
    )
    solar_replacement_pv = (
        solar_costs['total_initial_cost_usd'] * solar_replacement_cost_factor / 
        (1 + discount_rate) ** 20
    )
    solar_total_pv = solar_initial_pv + solar_annual_pv + solar_replacement_pv
    
    # Nuclear components
    nuclear_initial_pv = nuclear_costs['nuclear_construction_cost_usd']
    nuclear_annual_pv = sum(
        (nuclear_costs['nuclear_annual_fuel_cost_usd'] + nuclear_costs['nuclear_annual_maintenance_usd']) 
        / (1 + discount_rate) ** (year + nuclear_construction_years)  # Costs start after construction
        for year in range(1, YEARS - nuclear_construction_years + 1)
    )
    nuclear_decommissioning_pv = nuclear_costs['nuclear_decommissioning_cost_usd'] / (1 + discount_rate) ** YEARS
    nuclear_total_pv = nuclear_initial_pv + nuclear_annual_pv + nuclear_decommissioning_pv
    
    return {
        'solar': {
            'initial': solar_initial_pv,
            'annual': solar_annual_pv,
            'replacement': solar_replacement_pv,
            'total': solar_total_pv
        },
        'nuclear': {
            'initial': nuclear_initial_pv,
            'annual': nuclear_annual_pv,
            'decommissioning': nuclear_decommissioning_pv,
            'total': nuclear_total_pv
        }
    }

def calculate_lcoe_components(pv_costs, solar_costs, nuclear_costs, solar_annual_energy, nuclear_annual_energy, discount_rate_pct, nuclear_construction_years=5, solar_construction_years=2):
    """Calculate LCOE components for both systems."""
    discount_rate = discount_rate_pct / 100
    
    # Calculate capital recovery factors using total project years
    solar_crf = (discount_rate * (1 + discount_rate) ** YEARS) / ((1 + discount_rate) ** YEARS - 1)
    nuclear_crf = (discount_rate * (1 + discount_rate) ** YEARS) / ((1 + discount_rate) ** YEARS - 1)
    
    # Solar components
    solar_components = {
        'Solar upfront': (solar_crf * solar_costs['solar_upfront_cost_usd']) / solar_annual_energy,
        'Battery upfront': (solar_crf * solar_costs['battery_cost_usd']) / solar_annual_energy,
        'Solar land': (solar_crf * solar_costs['land_cost_usd']) / solar_annual_energy,
        'Solar O&M': solar_costs['annual_maintenance_usd'] / solar_annual_energy,
        'Solar replacement': (solar_crf * solar_costs['total_initial_cost_usd'] * 0.7 / (1 + discount_rate) ** 20) / solar_annual_energy
    }
    
    # Nuclear components
    nuclear_components = {
        'Nuclear upfront': (nuclear_crf * nuclear_costs['nuclear_construction_cost_usd']) / nuclear_annual_energy,
        'Nuclear fuel': nuclear_costs['nuclear_annual_fuel_cost_usd'] / nuclear_annual_energy,
        'Nuclear O&M': nuclear_costs['nuclear_annual_maintenance_usd'] / nuclear_annual_energy,
        'Decommissioning (Nuc)': (nuclear_crf * nuclear_costs['nuclear_decommissioning_cost_usd']) / nuclear_annual_energy
    }
    
    # Calculate total LCOE from components
    solar_lcoe = sum(solar_components.values())
    nuclear_lcoe = sum(nuclear_components.values())
    
    return solar_components, nuclear_components, solar_lcoe, nuclear_lcoe

def calculate_lcoe(solar_costs, nuclear_costs, power_output, discount_rate_pct=7.0, solar_replacement_cost_factor=0.7):
    """Calculate Levelized Cost of Energy (LCOE) for both systems."""
    # Calculate total energy production (MWh)
    solar_annual_energy = np.sum(power_output)  # This is already in MWh
    nuclear_annual_energy = FACILITY_SIZE_MW * HOURS_PER_YEAR * nuclear_costs['nuclear_capacity_factor']
    
    # Calculate present value costs
    pv_costs = calculate_present_value_costs(
        solar_costs, 
        nuclear_costs, 
        discount_rate_pct,
        solar_replacement_cost_factor
    )
    
    # Calculate LCOE components and totals
    solar_components, nuclear_components, solar_lcoe, nuclear_lcoe = calculate_lcoe_components(
        pv_costs,
        solar_costs,
        nuclear_costs,
        solar_annual_energy,
        nuclear_annual_energy,
        discount_rate_pct,
        nuclear_costs.get('nuclear_construction_time_years', 5),
        solar_costs.get('solar_construction_time', 2)
    )
    
    return solar_lcoe, nuclear_lcoe

def create_sensitivity_analysis(solar_costs, nuclear_costs, power_output):
    """Create sensitivity analysis charts for LCOE and total cost vs discount rate."""
    discount_rates = np.linspace(0.03, 0.12, num=10)
    solar_lcoes = []
    nuclear_lcoes = []
    solar_costs_pv = []
    nuclear_costs_pv = []
    
    for rate in discount_rates:
        solar_lcoe, nuclear_lcoe = calculate_lcoe(solar_costs, nuclear_costs, power_output, rate * 100)
        solar_lcoes.append(solar_lcoe)
        nuclear_lcoes.append(nuclear_lcoe)
        
        # Calculate present value costs
        solar_pv = (
            solar_costs['total_initial_cost_usd'] +
            sum(solar_costs['annual_maintenance_usd'] / (1 + rate) ** year 
                for year in range(1, YEARS + 1))
        )
        nuclear_pv = (
            nuclear_costs['nuclear_construction_cost_usd'] +
            sum((nuclear_costs['nuclear_annual_fuel_cost_usd'] + nuclear_costs['nuclear_annual_maintenance_usd']) 
                / (1 + rate) ** year for year in range(1, YEARS + 1)) +
            nuclear_costs['nuclear_decommissioning_cost_usd'] / (1 + rate) ** YEARS
        )
        solar_costs_pv.append(solar_pv)
        nuclear_costs_pv.append(nuclear_pv)
    
    # Create LCOE sensitivity plot
    fig_lcoe = go.Figure()
    fig_lcoe.add_trace(go.Scatter(
        x=discount_rates * 100,
        y=solar_lcoes,
        name='Solar PV + storage',
        line=dict(color='orange')
    ))
    fig_lcoe.add_trace(go.Scatter(
        x=discount_rates * 100,
        y=nuclear_lcoes,
        name='Nuclear',
        line=dict(color='blue')
    ))
    fig_lcoe.update_layout(
        xaxis_title='Discount Rate (%)',
        yaxis_title='Levelized Cost of Energy ($/MWh)',
        height=400,
        showlegend=False,
        yaxis=dict(
            range=[0, max(max(solar_lcoes), max(nuclear_lcoes)) * 1.1]  # Add 10% padding
        )
    )
    
    # Create total cost sensitivity plot
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=discount_rates * 100,
        y=solar_costs_pv,
        name='Solar PV + storage',
        line=dict(color='orange')
    ))
    fig_cost.add_trace(go.Scatter(
        x=discount_rates * 100,
        y=nuclear_costs_pv,
        name='Nuclear',
        line=dict(color='blue')
    ))
    fig_cost.update_layout(
        xaxis_title='Discount Rate (%)',
        yaxis_title='Present Value Cost ($)',
        height=400,
        showlegend=False,
        yaxis=dict(
            range=[0, max(max(solar_costs_pv), max(nuclear_costs_pv)) * 1.1]  # Add 10% padding
        )
    )
    
    return fig_lcoe, fig_cost

def optimize_solar_battery(
    solar_data,
    target_power_mw,
    target_reliability,
    solar_module_cost,
    battery_cost,
    land_cost,
    round_trip_efficiency_pct,
    progress_bar
):
    """Optimize solar and battery capacity to meet target reliability at minimum cost."""
    
    # Calculate a reference cost for scaling
    reference_cost = (
        target_power_mw * 1000 * solar_module_cost +  # Solar cost
        target_power_mw * 24 * 1000 * battery_cost +  # 24h battery cost
        target_power_mw * 5 * land_cost  # Land cost
    )
    
    def objective_function(x):
        """Calculate total cost for given solar and battery capacity."""
        solar_capacity = x[0]
        battery_capacity = x[1]
        
        # Calculate costs
        solar_panel_cost = solar_capacity * 1000 * solar_module_cost  # Convert MW to kW
        battery_cost_total = battery_capacity * 1000 * battery_cost  # Convert MWh to kWh
        land_cost_total = solar_capacity * 5 * land_cost  # 5 acres per MW
        
        # Scale the total cost to be in a similar range as reliability (0-1)
        total_cost = solar_panel_cost + battery_cost_total + land_cost_total
        scaled_cost = total_cost / reference_cost
        
        return scaled_cost
    
    def reliability_constraint(x):
        """Calculate reliability for given solar and battery capacity."""
        solar_capacity = x[0]
        battery_capacity = x[1]
        
        print(f"\nTrying configuration:")
        print(f"solar_capacity: {solar_capacity:,.0f} MW")
        print(f"battery_capacity: {battery_capacity:,.0f} MWh")
        
        # Run battery simulation
        battery_soc, power_output, _, _ = simulate_battery_storage(
            solar_data,
            solar_capacity,
            battery_capacity,
            target_power_mw,
            round_trip_efficiency_pct
        )
        
        # Calculate reliability
        reliability = calculate_reliability(power_output, target_power_mw)
        
        # Update progress bar
        progress_bar.progress(min(1.0, reliability / target_reliability))
        
        print(f"reliability: {reliability:.3f}, target: {target_reliability:.3f}")
        print(f"constraint value: {reliability - target_reliability:.3f}")
        
        # Return reliability - target_reliability (must be >= 0 for constraint satisfaction)
        return reliability - target_reliability
    
    # Initial guess (start with more aggressive values)
    x0 = [target_power_mw * 8, target_power_mw * 32]  # Increased initial values
    
    # Bounds for solar and battery capacity (more generous bounds)
    bounds = [
        (target_power_mw * 4, target_power_mw * 15),  # Solar bounds
        (target_power_mw * 8, target_power_mw * 72)   # Battery bounds
    ]
    
    # Constraints
    constraints = [
        {
            'type': 'ineq',
            'fun': reliability_constraint,
            'name': 'reliability'
        }
    ]
    
    print("\nStarting optimization with:")
    print(f"Initial guess: {x0}")
    print(f"Bounds: {bounds}")
    print(f"Target reliability: {target_reliability}")
    
    # Run optimization with more iterations and better tolerance
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': MAX_ITERATIONS,
            'ftol': 1e-9,     # convergence criterion
            'eps': 1e-3,      # Step size for numerical derivatives
            'disp': True,     # Show optimization progress
            'iprint': 2       # Print optimization progress
        }
    )
    
    print("\nOptimization result:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final configuration: {result.x}")
    
    if not result.success:
        st.warning("Optimization did not converge. Try adjusting the target reliability.")
        return None
    
    # Get optimal values
    optimal_solar = result.x[0]
    optimal_battery = result.x[1]
    
    # Round to nearest step
    optimal_solar = round(optimal_solar / SOLAR_CAPACITY_STEP) * SOLAR_CAPACITY_STEP
    optimal_battery = round(optimal_battery / BATTERY_CAPACITY_STEP) * BATTERY_CAPACITY_STEP
    
    # Calculate final reliability with rounded values
    battery_soc, power_output, _, _ = simulate_battery_storage(
        solar_data,
        optimal_solar,
        optimal_battery,
        target_power_mw,
        round_trip_efficiency_pct
    )
    final_reliability = calculate_reliability(power_output, target_power_mw)
    
    print(f"\nFinal configuration after rounding:")
    print(f"Solar capacity: {optimal_solar:,.0f} MW")
    print(f"Battery capacity: {optimal_battery:,.0f} MWh")
    print(f"Final reliability: {final_reliability:.3f}")
    
    # Calculate actual cost (not scaled)
    actual_cost = (
        optimal_solar * 1000 * solar_module_cost +
        optimal_battery * 1000 * battery_cost +
        optimal_solar * 5 * land_cost
    )
    
    return {
        'solar_capacity': optimal_solar,
        'battery_capacity': optimal_battery,
        'reliability': final_reliability,
        'total_cost': actual_cost
    }

def main():
    # Display the image
    st.image("images/nuc-vs-solar.png", use_container_width=True)
    # Title
    st.title("Should I power my data center with solar and batteries or nuclear?")
    
    # Initialize session state for metric type if it doesn't exist
    if 'metric_type' not in st.session_state:
        st.session_state.metric_type = "Present value cost"
    
    # Initialize session state for system size if it doesn't exist
    if 'solar_capacity' not in st.session_state:
        st.session_state.solar_capacity = DEFAULT_PARAMETERS['solar_capacity_mw']['value']
    if 'battery_capacity' not in st.session_state:
        st.session_state.battery_capacity = DEFAULT_PARAMETERS['battery_capacity_mwh']['value']
    
    # Sidebar for common parameters
    st.sidebar.header("Common parameters")
    
    # Common parameters
    discount_rate_pct = st.sidebar.slider(
        "Discount rate (%)",
        min_value=DEFAULT_PARAMETERS['discount_rate_pct']['min'],
        max_value=DEFAULT_PARAMETERS['discount_rate_pct']['max'],
        value=DEFAULT_PARAMETERS['discount_rate_pct']['value'],
        step=DEFAULT_PARAMETERS['discount_rate_pct']['step'],
        help=f"{TOOLTIPS['discount_rate']} {DEFAULT_PARAMETERS['discount_rate_pct']['citation']}"
    )
    
    target_power_mw = st.sidebar.slider(
        "Facility power rquirement (MW)",
        min_value=DEFAULT_PARAMETERS['target_power_mw']['min'],
        max_value=DEFAULT_PARAMETERS['target_power_mw']['max'],
        value=DEFAULT_PARAMETERS['target_power_mw']['value'],
        step=DEFAULT_PARAMETERS['target_power_mw']['step'],
        help=DEFAULT_PARAMETERS['target_power_mw']['citation']
    )
    
    # Add recalculate button
    if st.sidebar.button("Recalculate Costs", help="Click to recalculate all costs and update visualizations"):
        st.rerun()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Solar + battery system", "Nuclear system", "Results & analysis"])
    
    # Load all solar data first
    sites = get_solar_sites()
    # Sort sites by latitude (north to south)
    sites.sort(key=lambda x: x['lat'], reverse=True)
    
    solar_data_dict = {
        site['file']: load_solar_data(os.path.join("data", site['file']))
        for site in sites
    }
    
    # Calculate capacity factors for all sites
    site_capacity_factors = []
    for site in sites:
        cf = calculate_capacity_factor(solar_data_dict[site['file']], site['capacity'])
        site_capacity_factors.append((site, cf))
    
    # Sort sites by capacity factor (highest first)
    site_capacity_factors.sort(key=lambda x: x[1], reverse=True)
    best_site = site_capacity_factors[0][0]
    
    with tab1:
        # Read and display the introduction
        with open('intro.md', 'r') as f:
            intro_text = f.read()
        st.markdown(intro_text)
    
    with tab2:
        st.header("Solar + battery system configuration")
        
        # Solar site selection
        st.subheader("Select a solar site.")
        # Create site options sorted by capacity factor (highest first)
        site_options = [f"{site['location_name']} (CF: {calculate_capacity_factor(solar_data_dict[site['file']], site['capacity']):.1%})" 
                       for site, _ in site_capacity_factors]
        selected_site = st.selectbox("Select solar site. See the map below for locations.", site_options, index=0)  # Default to first (highest CF) site
        selected_site_data = next(site for site, _ in site_capacity_factors 
                                if f"{site['location_name']} (CF: {calculate_capacity_factor(solar_data_dict[site['file']], site['capacity']):.1%})" == selected_site)

        # Display capacity factor for selected site
        selected_capacity_factor = calculate_capacity_factor(
            solar_data_dict[selected_site_data['file']], 
            selected_site_data['capacity']
        )
        st.markdown(f"**Selected site capacity factor:** {selected_capacity_factor:.1%}")

        # add the map
        map_fig = create_solar_sites_map(sites, solar_data_dict)
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Solar system parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System size")
            solar_capacity_mw = st.slider(
                "Solar plant DC capacity (MW)",
                min_value=DEFAULT_PARAMETERS['solar_capacity_mw']['min'],
                max_value=DEFAULT_PARAMETERS['solar_capacity_mw']['max'],
                value=st.session_state.solar_capacity,
                step=DEFAULT_PARAMETERS['solar_capacity_mw']['step'],
                help=DEFAULT_PARAMETERS['solar_capacity_mw']['citation'],
                key='solar_capacity_slider'
            )
            
            battery_capacity_mwh = st.slider(
                "Battery storage capacity (MWh)",
                min_value=DEFAULT_PARAMETERS['battery_capacity_mwh']['min'],
                max_value=DEFAULT_PARAMETERS['battery_capacity_mwh']['max'],
                value=st.session_state.battery_capacity,
                step=DEFAULT_PARAMETERS['battery_capacity_mwh']['step'],
                help=DEFAULT_PARAMETERS['battery_capacity_mwh']['citation'],
                key='battery_capacity_slider'
            )
            
            solar_construction_time = st.slider(
                "Solar plant construction time (years)",
                min_value=DEFAULT_PARAMETERS['solar_construction_time']['min'],
                max_value=DEFAULT_PARAMETERS['solar_construction_time']['max'],
                value=DEFAULT_PARAMETERS['solar_construction_time']['value'],
                step=DEFAULT_PARAMETERS['solar_construction_time']['step'],
                help=DEFAULT_PARAMETERS['solar_construction_time']['citation']
            )
            
            solar_replacement_cost_factor = st.slider(
                "Solar replacement cost factor",
                min_value=DEFAULT_PARAMETERS['solar_replacement_cost_factor']['min'],
                max_value=DEFAULT_PARAMETERS['solar_replacement_cost_factor']['max'],
                value=DEFAULT_PARAMETERS['solar_replacement_cost_factor']['value'],
                step=DEFAULT_PARAMETERS['solar_replacement_cost_factor']['step'],
                help="Fraction of initial cost required for solar panel replacement (typically 70% due to reduced installation costs)",
                format="%.2f"
            )
            
            round_trip_efficiency_pct = st.slider(
                "Battery round-trip efficiency (%)",
                min_value=DEFAULT_PARAMETERS['round_trip_efficiency_pct']['min'],
                max_value=DEFAULT_PARAMETERS['round_trip_efficiency_pct']['max'],
                value=DEFAULT_PARAMETERS['round_trip_efficiency_pct']['value'],
                step=DEFAULT_PARAMETERS['round_trip_efficiency_pct']['step'],
                help=f"{TOOLTIPS['battery_efficiency']} {DEFAULT_PARAMETERS['round_trip_efficiency_pct']['citation']}"
            )
        
        with col2:
            st.subheader("Cost parameters")
            solar_module_cost = st.number_input(
                "Solar and inverter cost ($/kW)",
                min_value=DEFAULT_PARAMETERS['solar_and_inverter_cost']['min'],
                max_value=DEFAULT_PARAMETERS['solar_and_inverter_cost']['max'],
                value=DEFAULT_PARAMETERS['solar_and_inverter_cost']['value'],
                step=DEFAULT_PARAMETERS['solar_and_inverter_cost']['step'],
                help="Cost per kilowatt of solar panel capacity and power conversion system, including installation. Source: Lazard Levelized Cost of Energy 2023.",
                format="%.0f"
            )
            
            battery_cost = st.number_input(
                "Battery cost ($/kWh)",
                min_value=DEFAULT_PARAMETERS['battery_cost']['min'],
                max_value=DEFAULT_PARAMETERS['battery_cost']['max'],
                value=DEFAULT_PARAMETERS['battery_cost']['value'],
                step=DEFAULT_PARAMETERS['battery_cost']['step'],
                help=f"{TOOLTIPS['battery_cost']} {DEFAULT_PARAMETERS['battery_cost']['citation']}",
                format="%.0f"
            )
            
            land_cost = st.number_input(
                "Land cost ($/acre)",
                min_value=DEFAULT_PARAMETERS['land_cost']['min'],
                max_value=DEFAULT_PARAMETERS['land_cost']['max'],
                value=DEFAULT_PARAMETERS['land_cost']['value'],
                step=DEFAULT_PARAMETERS['land_cost']['step'],
                help=DEFAULT_PARAMETERS['land_cost']['citation'],
                format="%.0f"
            )
            
            solar_maintenance_cost = st.number_input(
                "Annual maintenance cost ($/kW-year)",
                min_value=DEFAULT_PARAMETERS['solar_maintenance_cost']['min'],
                max_value=DEFAULT_PARAMETERS['solar_maintenance_cost']['max'],
                value=DEFAULT_PARAMETERS['solar_maintenance_cost']['value'],
                step=DEFAULT_PARAMETERS['solar_maintenance_cost']['step'],
                help=f"{DEFAULT_PARAMETERS['solar_maintenance_cost']['citation']}",
                format="%.2f"
            )
        
        # Add optimization section
        st.subheader("Optimize system size")
        st.markdown("Use the controls below to choose the optimal combination of battery and solar capacity to meet a chosen reliability level.")
        col1, col2 = st.columns(2)
        
        with col1:
            target_reliability = st.slider(
                "Target reliability (%)",
                min_value=80.0,
                max_value=99.9,
                value=95.0,
                step=0.1,
                help="Target reliability for the solar + battery system"
            )
            
            if st.button("Optimize", help="Find the minimum cost combination of solar and battery capacity that meets the target reliability"):
                # Create progress bar
                progress_bar = st.progress(0.0)
                
                # Run optimization
                optimization_result = optimize_solar_battery(
                    solar_data_dict[selected_site_data['file']],
                    target_power_mw,
                    target_reliability / 100,  # Convert to decimal
                    DEFAULT_PARAMETERS['solar_and_inverter_cost']['value'],
                    DEFAULT_PARAMETERS['battery_cost']['value'],
                    DEFAULT_PARAMETERS['land_cost']['value'],
                    DEFAULT_PARAMETERS['round_trip_efficiency_pct']['value'],
                    progress_bar
                )
                
                if optimization_result:
                    # Update session state with optimal values
                    st.session_state.solar_capacity = optimization_result['solar_capacity']
                    st.session_state.battery_capacity = optimization_result['battery_capacity']
                    
                    # Show results
                    st.success(f"""
                    Optimization complete! Optimal configuration:
                    - Solar capacity: {optimization_result['solar_capacity']:,.0f} MW
                    - Battery capacity: {optimization_result['battery_capacity']:,.0f} MWh
                    - Achieved reliability: {optimization_result['reliability']:.1%}
                    - Total upfront cost: ${optimization_result['total_cost']:,.0f}
                    """)
                    
                    # Store optimization results for comparison
                    if 'optimization_history' not in st.session_state:
                        st.session_state.optimization_history = []
                    
                    st.session_state.optimization_history.append({
                        'target_reliability': target_reliability,
                        'solar_capacity': optimization_result['solar_capacity'],
                        'battery_capacity': optimization_result['battery_capacity'],
                        'achieved_reliability': optimization_result['reliability'],
                        'total_cost': optimization_result['total_cost']
                    })
                    
                    # Force a rerun to update the sliders
                    st.rerun()
        
        with col2:
            # Display optimization history
            if 'optimization_history' in st.session_state and st.session_state.optimization_history:
                st.subheader("Optimization history")
                history_df = pd.DataFrame(st.session_state.optimization_history)
                
                # Rename columns to be more user-friendly
                column_renames = {
                    'target_reliability': 'Target Reliability',
                    'solar_capacity': 'Solar Capacity',
                    'battery_capacity': 'Battery Capacity',
                    'achieved_reliability': 'Achieved Reliability',
                    'total_cost': 'Total Cost'
                }
                history_df = history_df.rename(columns=column_renames)
                
                # Format the values
                history_df['Target Reliability'] = history_df['Target Reliability'].map('{:.1f}%'.format)
                history_df['Achieved Reliability'] = (history_df['Achieved Reliability'] * 100).map('{:.1f}%'.format)
                history_df['Total Cost'] = history_df['Total Cost'].map('${:,.0f}'.format)
                history_df['Solar Capacity'] = history_df['Solar Capacity'].map('{:.0f} MW'.format)
                history_df['Battery Capacity'] = history_df['Battery Capacity'].map('{:.0f} MWh'.format)
                
                # Reorder columns for better readability
                column_order = [
                    'Target Reliability',
                    'Achieved Reliability',
                    'Solar Capacity',
                    'Battery Capacity',
                    'Total Cost'
                ]
                history_df = history_df[column_order]
                
                st.dataframe(history_df, use_container_width=True)
                
                if st.button("Clear history"):
                    st.session_state.optimization_history = []
                    st.rerun()
        
        # Run simulation
        battery_soc, power_output, battery_charge, battery_discharge = simulate_battery_storage(
            solar_data_dict[selected_site_data['file']],
            solar_capacity_mw,
            battery_capacity_mwh,
            target_power_mw,
            round_trip_efficiency_pct
        )
        
        # Weekly data visualization
        st.subheader("Weekly solar power, battery, and plant power data")
        unique_dates = solar_data_dict[selected_site_data['file']]['LocalTime'].dt.date.unique()
        week_start_dates = [d for d in unique_dates if d.weekday() == 0]
        selected_week_start = st.selectbox(
            "Select a start date for the graphic below. Note that the data come from a NREL model of 2006, but hopefully are useful to get a rough understanding of what the plant would do for a given time of year.",
            options=week_start_dates,
            format_func=lambda x: x.strftime("%B %d")
        )
        
        weekly_fig = plot_weekly_data(
            solar_data_dict[selected_site_data['file']],
            power_output,
            battery_soc,
            battery_charge,
            battery_discharge,
            selected_week_start,
            solar_capacity_mw
        )
        st.plotly_chart(weekly_fig, use_container_width=True)
    
    with tab3:
        st.header("Nuclear system configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Construction parameters")
            nuclear_construction_cost = st.number_input(
                "Nuclear construction cost ($/kW)",
                min_value=DEFAULT_PARAMETERS['nuclear_construction_cost']['min'],
                max_value=DEFAULT_PARAMETERS['nuclear_construction_cost']['max'],
                value=DEFAULT_PARAMETERS['nuclear_construction_cost']['value'],
                step=DEFAULT_PARAMETERS['nuclear_construction_cost']['step'],
                help=DEFAULT_PARAMETERS['nuclear_construction_cost']['citation']
            )
            
            nuclear_construction_time = st.slider(
                "Nuclear construction time (years)",
                min_value=DEFAULT_PARAMETERS['nuclear_construction_time']['min'],
                max_value=DEFAULT_PARAMETERS['nuclear_construction_time']['max'],
                value=DEFAULT_PARAMETERS['nuclear_construction_time']['value'],
                step=DEFAULT_PARAMETERS['nuclear_construction_time']['step'],
                help=DEFAULT_PARAMETERS['nuclear_construction_time']['citation']
            )
        
        with col2:
            st.subheader("Operating parameters")
            nuclear_capacity_factor = st.slider(
                "Nuclear capacity factor (%)",
                min_value=DEFAULT_PARAMETERS['nuclear_capacity_factor']['min'],
                max_value=DEFAULT_PARAMETERS['nuclear_capacity_factor']['max'],
                value=DEFAULT_PARAMETERS['nuclear_capacity_factor']['value'],
                step=DEFAULT_PARAMETERS['nuclear_capacity_factor']['step'],
                help=f"{TOOLTIPS['capacity_factor']} {DEFAULT_PARAMETERS['nuclear_capacity_factor']['citation']}"
            )
            
            nuclear_fuel_cost = st.number_input(
                "Nuclear fuel cost ($/MWh)",
                min_value=DEFAULT_PARAMETERS['nuclear_fuel_cost']['min'],
                max_value=DEFAULT_PARAMETERS['nuclear_fuel_cost']['max'],
                value=DEFAULT_PARAMETERS['nuclear_fuel_cost']['value'],
                step=DEFAULT_PARAMETERS['nuclear_fuel_cost']['step'],
                help=f"{TOOLTIPS['fuel_cost']} {DEFAULT_PARAMETERS['nuclear_fuel_cost']['citation']}"
            )
            
            nuclear_maintenance_cost = st.number_input(
                "Nuclear maintenance cost ($/kW-year)",
                min_value=DEFAULT_PARAMETERS['nuclear_maintenance_cost']['min'],
                max_value=DEFAULT_PARAMETERS['nuclear_maintenance_cost']['max'],
                value=DEFAULT_PARAMETERS['nuclear_maintenance_cost']['value'],
                step=DEFAULT_PARAMETERS['nuclear_maintenance_cost']['step'],
                help=DEFAULT_PARAMETERS['nuclear_maintenance_cost']['citation']
            )
            
            nuclear_decommissioning_cost = st.number_input(
                "Nuclear decommissioning cost ($/kW)",
                min_value=DEFAULT_PARAMETERS['nuclear_decommissioning_cost']['min'],
                max_value=DEFAULT_PARAMETERS['nuclear_decommissioning_cost']['max'],
                value=DEFAULT_PARAMETERS['nuclear_decommissioning_cost']['value'],
                step=DEFAULT_PARAMETERS['nuclear_decommissioning_cost']['step'],
                help=DEFAULT_PARAMETERS['nuclear_decommissioning_cost']['citation']
            )
    
    with tab4:
        st.header("Results & analysis")
        
        # Calculate costs
        solar_costs = calculate_solar_costs(
            solar_capacity_mw, 
            battery_capacity_mwh,
            solar_module_cost_usd_per_kw=solar_module_cost,
            battery_cost_usd_per_kwh=battery_cost,
            land_cost_usd_per_acre=land_cost,
            maintenance_cost_usd_per_kw_year=solar_maintenance_cost
        )
        
        nuclear_costs = calculate_nuclear_costs(
            nuclear_construction_time_years=nuclear_construction_time,
            nuclear_capacity_factor_pct=nuclear_capacity_factor,
            nuclear_fuel_cost_usd_per_mwh=nuclear_fuel_cost,
            nuclear_maintenance_cost_usd_per_kw_year=nuclear_maintenance_cost,
            nuclear_decommissioning_cost_usd_per_kw=nuclear_decommissioning_cost,
            nuclear_construction_cost_usd_per_kw=nuclear_construction_cost
        )
        
        # Calculate reliability
        reliability = calculate_reliability(power_output, target_power_mw)
        
        # Calculate annual energy production
        solar_annual_energy = np.sum(power_output)  # MWh
        nuclear_annual_energy = FACILITY_SIZE_MW * HOURS_PER_YEAR * nuclear_costs['nuclear_capacity_factor']
        
        # Calculate present value costs
        pv_costs = calculate_present_value_costs(
            solar_costs, 
            nuclear_costs, 
            discount_rate_pct,
            solar_replacement_cost_factor
        )
        
        # Calculate LCOE
        solar_lcoe, nuclear_lcoe = calculate_lcoe(
            solar_costs, 
            nuclear_costs, 
            power_output, 
            discount_rate_pct,
            solar_replacement_cost_factor
        )
        
        # Calculate LCOE components
        solar_components, nuclear_components, solar_lcoe, nuclear_lcoe = calculate_lcoe_components(
            pv_costs,
            solar_costs,
            nuclear_costs,
            solar_annual_energy,
            nuclear_annual_energy,
            discount_rate_pct,
            nuclear_construction_time,
            solar_costs.get('solar_construction_time', 2)
        )
        
        # Create comparison table
        st.subheader("System comparison")
        comparison_df = create_comparison_table(
            solar_costs, 
            nuclear_costs, 
            solar_capacity_mw, 
            battery_capacity_mwh, 
            power_output, 
            reliability,
            discount_rate_pct,
            solar_lcoe,
            nuclear_lcoe,
            solar_construction_time
        )
        st.dataframe(
            comparison_df,
            use_container_width=True,
            height=len(comparison_df) * 35 + 38,  # Adjust height based on number of rows
            hide_index=True  # Hide the index column
        )
        
        # Add title above the metric selector
        st.subheader("Cost components and sensitivity analysis")
        
        # Add metric selector
        metric_type = st.selectbox(
            "Select cost metric",
            options=["Present value cost", "LCOE"],
            index=0 if st.session_state.metric_type == "Present value cost" else 1,
            key="metric_type"
        )
        
        # Create columns for the charts
        col1, col2 = st.columns(2)
        
        with col1:
            if metric_type == "Present value cost":
                fig = create_cost_comparison_chart(solar_costs, nuclear_costs, discount_rate_pct)
                st.plotly_chart(fig, use_container_width=True)
            else:  # LCOE
                # Create LCOE comparison chart
                fig = go.Figure()
                
                # Add solar components
                solar_y = 0
                for component, value in solar_components.items():
                    fig.add_trace(go.Bar(
                        name=component,
                        x=['Solar'],
                        y=[value],
                        marker_color=COLORS[component],
                        text=[f"{component}: ${value:.1f}/MWh"],
                        textposition='auto',
                        textfont=dict(size=10, color='black'),
                        hovertemplate=f"{component}: ${value:.1f}/MWh<extra></extra>"
                    ))
                    solar_y += value
                
                # Add nuclear components
                nuclear_y = 0
                for component, value in nuclear_components.items():
                    fig.add_trace(go.Bar(
                        name=component,
                        x=['Nuclear'],
                        y=[value],
                        marker_color=COLORS[component],
                        text=[f"{component}: ${value:.1f}/MWh"],
                        textposition='auto',
                        textfont=dict(size=10, color='black'),
                        hovertemplate=f"{component}: ${value:.1f}/MWh<extra></extra>"
                    ))
                    nuclear_y += value
                
                fig.update_layout(
                    yaxis_title='LCOE ($/MWh)',
                    barmode='stack',
                    showlegend=False,
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        tickmode='array',
                        ticktext=['Solar', 'Nuclear'],
                        tickvals=[0, 1],
                        tickangle=0,
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        tickformat='$.1f',
                        tickfont=dict(size=12)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                        
        with col2:
            fig_lcoe, fig_cost = create_sensitivity_analysis(solar_costs, nuclear_costs, power_output)
            if metric_type == "Present value cost":
                st.plotly_chart(fig_cost, use_container_width=True)
            else:  # LCOE
                st.plotly_chart(fig_lcoe, use_container_width=True)

if __name__ == "__main__":
    main() 