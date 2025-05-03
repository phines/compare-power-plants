# Solar vs Nuclear Power Cost Comparison

This Streamlit application compares the costs of supplying 24/7 power to a 600 MW facility using either solar power with storage or nuclear power over a 40-year period.

## Features

- Interactive cost comparison between solar and nuclear power systems
- Adjustable parameters for both power generation methods
- Present value cost calculations over 40 years
- Detailed cost breakdowns
- Interactive visualizations

## Setup

1. Clone this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To run the application, execute:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Use the sidebar sliders to adjust various parameters for both solar and nuclear systems
2. View the cost comparison chart in the main area
3. Check the detailed cost breakdowns below the chart

## Parameters

### Solar System Parameters
- Solar Panel Efficiency
- Annual Degradation Rate
- Panel Lifetime
- Battery Round-trip Efficiency
- Battery Lifetime

### Nuclear System Parameters
- Construction Time
- Capacity Factor
- Fuel Cost per MWh

### Economic Parameters
- Discount Rate

## Notes

- All costs are calculated in present value terms
- The comparison includes initial costs, maintenance, replacements, and decommissioning
- The solar system includes battery storage for 24/7 operation
- The nuclear system includes construction, fuel, and decommissioning costs 