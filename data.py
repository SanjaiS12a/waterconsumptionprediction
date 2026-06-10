import pandas as pd
import numpy as np

np.random.seed(42)

# =====================================================
# DATE RANGE (6 YEARS)
# =====================================================

dates = pd.date_range(
    start="2020-01-01",
    end="2025-12-31",
    freq="D"
)

n = len(dates)

# =====================================================
# TEMPORAL FEATURES
# =====================================================

df = pd.DataFrame()

df["Date"] = dates
df["Day_of_Week"] = dates.dayofweek
df["Month"] = dates.month
df["Year"] = dates.year
df["Day_of_Year"] = dates.dayofyear

df["Is_Weekend"] = (
    dates.dayofweek >= 5
).astype(int)

# =====================================================
# HOLIDAYS
# =====================================================

holidays = set()

for y in range(2020, 2026):

    holiday_list = [

        (1, 26),
        (4, 14),
        (5, 1),
        (8, 15),
        (10, 2),
        (12, 25)

    ]

    for m, d in holiday_list:

        holidays.add(
            pd.Timestamp(
                year=y,
                month=m,
                day=d
            )
        )

df["Holiday_Flag"] = (
    dates.isin(holidays)
).astype(int)

# =====================================================
# WEATHER FEATURES
# =====================================================

temp_base = (
    28
    +
    8 *
    np.sin(
        2*np.pi*
        (df["Day_of_Year"]-100)
        /365
    )
)

df["Temperature_C"] = np.round(

    temp_base
    +
    np.random.normal(0,1.5,n),

    1
)

rain_prob = np.where(

    df["Month"].isin([10,11,12]),

    0.45,

    np.where(
        df["Month"].isin(
            [6,7,8,9]
        ),
        0.20,
        0.05
    )

)

rainfall = (

    np.random.exponential(
        8,
        n
    )

    *

    (
        np.random.rand(n)
        <
        rain_prob
    )

)

df["Rainfall_mm"] = np.round(
    rainfall,
    1
)

humidity_base = (

    65

    +

    20 *

    np.sin(
        2*np.pi*
        (df["Day_of_Year"]-270)
        /365
    )

)

df["Humidity_pct"] = np.clip(

    np.round(

        humidity_base

        +

        np.random.normal(
            0,
            5,
            n
        ),

        1
    ),

    30,
    98

)

evap_base = (

    4.5

    -

    2 *

    np.sin(
        2*np.pi*
        (df["Day_of_Year"]-180)
        /365
    )

)

df["Evaporation_Rate_mm"] = np.round(

    np.clip(

        evap_base

        +

        np.random.normal(
            0,
            0.4,
            n
        ),

        1,
        8

    ),

    2

)

# =====================================================
# POPULATION
# =====================================================

year_frac = (
    (dates - dates[0]).days
    /
    365.25
)

population_base = (

    1_000_000

    *

    (1.02 ** year_frac)

)

df["Population"] = np.round(

    population_base

    +

    np.random.normal(
        0,
        2000,
        n
    )

).astype(int)

df["Household_Count"] = (

    df["Population"]

    /

    3.8

).round().astype(int)

urban_base = (

    0.72

    +

    0.01 *

    year_frac

    /

    5

)

df["Urbanization_Index"] = np.round(

    np.clip(

        urban_base

        +

        np.random.normal(
            0,
            0.005,
            n
        ),

        0,
        1

    ),

    4

)

# =====================================================
# PUMP STATUS
# =====================================================

df["Pump_Status"] = np.where(

    np.random.rand(n) < 0.05,

    0,

    1

)

# =====================================================
# EXTREME EVENTS
# =====================================================

heatwave = (

    df["Month"].isin([4,5])

    &

    (
        np.random.rand(n)
        < 0.08
    )

)

heatwave_effect = np.where(

    heatwave,

    np.random.uniform(
        10,
        20,
        n
    ),

    0

)

drought = (

    df["Month"].isin([6,7,8])

    &

    (
        np.random.rand(n)
        < 0.05
    )

)

drought_effect = np.where(

    drought,

    np.random.uniform(
        8,
        15,
        n
    ),

    0

)

# =====================================================
# WEEKLY EFFECT
# =====================================================

weekly_pattern = np.array([

    4,
    3,
    2,
    2,
    1,
    -3,
    -5

])

weekly_effect = weekly_pattern[
    df["Day_of_Week"]
]

# =====================================================
# WATER DEMAND
# =====================================================

base_demand = 90

seasonal = (

    15

    *

    np.sin(

        2*np.pi

        *

        (
            df["Day_of_Year"]
            - 80
        )

        /365

    )

)

temp_effect = (

    0.8

    *

    (
        df["Temperature_C"]
        - 28
    )

)

rain_effect = (

    -0.3

    *

    df["Rainfall_mm"]

)

holiday_effect = (

    -5

    *

    df["Holiday_Flag"]

)

population_effect = (

    (
        df["Population"]
        - 1_000_000
    )

    *

    0.0003

)

noise = np.random.normal(
    0,
    1.5,
    n
)

daily_demand = (

    base_demand

    +

    seasonal

    +

    weekly_effect

    +

    temp_effect

    +

    rain_effect

    +

    holiday_effect

    +

    population_effect

    +

    heatwave_effect

    +

    drought_effect

    +

    noise

)

# =====================================================
# PUMP FAILURE EFFECT
# =====================================================

daily_demand = np.where(

    df["Pump_Status"] == 0,

    daily_demand * 0.85,

    daily_demand

)

# =====================================================
# TARGET
# =====================================================

df["Total_Daily_Usage_ML"] = np.round(

    np.clip(
        daily_demand,
        30,
        160
    ),

    3

)

# =====================================================
# CORRELATED FEATURES
# =====================================================

df["Treatment_Output_ML"] = np.round(

    df["Total_Daily_Usage_ML"]

    *

    1.05

    +

    np.random.normal(
        0,
        3,
        n
    ),

    2

)

df["Pressure_Level_bar"] = np.round(

    4

    -

    (
        (
            df["Total_Daily_Usage_ML"]
            -
            90
        )

        /

        100
    )

    +

    np.random.normal(
        0,
        0.15,
        n
    ),

    2

)

# =====================================================
# RESERVOIR LEVEL
# =====================================================

reservoir = []

level = 85

for i in range(n):

    level += (

        df["Rainfall_mm"].iloc[i]

        *

        0.05

        -

        df["Total_Daily_Usage_ML"].iloc[i]

        *

        0.02

    )

    level = np.clip(
        level,
        30,
        100
    )

    reservoir.append(level)

df["Reservoir_Level_pct"] = np.round(
    reservoir,
    2
)

# =====================================================
# PER CAPITA
# =====================================================

df["Per_Capita_Usage_L"] = np.round(

    (

        df["Total_Daily_Usage_ML"]

        *

        1_000_000

    )

    /

    df["Population"],

    2

)

# =====================================================
# DEMAND LEVEL
# =====================================================

df["Demand_Level"] = pd.cut(

    df["Total_Daily_Usage_ML"],

    bins=[
        0,
        80,
        110,
        200
    ],

    labels=[
        "Low",
        "Medium",
        "High"
    ]

)

# =====================================================
# SAVE CSV
# =====================================================

df.to_csv(
    "water_consumption_dataset.csv",
    index=False
)

print("Dataset Generated Successfully")
print(df.head())
print(df.shape)