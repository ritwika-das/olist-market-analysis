import pandas as pd
import geopandas as gpd
from unidecode import unidecode
import numpy as np

def clean_orders(olist_orders_dataset):
    """
    Cleans the input dataset by removing outliers based on the number of orders per day and filtering
    data based on specific date conditions.

    Steps performed:
    1. Groups the dataset by the purchase date and calculates the number of orders per day.
    2. Detects outlier dates based on the Interquartile Range (IQR) of the order counts.
    3. Removes orders from outlier dates.
    4. Removes orders from the year 2016.
    5. Removes orders that occurred after August 1st, 2018.

    Parameters:
    -----------
    olist_orders_dataset : pandas.DataFrame
        A DataFrame containing order information, where one row represents one order.
        The dataset must contain a column 'order_purchase_timestamp' with datetime values.

    Returns:
    --------
    pandas.DataFrame
        A cleaned version of the input dataset with outliers removed and the specified date filters applied.
    """

    olist_orders_dataset["order_purchase_timestamp"] = pd.to_datetime(olist_orders_dataset["order_purchase_timestamp"])

    # Group by date and count the number of orders
    orders_by_date = olist_orders_dataset.groupby(olist_orders_dataset['order_purchase_timestamp'].dt.date).size()

    # Remove outliers based on the number of orders per day using IQR
    Q1 = orders_by_date.quantile(0.25)
    Q3 = orders_by_date.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify dates with order counts outside the IQR range
    outlier_dates = orders_by_date[(orders_by_date < lower_bound) | (orders_by_date > upper_bound)].index

    # Remove orders from the outlier dates
    orders_clean = olist_orders_dataset[~olist_orders_dataset['order_purchase_timestamp'].dt.date.isin(outlier_dates)]

    # Remove data from 2016 as they seems to be
    orders_clean = orders_clean[orders_clean['order_purchase_timestamp'].dt.year != 2016]

    # Remove data after September 2018
    orders_clean = orders_clean[orders_clean['order_purchase_timestamp'] < '2018-08-01']

    # Add derived time columns
    orders_clean['order_year'] = orders_clean['order_purchase_timestamp'].dt.year
    orders_clean['order_month'] = orders_clean['order_purchase_timestamp'].dt.to_period('M')
    orders_clean['order_weekday'] = orders_clean['order_purchase_timestamp'].dt.weekday
    orders_clean['order_hour'] = orders_clean['order_purchase_timestamp'].dt.hour
    orders_clean['order_quarter'] = orders_clean['order_purchase_timestamp'].dt.quarter
    orders_clean['order_date'] = orders_clean['order_purchase_timestamp'].dt.date

    return orders_clean

def transform_demographic_information(df):
    """
    Transforms demographic information in a DataFrame by calculating the state population,
    its proportion of the total population, and the distribution percentages for specific columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing demographic information, including columns for gender-based population,
        ethnic groups, color, and literacy statistics.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with new columns for state population, proportion of total population,
        and percentage distributions for selected columns.
    """
    # Calculate the total population of each state
    df["State population"] = df["Gender  - Males"] + df["Gender  - Females"]

    # Cluster countries into two categories
    df["IDHM_low"] = df["IDHM"] < 0.725

    # Calculate the total population across all states
    total_population = sum(df["State population"])

    # Add a new column for the percentage of the total population each state represents
    df["Proc of total population"] = df["State population"] / total_population * 100

    # Define columns for which to calculate the percentage distribution
    selected_col = [
        'Ethnic Group  - Indigenous', 'Ethnic Group  - Non-Indigenous',
        'Color  - Black', 'Color  - White', 'Color  - Mixed / Other',
        'Literacy (A15+)  - yes', 'Literacy (A15+)  - no'
    ]

    # Calculate percentage distribution for selected columns based on state population
    for col in selected_col:
        # Ensure the column is numeric before calculation
        df[col] = df[col].astype(float)

        # Convert the values to percentage of the state population
        df[col] = df[col] / df["State population"] * 100

    return df

def upload_map(filepath):

    brazil_map = gpd.read_file(filepath)
    brazil_map["original_name"] = brazil_map["NAME_1"]
    brazil_map["NAME_1"] = brazil_map["NAME_1"].str.lower().apply(unidecode)

    return brazil_map

def get_orders_per_month(data):

    return data['order_month'].value_counts().sort_index()

def get_avg_orders_per_weekday(data):
    orders_per_weekday = data.groupby('order_weekday').size()
    unique_days_per_weekday = data.groupby('order_weekday')['order_date'].nunique()

    return orders_per_weekday / unique_days_per_weekday

def get_avg_orders_per_hour(data):
    hourly_orders = data.groupby(['order_date', 'order_hour']).size().reset_index(name='hourly_order_count')

    return hourly_orders.groupby('order_hour')['hourly_order_count'].mean()

def clean_product_category_names(orders_products):
    """
    Cleans and standardizes product category names in the dataset.

    Parameters:
    orders_products (DataFrame): The dataset containing product category information.

    Returns:
    DataFrame: The updated DataFrame with cleaned product category names.
    """
    # Replace underscores with spaces and capitalize words
    orders_products["product_category_name_english"] = orders_products["product_category_name_english"].str.replace('_', ' ').str.title()

    # Shorten long category names for readability
    orders_products.loc[orders_products["product_category_name_english"] == "Computers Accessories",
                        "product_category_name_english"] = "Computers Acc."

    return orders_products

def calculate_category_statistics(orders_products):
    """
    Calculates the distribution.

    Parameters:
    orders_products (DataFrame): The dataset containing product category information.

    Returns:
    DataFrame: A DataFrame containing:
        - 'product_category_name_english': Name of the category.
        - 'category_size': Count of orders per category.
        - 'category_percentage': Percentage distribution of categories.
    """
    # Count occurrences of each category
    category_counts = orders_products.groupby('product_category_name_english').agg(category_size=('product_category_name_english', 'size'), avg_price=('price', 'mean')).round(2).reset_index()

    # Calculate percentage distribution of categories
    category_counts["category_percentage"] = category_counts["category_size"] / category_counts["category_size"].sum() * 100

    # Sort categories for better visualization
    category_counts = category_counts.sort_values(by="category_size", ascending=False).reset_index(drop=True)

    return category_counts

def top_n_categories(category_counts, n):
    """
    Filters categories based on a percentage threshold.

    Parameters:
    category_counts (DataFrame): A DataFrame containing category statistics, including percentage distribution.
                                 Expected columns: ['product_category_name_english', 'category_size', 'category_percentage'].
    n (float): The percentage threshold. Categories with a percentage greater than 'n' will be included.

    Returns:
    DataFrame: A filtered DataFrame containing categories with a percentage greater than 'n'.
    """

    return category_counts.iloc[:n]

def add_others_category(orders_products, n):
    """
    Adds an 'Others' category to the product categories DataFrame for categories
    whose percentage is below or equal to the given threshold.

    This function calculates the distribution of product categories, filters categories
    based on their percentage exceeding the specified threshold, and aggregates the
    categories that fall below or equal to the threshold into an 'Others' category.

    Parameters:
    orders_products (DataFrame): A DataFrame containing product order data with
                                  at least the columns 'product_category_name_english',
                                  'category_size', and 'category_percentage'.
    n (float): The percentage threshold used to filter product categories. Categories
               with a percentage greater than `n` are kept, and the rest are aggregated into 'Others'.

    Returns:
    DataFrame: A DataFrame containing the filtered product categories along with an
               'Others' category that aggregates all categories with a percentage
               below or equal to `n`.
    """

    # Calculate the distribution
    category_counts = calculate_category_statistics(orders_products)

    # Filter categories above the threshold
    filtered_categories = category_counts[category_counts['category_percentage'] > n]

    # Aggregate the "Others" category for categories below or equal to the threshold
    others_size = category_counts[category_counts['category_percentage'] <= n]['category_size'].sum()
    others_percentage = category_counts[category_counts['category_percentage'] <= n]['category_percentage'].sum()

    # Create a DataFrame for the "Others" category
    others_row = pd.DataFrame({
        'product_category_name_english': ['Others'],
        'category_size': [others_size],
        'category_percentage': [others_percentage]
    })

    # Concatenate the filtered categories with the "Others" category
    filtered_cat_with_others = pd.concat([filtered_categories, others_row], ignore_index=True)

    return filtered_cat_with_others

def pivot_monthly_orders(orders_products, n):
    """
    Prepares a pivot table for monthly order counts filtered by category percentage threshold.

    Parameters:
    orders_products (DataFrame): The dataset containing order information, including order dates and product categories.
                                  Expected columns:
                                  - 'order_month': Month of the order (datetime or string format).
                                  - 'product_category_name_english': Name of the product category.
    n (float): The percentage threshold for filtering categories. Only categories with a percentage greater than this value will be included.

    Returns:
    DataFrame:
        A pivot table where:
        - Rows represent order months.
        - Columns represent product categories (filtered by the threshold `n`).
        - Values represent the count of orders for each category per month.
    """
    # Group monthly order counts by category
    monthly_orders = (
        orders_products
        .groupby(['order_month', 'product_category_name_english'])
        .size()
        .reset_index(name='order_count')
    )

    # Calculate category statistics
    category_counts = calculate_category_statistics(orders_products)

    # Filter categories based on the percentage threshold
    filtered_categories = top_n_categories(category_counts, n)

    # Retain only rows for the filtered categories
    monthly_orders = monthly_orders[
        monthly_orders['product_category_name_english'].isin(filtered_categories['product_category_name_english'])
    ]

    # Create a pivot table
    monthly_orders_pivot = monthly_orders.pivot(
        index='order_month',
        columns='product_category_name_english',
        values='order_count'
    )

    return monthly_orders_pivot

def calculate_rolling_avg(orders_products, n):
    """
    Calculates the rolling average of monthly order counts over a 3-month window for filtered categories.

    Parameters:
    orders_products (DataFrame): The dataset containing order information, including order dates and product categories.
                                  Expected columns:
                                  - 'order_month': Month of the order (datetime or string format).
                                  - 'product_category_name_english': Name of the product category.
    n (float): The percentage threshold for filtering categories. Only categories with a percentage greater than this value will be included in the calculations.

    Returns:
    DataFrame:
        A DataFrame representing the rolling average of order counts for each product category, calculated over a 3-month window.
        - Rows represent order months.
        - Columns represent product categories (filtered by the threshold `n`).
        - Values represent the 3-month rolling average of order counts.
    """
    # Generate the pivot table for monthly order counts
    monthly_orders_pivot = pivot_monthly_orders(orders_products, n)

    # Calculate the rolling average with a 3-month window
    rolling_avg = monthly_orders_pivot.rolling(window=3).mean()

    return rolling_avg

def calculate_annual_differences(orders_products, n):
    """
    Calculates the annual differences in order counts for the top 'n' product categories
    between the years 2017 and 2018, based on orders placed before July.

    Args:
        orders_products (DataFrame): A DataFrame containing order data, including columns
                                      for 'order_purchase_timestamp' and 'product_category_name_english'.
        n (int): The number of top product categories to consider based on order count.

    Returns:
        DataFrame: A DataFrame showing the order counts for 2017 and 2018, the differences between the years,
                   and the sum of order counts for each category, sorted by the total order count.
    """

    # Calculate category statistics
    category_counts = calculate_category_statistics(orders_products)

    # Filter categories based on the percentage threshold
    filtered_categories = top_n_categories(category_counts, n)

    # Filter orders to include only those in the top 'n' categories
    data_filtered = orders_products[orders_products['product_category_name_english'].isin(filtered_categories['product_category_name_english'])]

    # Filter for orders in 2017 and 2018, and only those before July
    data_filtered = data_filtered[
        (data_filtered['order_purchase_timestamp'].dt.year.isin([2017, 2018])) &
        (data_filtered['order_purchase_timestamp'].dt.month <= 6)
    ]

    # Group by year and category, calculating the order count
    total_orders = data_filtered.groupby(['order_year', 'product_category_name_english']).size().reset_index(name='order_count')

    # Pivot data to compare order counts between 2017 and 2018
    pivot_data = total_orders.pivot_table(index='product_category_name_english', columns='order_year', values='order_count')

    # Calculate the difference and total for each category
    pivot_data['difference'] = pivot_data[2018] - pivot_data[2017]
    pivot_data['sum'] = pivot_data[2018] + pivot_data[2017]

    # Sort by the total order count in ascending order
    pivot_data = pivot_data.sort_values(by='sum', ascending=True)

    return pivot_data

def prepare_product_distribution_by_state(orders_products):
    """
    Prepares a pivoted DataFrame for visualizing proportions of product category orders
    by state and socioeconomic index (IDHM).

    Parameters:
        orders_products (pd.DataFrame): DataFrame containing order, product, and state information.

    Returns:
        pd.DataFrame: Pivoted DataFrame with proportions of product categories by state.
    """
    # Step 1: Calculate the average price by product category
    state_category_price = (
        orders_products.groupby(['product_category_name_english'])
        .agg({'price': 'mean'})
        .round(2)
    )

    # Step 2: Calculate the order count for each category by state and IDHM
    state_category_count = (
        orders_products.groupby(['customer_state', 'IDHM', 'product_category_name_english'])
        .size()
        .reset_index(name='order_count')
    )

    # Merge the average price and order count data
    state_category = pd.merge(state_category_price, state_category_count, on=['product_category_name_english'])

    # Step 3: Normalize the order count to get proportions
    state_totals = state_category.groupby('customer_state')['order_count'].transform('sum')
    state_category['proportion'] = state_category['order_count'] / state_totals

    # Step 4: Sort product categories by average price and IDHM
    state_category = state_category.sort_values(by=['IDHM', 'price'], ascending=[True, False]).reset_index(drop=True)

    # Pivot the data for visualization
    state_category_pivot = state_category.pivot(
        index='customer_state',
        columns='product_category_name_english',
        values='proportion'
    ).fillna(0)

    # Sort columns in pivot table by average price
    sorted_categories = state_category.sort_values(by='price', ascending=False)['product_category_name_english'].unique()
    state_category_pivot = state_category_pivot[sorted_categories]

    # Step 5: Order the states based on their occurrence in the data
    state_order = orders_products['customer_state'].drop_duplicates().tolist()
    state_category_pivot.index = pd.Categorical(state_category_pivot.index, categories=state_order, ordered=True)
    state_category_pivot = state_category_pivot.sort_index()

    return state_category_pivot

    """
    Generates a word cloud for reviews based on the IDHM category (low or high).

    Args:
        data (pd.DataFrame): DataFrame containing the reviews with IDHM categories.
        is_low_idhm (bool): A boolean flag to select either low IDHM (True) or high IDHM (False) reviews.

    Returns:
        WordCloud: A WordCloud object generated based on the reviews for the selected IDHM category.
    """
    reviews = " ".join(data.loc[data['IDHM_low'] == is_low_idhm, 'review_comment_message_english'].dropna())

    coolwarm = plt.cm.coolwarm
    colormap = LinearSegmentedColormap.from_list(
        "low_coolwarm", coolwarm(np.linspace(0, 0.5, 100))
    ) if is_low_idhm else LinearSegmentedColormap.from_list(
        "high_coolwarm", coolwarm(np.linspace(0.5, 1, 100))
    )

    wordcloud = WordCloud(
        background_color="white",
        colormap=colormap,
        stopwords=ENGLISH_STOP_WORDS,
        width=800,
        height=400
    ).generate(reviews)

    return wordcloud

def prepare_most_popular_categories_by_state(orders_products, top_n_categories):
    """
    Prepares a pivot table of normalized order data for product categories by state.

    Args:
        orders_products (pd.DataFrame): The DataFrame containing orders and product details with states and populations.
                                         Must include columns: 'product_category_name_english', 'State', and 'State population'.
        top_n_categories (int): Number of top product categories to include in the analysis.

    Returns:
        pd.DataFrame: Transposed pivot table of normalized orders per habitant.
    """
    # Step 1: Aggregate data by product category and state to calculate total orders
    state_category_orders = orders_products.groupby(
        ['product_category_name_english', 'State']
    ).size().reset_index(name='total_orders')

    # Step 2: Calculate orders per habitant for each category and state
    state_category_orders['orders_per_habitant'] = round(
        state_category_orders['total_orders'] /
        orders_products.groupby('State')['State population'].transform('first') * 1000000, 2
    )

    # Step 3: Identify the top N most common product categories
    top_categories = state_category_orders.groupby('product_category_name_english')['total_orders'].sum() \
        .nlargest(top_n_categories).index

    # Step 4: Filter for only the top N categories
    state_category_filtered = state_category_orders[
        state_category_orders['product_category_name_english'].isin(top_categories)
    ]

    # Step 5: Pivot the data to create a matrix with states as columns and categories as rows
    state_category_pivot = state_category_filtered.pivot(
        index='product_category_name_english',
        columns='State',
        values='orders_per_habitant'
    ).fillna(0)

    # Step 6: Transpose the data to switch x and y axes
    state_category_pivot_transposed = state_category_pivot.T

    return state_category_pivot_transposed


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    The function uses the Haversine formula to compute the shortest distance over the Earth's surface
    between two geographic coordinates given in degrees (latitude and longitude).

    Parameters:
    ----------
    lat1 : float
        Latitude of the first point in degrees.
    lon1 : float
        Longitude of the first point in degrees.
    lat2 : float
        Latitude of the second point in degrees.
    lon2 : float
        Longitude of the second point in degrees.

    Returns:
    -------
    int
        The distance between the two points in kilometers, rounded to the nearest integer.
    """
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return int(distance)
