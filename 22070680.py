import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data source: https://data.worldbank.org
df = pd.read_csv("military expenditure.csv")

if df["incomeLevel"].isnull().any():
  df.dropna(subset=["incomeLevel"], inplace=True)

columns_to_remove = ['iso3c', 'iso2c']

# Remove the specified columns
df = df.drop(columns=columns_to_remove)

df.fillna(0 , inplace = True)

column_mapping = {"Military expenditure (current USD)" : "Military_expenditure" ,
                  'Military expenditure (% of general government expenditure)' :
                  "Military_expenditure_%_government_expenditure" ,
                  "Military expenditure (% of GDP)" : "Military_expenditure_%_gdp"}

df.rename(columns = column_mapping , inplace = True)

df = df[df['incomeLevel'] != 'Not classified']

#Statistical Summary using pandas
df.describe()

#Statistical Summary using numpy
select_col = df.select_dtypes(include = ['int64' , 'float64']).columns

stats_numpy = df[select_col].agg([np.mean, np.median, np.std, np.min, np.max])

rows_to_r =  ['World', 'High income', 'OECD members', 'Post-demographic dividend',
       'North America',  'IDA & IBRD total',
       'Low & middle income', 'Middle income', 'IBRD only' , 'East Asia & Pacific', 'Upper middle income',
       'Late-demographic dividend', 'Europe & Central Asia',
       'East Asia & Pacific (excluding high income)',
       'East Asia & Pacific (IDA & IBRD countries)',
       'Early-demographic dividend', 'European Union' , 'Euro area', 'Lower middle income',
       'Middle East & North Africa',
       'Europe & Central Asia (IDA & IBRD countries)', 'Arab World',
       'Europe & Central Asia (excluding high income)',
       'South Asia (IDA & IBRD)', 'South Asia', 'Latin America & the Caribbean (IDA & IBRD countries)'
       , 'Fiji' , 'Middle East & North Africa (IDA & IBRD countries)	', "Middle East & North Africa (IDA & IBRD countries)"
       ,"Middle East & North Africa (excluding high income)", 'Latin America & Caribbean (excluding high income)']

df = df[~df['country'].isin(rows_to_r)]


def plot_top_countries_increase(df, ax=None):
    """
    Generate a line plot showing the military spending increase
    for the top 10 countries with the largest percentage increase.

    Parameters:
        df (DataFrame): The DataFrame containing military spending data.
        ax (Axes, optional): The axes on which to plot. If not provided, a new figure will be created.

    Returns:
        None. The function saves the plot as an image file if ax is not provided.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10 , 3))

    df_increase = df.groupby('country')['Military_expenditure'].last() - df.groupby('country')['Military_expenditure'].first()
    initial_expenditure = df.groupby('country')['Military_expenditure'].first()
    percentage_increase = (df_increase / initial_expenditure)

    percentage_increase[initial_expenditure == 0] = 0
    percentage_increase = percentage_increase.sort_values(ascending=False).head(10)

    top_10_countries = percentage_increase.index
    df_top_10 = df[df['country'].isin(top_10_countries)]

    sns.lineplot(x='year', y='Military_expenditure', hue='country', data=df_top_10, ax=ax)

    ax.set_title('Top 10 Countries with Largest Increase in Military Spending (1970-2020)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Military Expenditure (in Billion USD)')

    legend_labels = [f'{country} ({percentage:.2f}%)' for country, percentage in zip(top_10_countries, percentage_increase)]
    ax.legend(labels=legend_labels, loc='upper left')



def plot_military_expenditure_by_income_level(df, ax=None):
    """
    Generate a dodged bar plot showing the percentage of government
    expenditure on military by income level.

    Parameters:
        df (DataFrame): The DataFrame containing military spending data.
        ax (Axes, optional): The axes on which to plot. If not provided, a new figure will be created.

    Returns:
        fig (Figure): The matplotlib figure. Returns the figure if ax is not provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10 , 3))
    else:
        fig = plt.gcf()

    ax = sns.barplot(x='incomeLevel', y='Military_expenditure_%_government_expenditure', hue='incomeLevel',
                     dodge=True, data=df, edgecolor='None', ax=ax)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Percentage of Government Expenditure by Income Level on Military')
    ax.set_xlabel('Income Level')
    ax.set_ylabel('Military Expenditure (% of Government Expenditure)')

def plot_top_10_countries_2020(df, ax=None):
    """
    Generate a horizontal bar chart showing the top 10 countries with the
    highest military expenditure in the year 2020.

    Parameters:
        df (DataFrame): The DataFrame containing military spending data.
        ax (Axes, optional): The axes on which to plot. If not provided, a new figure will be created.

    Returns:
        fig (Figure): The matplotlib figure. Returns the figure if ax is not provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    else:
        fig = plt.gcf()

    df_2020 = df[df['year'] == 2020]
    top_10_countries = df_2020.sort_values(by='Military_expenditure', ascending=False).head(10)

    colors = sns.color_palette('viridis', n_colors=10)  # Use viridis color palette

    ax = sns.barplot(x='Military_expenditure', y='country', data=top_10_countries, palette=colors, ax=ax)

    for p in ax.patches:
        ax.annotate(f'{p.get_width() / 1e12:.2f} Trillion USD', (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points', color='black')

    ax.set_title('Top 10 Military Expenditure Countries in 2020')
    ax.set_xlabel('Military Expenditure (in Trillion USD)')
    ax.set_ylabel('Country')


def plot_top_countries(df, year=2020, threshold_percentage=5, ax=None):
    """
    Generate a pie chart showing the top 5 countries spending more than
    a specified percentage of their Military_expenditure_%_gdp in a
    given year.

    Parameters:
        df (DataFrame): The DataFrame containing military spending data.
        year (int): The year for which the analysis is done.
        threshold_percentage (float): The threshold percentage for filtering countries.
        ax (Axes, optional): The axes on which to plot. If not provided, a new figure will be created.

    Returns:
        fig (Figure): The matplotlib figure. Returns the figure if ax is not provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3))
    else:
        fig = plt.gcf()

    df_high_spending = df[(df['year'] == year) & (df['Military_expenditure_%_gdp'] > threshold_percentage)]
    df_high_spending = df_high_spending.sort_values(by='Military_expenditure_%_gdp', ascending=False).head(5)

    colors = plt.cm.tab10.colors

    # Pie chart with legend labels
    wedges, texts, autotexts = ax.pie(df_high_spending['Military_expenditure_%_gdp'],
                                       labels=df_high_spending['country'],
                                       autopct=lambda p: '{:.2f}%'.format(p) if p > 0 else '',
                                       colors=colors, startangle=90,
                                       wedgeprops=dict(width=0.4, edgecolor='w'))

    # Add legend with country names
    ax.legend(wedges, df_high_spending['country'], title='Country Names', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title(f'Top 5 Countries Spending More Than {threshold_percentage}% of GDP on Military Expenditure in ({year})')




# Create a single figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(21 , 21))
fig.suptitle("Military expenditure by country from 1970-2020", fontsize=32 , fontweight = "bold")
fig.set_facecolor('#FFFDD0')


# Plot Military Expenditure Increase
plot_top_countries_increase(df, ax=axes[0, 0])

# Plot Military Expenditure by Income Level
plot_military_expenditure_by_income_level(df, ax=axes[0, 1])

# Plot Top 10 Countries in 2020
plot_top_10_countries_2020(df, ax=axes[1, 0])

# Plot Top Countries by Percentage of GDP
plot_top_countries(df, year=2020, threshold_percentage=5, ax=axes[1, 1])

# Add a common description
description = ("The United States leads global military spending at 0.78 trillion,"
"showcasing a considerable financial commitment to defense. China and India follow, "
"contributing 0.25 trillion and 0.07 trillion, respectively. This emphasizes the significant disparities in military expenditures among nations."

"The drastic upward trends in military spending for certain countries highlight shifts"
"in global security concerns and regional dynamics, shaping the trajectory of defense investment"
"Saudi Arabia stands out with an extraordinary 120-fold increase in military spending over the past five decades,"

"Low-income countries allocate the highest percentage of government expenditure to"
"the military, with a notable 4.49%. Lower-middle-income and upper-middle-income countries follow closely, allocating 2.98% and 2.67%, respectively."


"The pie chart reveals the economic impact of military spending, emphasizing how certain"
"countries dedicate a significant portion of their GDP to defense, potentially affecting other sectors of their economies."
"Oman leads in allocating a substantial 28.5% of its GDP to military expenditure,"
"followed by Saudi Arabia (22%), Algeria (17%), Kuwait (17%), and Israel (15%).")
# Add student information
student_info = (
    "     \n\nName:praveenraj gnanakumar "
    "     \nStudent ID : 22070680"
)

fig.text(0.5, 0.02, description , ha='center', va='center', fontsize=15, wrap=True)
fig.text(0.5, 0.1, student_info, ha='center', va='center', fontsize=16, wrap=True)



# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig("22070680.png", dpi=300)

