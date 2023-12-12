# Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis


def read_and_transpose_dataframe(file_path):
    """
        Read a CSV file into a Pandas DataFrame and transpose it.

        Parameters:
        - file_path (str): The path to the CSV file.

        Returns:
        - original_df : Time as column dataframe.
        - transposed_df : country as column dataframe.
    """

    original_df = pd.read_csv(file_path)
    transposed_df = original_df.copy()
    transposed_df[['Time' , 'Country Name']] = \
        transposed_df[['Country Name' , 'Time']]
    transposed_df = transposed_df.rename(columns =
                                         {'Time': 'Country Name' , 'Country Name': 'Time'})
    return original_df , transposed_df


def kurtosisRepresentationPlot(data):
    """
        Create a histogram representation of the given data
         along with its kurtosis value.

        Parameters:
        - data (array-like): The input data for which the
         histogram and kurtosis will be plotted.

        Returns:
        None
    """
    sns.histplot(data , kde = True)
    plt.title('Histogram with Kurtosis value 10.8128' , fontsize = 17)
    plt.xlabel('Oil rents (% of GDP)')
    plt.ylabel('Frequency')
    plt.show()


def BarGraph(data):
    """
        Plot a bar graph comparing 'Forest Area' and 'Forest Rents' for
        different years in Australia.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing the necessary columns,
        including 'Time', 'Forest area (% of land area)',
          and 'Forest rents (% of GDP) '.

        Returns:
        None
    """

    plt.figure(figsize=(10 , 6))
    bar_width = 0.35

    plt.bar(data['Time'] - bar_width / 2 , data['Forest area (% of land area) '] ,
            width = bar_width , label = 'Forest Area')
    plt.bar(data['Time'] + bar_width / 2 , data['Forest rents (% of GDP) [NY.GDP.FRST.RT.ZS]'] ,
            width = bar_width , label = 'Forest Rents')

    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.title('Forest Area vs Forest Rents for Different Years in Australia' , fontsize=17)
    plt.xticks(data['Time'] + bar_width / 2 , data['Time'])
    plt.legend()

    plt.show()


def heatMapPlot(correlation_matrix):
    """
        Create a heatmap for the given correlation matrix.

        Parameters:
        - correlation_matrix (pd.DataFrame): The correlation matrix to
        be visualized.

        Returns:
        None
    """
    plt.imshow(correlation_matrix , cmap = 'coolwarm' , vmin = -1 , vmax = 1)

    # Add colorbar
    plt.colorbar()

    # Annotate each cell with the correlation value
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            plt.text(j , i , f'{correlation_matrix.iloc[i , j]:.2f}' ,
                     ha = 'center' , va = 'center' , color = 'black')

    # Customize the plot
    plt.title('Correlation Matrix of agriculture vs forest vs arable land area',
              fontsize = 17)
    plt.xticks(np.arange(correlation_matrix.shape[1]) ,
               labels = correlation_matrix.columns)
    plt.yticks(np.arange(correlation_matrix.shape[0]) ,
               labels = correlation_matrix.index)
    plt.xlabel('indicators')
    plt.ylabel('indicators')

    # Display the plot
    plt.show()


def BarGraphRents(data):
    """
        Plot a bar graph representing coal, mineral, and natural gas rents
        in the United Kingdom over the years.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing the necessary columns,
        including 'Country Name', 'Time',
          'Coal rents (% of GDP) [NY.GDP.COAL.RT.ZS]', 'Mineral rents (% of GDP)
           [NY.GDP.MINR.RT.ZS]', and
          'Natural gas rents (% of GDP) [NY.GDP.NGAS.RT.ZS]'.

        Returns:
        None
    """
    # Filter data for India
    india_df = data[data['Country Name'] == 'United Kingdom']
    india_df = india_df[(india_df['Time'] >= 2013) & (india_df['Time'] <= 2022)]
    # Plotting the bar graph
    plt.figure(figsize = (10 , 6))
    bar_width = 0.25

    plt.bar(india_df['Time'] , india_df['Coal rents (% of GDP) [NY.GDP.COAL.RT.ZS]'] ,
            width = bar_width ,
            label = 'Coal Rents' , align = 'center')
    plt.bar(india_df['Time'] + bar_width , india_df['Mineral rents (% of GDP) [NY.GDP.MINR.RT.ZS]'] ,
            width = bar_width ,
            label = 'Mineral Rents' , align = 'center')
    plt.bar(india_df['Time'] + 2 * bar_width ,
            india_df['Natural gas rents (% of GDP) [NY.GDP.NGAS.RT.ZS]'] ,
            width = bar_width , label = 'Natural Gas Rents' , align = 'center')

    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.title('Coal , Mineral , and Natural Gas Rents in UNITED KINGDOM Over Years ' ,
              fontsize=17)
    plt.xticks(india_df['Time'] + bar_width , india_df['Time'])
    plt.legend()

    plt.show()


def pieGraph(data):
    # Filter data for Japan
    japan_df = data[data['Country Name'] == 'Japan'].dropna()
    # Plotting the pie graph
    plt.figure(figsize = (8 , 8))
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = japan_df['Time']
    sizes = japan_df['Agricultural land (% of land area) ']
    plt.pie(sizes , labels = labels , autopct = '%1.1f%%' , startangle = 140)
    plt.title('Distribution of Agricultural Land in Japan Over Years' , fontsize = 17)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# Load and read the dataset
columnTime , columnCountry = read_and_transpose_dataframe('Dataset.csv')

# Statistical Tools
numeric_column = pd.to_numeric(columnCountry['Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]'] ,
                               errors = 'coerce')
methoddescribes = numeric_column.describe()
print('Summary Statistics')
print(methoddescribes)

#kurtosis
columnCountry['Time'] = pd.to_numeric(columnCountry['Time'] , errors = 'coerce')
datakurtosis = pd.to_numeric(columnCountry['Oil rents (% of GDP) [NY.GDP.PETR.RT.ZS]'] ,
                             errors = 'coerce').dropna()
kurtosis =  datakurtosis.kurtosis()
print("The kurtosis value for Oil rents (% of GDP) is " , kurtosis)

#Kurtosis plot
kurtosisRepresentationPlot(datakurtosis)

# Filter data for Australia
australia_df = columnCountry[columnCountry['Country Name'] == 'Australia']
australia_df = australia_df.copy()
australia_df['Time'] = pd.to_numeric(australia_df['Time'] , errors='coerce')
australia_df = australia_df[(australia_df['Time'] >= 2013) &
                            (australia_df['Time'] <= 2022)]

#Bar graph
BarGraph(australia_df)


#correlation matrix

columnCountry['Agricultural land (% of land area) '] = \
    pd.to_numeric(columnCountry['Agricultural land (% of land area) '] , errors = 'coerce')
columnCountry['Arable land (% of land area) '] = \
    pd.to_numeric(columnCountry['Arable land (% of land area) '] , errors = 'coerce')
columnCountry['Forest area (% of land area) '] = \
    pd.to_numeric(columnCountry['Forest area (% of land area) '] , errors = 'coerce')

correlatioIndicators = ['Agricultural land (% of land area) ' , 'Arable land (% of land area) ' ,
                        'Forest area (% of land area) ']

correlationdata = columnCountry[correlatioIndicators].corr()

#Heat map
heatMapPlot(correlationdata)

#Bar graph
BarGraphRents(columnCountry)

#pie graph
pieGraph(columnCountry)

