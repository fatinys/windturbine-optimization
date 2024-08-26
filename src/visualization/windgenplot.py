
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


path = os.path.join('notebooks','windgeneration', f"windgen.csv")
gen_df = pd.read_csv(path)

def gen_plot(df):
    df = df.copy()

    # Calculate percentage of annual total for each month
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        df.loc[:, f'{month}_Percentage'] = (df[month] / df['Total']) * 100


    plt.figure(figsize=(14, 8))

    # Plot each year's percentage for each month
    for year in df['Year']:
        sns.lineplot(x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    y=df[df['Year'] == year].loc[:, 'Jan_Percentage':'Dec_Percentage'].values.flatten(),
                    label=year, marker='o')


    plt.xlabel('Month')
    plt.ylabel('Percentage of Annual Total (%)')
    plt.title('Texas Monthly Wind Generation as Percentage of Annual Total Over Years')
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.ylim(3, 15)
    # Show plot
    plt.show()


gen_plot(gen_df)