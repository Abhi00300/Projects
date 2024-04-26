import yfinance as yf
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pytz


def download_and_save(ticker, filename, start_date='2022-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)

    data.to_csv(filename,   index=True)
    print(f"Data for {ticker} saved to {filename} ")

def input_company_info(num_companies):
    company_info = []
    for i in range(num_companies):
        print(f"\nEnter details For Index/Stock {i + 1}")
        company_name = input("Enter the company name: ").strip()
        ticker = input(f"Enter the ticker symbol{company_name}: ").strip()
        filename = input(f"Enter the filename for {company_name}: ").strip()

        company_details = {
            'company_name': company_name,
            'ticker': ticker,
            'filename': filename,
        }
        company_info.append(company_details)
    return company_info

def get_num_companies():
    while True:
        try:
            num = int(input("Enter the number of companies for which you want to enter details: "))
            if num > 0:
                return num
            else:
                print("please enter a valid number greater than 0.")
        except ValueError:
            print("Please enter a valid integer. ")

num_companies = get_num_companies()
company_info = input_company_info(num_companies)
print("\nCompany information entered: ")
print(company_info)

for company in company_info:
    download_and_save(company['ticker'], company['filename'])

def save_to_excel(dataframe, filename):
    dataframe.to_excel(filename, index=True)
    print(f"Data for {filename} saved successfully. ")

df = {}
for company in company_info:
    try:
        df[company['company_name']] = pd.read_csv(company['filename'], parse_dates=['Date'], index_col='Date')
        print(f"\n{company['company_name']} DataFrame Head:\n")
        print(df[company['company_name']].head())
    except FileNotFoundError:
        print(f"Data not found for {company['company_name']}. Skipping...")
        df[company['company_name']] = None



def analyze_data(df):
    choice =  input("Do you want Technical data(Yes/No): ").lower()
    if choice == 'yes':
        for company_name, dataframe in df.items():
            if dataframe is not None:
                dataframe['50_MA'] = dataframe['Close'].rolling(window=50).mean()
                dataframe['200_MA'] = dataframe['Close'].rolling(window=200).mean()

                delta = dataframe['Close'].diff(1)
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                dataframe['RSI'] = 100 - (100 / (1 + rs))

                EMA12 = dataframe['Close'].ewm(span=12, adjust=False).mean()
                EMA26 = dataframe['Close'].ewm(span=26, adjust=False).mean()
                dataframe['MACD'] = EMA12 - EMA26
                dataframe['Signal_Line'] = dataframe['MACD'].ewm(span=9, adjust=False).mean()
                dataframe['MACD_Histogram'] = dataframe['MACD'] - dataframe['Signal_Line']

                high_low = dataframe['High'] - dataframe['Low']
                high_close = np.abs(dataframe['High'] - dataframe['Close'].shift(1))
                low_close = np.abs(dataframe['Low'] - dataframe['Close'].shift(1))
                high_low = high_low.rolling(window=14).mean()
                high_close = high_close.rolling(window=14).mean()
                low_close = low_close.rolling(window=14).mean()
                plus_di = 100 * (high_close / high_low)
                minus_di = 100 * (low_close / high_low)

                dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
                adx = dx.rolling(window=14).mean()
                dataframe['ADX'] = adx
                print(f"\nMetrics for {company_name}\n")
                print(dataframe.tail())

                choice = input("Do you want to Plot Technical data(Yes/No): ").lower()
                if choice == 'yes':

                    plt.figure(figsize=(10, 20))

                    ax1 = plt.subplot(5, 1, 1)
                    dataframe['Close'].plot(label='Close Price', color='blue')
                    dataframe['50_MA'].plot(label='50-day MA', color='red')
                    dataframe['200_MA'].plot(label='200-day MA', color='green')
                    plt.title(f"{company_name} Stock Analysis")
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    ax1v = ax1.twinx()
                    volume_color = np.where(dataframe['Close'].diff() > 0, 'g', 'r')
                    ax1v.bar(dataframe.index, dataframe['Volume'], color=volume_color, alpha=0.8)
                    ax1v.set_ylabel('Volume')
                    ax1v.legend(['Volume'], loc='upper left')

                    plt.subplot(5, 1, 2)
                    dataframe['RSI'].plot(label='RSI', color='g')
                    plt.title('RSI')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.axhline(y=20, color='g', linestyle='--', alpha=0.2)
                    plt.axhline(y=80, color='g', linestyle='--', alpha=0.2)
                    plt.legend()

                    plt.subplot(5, 1, 3)
                    dataframe['MACD'].plot(label='MACD', color='cyan')
                    dataframe['Signal_Line'].plot(label='Signal Line', color='y')
                    plt.title('MACD')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.axhline(y=0, color='b', linestyle='--', alpha=0.2)
                    plt.legend()

                    plt.subplot(5, 1, 4)
                    dataframe['ADX'].plot(label='ADX', color='orange')
                    plt.title('ADX')
                    plt.xlabel('Date')
                    plt.ylabel('Value')
                    plt.legend()

                    plt.tight_layout()
                    plt.show()
                elif choice == 'no':
                    print("Exiting...")

            else:
                print(f"No Data is available for {company_name}. ")
    elif choice == 'no':
        print("Exiting...")
    else:
        print("Invalid choice. Please enter 'yes' or 'no'. ")

analyze_data(df)


def calculate_quarterly_yearly_gains(dataframe):
    dataframe['Daily_Return'] = dataframe['Close'].pct_change()
    dataframe['Quarterly_Return'] = dataframe['Close'].pct_change(63)
    dataframe['Yearly_Return'] = dataframe['Close'].pct_change(252)
    print(dataframe[['Close', 'Daily_Return', 'Quarterly_Return', 'Yearly_Return']].tail())
for company_name, dataframe in df.items():
    if dataframe is not None:
        calculate_quarterly_yearly_gains(dataframe)
        save_filename = f"{company_name}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        save_to_excel(dataframe, save_filename)
    else:
        print(f"No data available for {company_name}")

def fetch_fundamental_info(ticker):
    info = yf.Ticker(ticker).info
    fundamental_data = pd.DataFrame.from_dict(info, orient='index', columns=['Value'])
    fundamental_data.index.name = 'Attribute'
    return fundamental_data

def fetch_intraday_data(ticker, period="1d", interval="1m"):
    data = yf.download(ticker, period=period, interval=interval)
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')  # Localize to UTC if timezone is naive
    data.index = data.index.tz_convert(pytz.timezone("Asia/Kolkata")).tz_localize(None)
    return data

def fetch_key_statistics(ticker):
    stats = yf.Ticker(ticker).info
    key_stats = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
    key_stats.index.name = 'Attribute'
    return key_stats

def save_data_to_excel(dataframe, fundamental_data, intraday_data, key_stats, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        dataframe.to_excel(writer, sheet_name='Stock_Data', index=True)
        fundamental_data.to_excel(writer, sheet_name=f"{fundamental_data['company_name'].iloc[0]}_Fundamentals",
                                  index=True, header=True)
        intraday_data.to_excel(writer, sheet_name='Intraday_Data', index=True)
        key_stats.to_excel(writer, sheet_name='Key_Statistics', index=True, header=True)

ticker = company_info[0]['ticker']
fundamental_data = fetch_fundamental_info(ticker)
fundamental_data['company_name'] = company_info[0]['company_name']

for company_info in company_info:
    ticker = company_info['ticker']
    company_name = company_info['company_name']
    filename = company_info['filename']

    if df[company_name] is not None:
        intraday_data = fetch_intraday_data(ticker)
        key_stats = fetch_key_statistics(ticker)
        fundamental_data = fetch_fundamental_info(ticker)
        fundamental_data['company_name'] = company_name

        save_filename = f"{company_name}_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
        save_data_to_excel(df[company_name], fundamental_data, intraday_data, key_stats, save_filename)

        print(f"Data and fundamental information for {company_name} saved successfully.")
    else:
        print(f"No data available for {company_name}")
def plot_linear_regression(dataframe, company_name):
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna(subset=['Volume', 'Close'])
    if len(dataframe) < 2:
        print(f"Error: Not enough data points for linear regression for {company_name}.")
        return
    x = dataframe['Volume'].values.reshape(-1, 1)
    y = dataframe['Close'].values
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_pred, color='red', linewidth=2, label='Linear Regression Line')
    plt.title(f"{company_name} Linear Regression: Close Price vs Volume")
    plt.xlabel('Volume')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()


for company_name, dataframe in df.items():
    if dataframe is not None:
        plot_linear_regression(dataframe, company_name)
    else:
        print(f"No data available for {company_name}")


choice = input("Do you want Correlation matrix(Yes/No): ").lower()
if choice == 'yes':
    selected_colums = ["Close", "50_MA", "200_MA",  "RSI", "MACD", "ADX"]
    if len(df) < 2:
        print("Error: Please add at least two companies to calculate the correlation matrix.")
    else:
        selected_df = {key: df[key][selected_colums] for key in df if df[key] is not None}
        concatented_df = pd.concat(selected_df.values(), axis=1, keys=selected_df.keys())
        correlation_matrix = concatented_df.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
        plt.title(f"Correlation matrix for {company_name} and {company_name}")
        plt.show()
elif choice == 'no':
    print("Exiting...")
