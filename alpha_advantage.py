import requests

def get_prices():
    api_key = 'undefined'
    symbol = 'SPY'  # SPDR S&P 500 ETF
    function = 'TIME_SERIES_DAILY'  # 일별 주가 데이터
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'

    response = requests.get(url)
    data = response.json()

    # 데이터 처리
    time_series = data.get('Time Series (Daily)', {})
    for date, metrics in time_series.items():
        open_price = metrics['1. open']
        high_price = metrics['2. high']
        low_price = metrics['3. low']
        close_price = metrics['4. close']
        volume = metrics['5. volume']
        print(f"Date: {date}, Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}")

import yfinance as yf
import mplfinance as mpf
import pandas as pd

def plot_candlestick(symbol='SPY', interval='1d', start_date=None, end_date=None):
    """
    SPY500의 주가 데이터를 수집하고, 지정된 구간과 간격으로 캔들스틱 차트를 시각화하는 함수입니다.

    Parameters:
    - symbol: 주식 심볼, 기본값은 'SPY' (SPY500 ETF).
    - interval: 데이터 간격 ('1d'는 일별, '1wk'는 주간).
    - start_date: 시작 날짜 (YYYY-MM-DD 형식의 문자열).
    - end_date: 종료 날짜 (YYYY-MM-DD 형식의 문자열).
    """
    # yfinance를 통해 주가 데이터 수집
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1y', interval=interval)
    
    # 시작 및 종료 날짜에 따른 데이터 필터링
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    # 캔들스틱 차트 시각화
    mpf.plot(data, type='candle', style='charles', title=f'{symbol} {interval.upper()} Candlestick Chart',
             ylabel='Price', volume=True)


import yfinance as yf
import matplotlib.pyplot as plt

def plot_bollinger_bands(symbol='SPY', start_date=None, end_date=None, window=20, num_std=2):
    """
    SPY500의 주가 데이터를 수집하고, 볼린저 밴드와 함께 시각화하는 함수입니다.

    Parameters:
    - symbol: 주식 심볼, 기본값은 'SPY' (SPY500 ETF).
    - start_date: 시작 날짜 (YYYY-MM-DD 형식의 문자열).
    - end_date: 종료 날짜 (YYYY-MM-DD 형식의 문자열).
    - window: 이동 평균 계산 기간, 기본값은 20일.
    - num_std: 표준 편차의 배수, 기본값은 2.
    """
    # yfinance를 통해 주가 데이터 수집
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='1y')
    
    # 시작 및 종료 날짜에 따른 데이터 필터링
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    # 볼린저 밴드 계산
    data['MA20'] = data['Close'].rolling(window=window).mean()  # 20일 이동 평균
    data['Upper'] = data['MA20'] + (data['Close'].rolling(window=window).std() * num_std)  # 상단 밴드
    data['Lower'] = data['MA20'] - (data['Close'].rolling(window=window).std() * num_std)  # 하단 밴드
    
    # 시각화
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['Close'], label=f'{symbol} Close Price', color='blue')
    plt.plot(data.index, data['MA20'], label='20-Day MA', color='green')
    plt.plot(data.index, data['Upper'], label='Upper Band', color='red')
    plt.plot(data.index, data['Lower'], label='Lower Band', color='red')
    
    plt.fill_between(data.index, data['Upper'], data['Lower'], color='lightgray', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{symbol} Bollinger Bands')
    plt.legend()
    plt.grid(True)
    plt.show()


from datetime import datetime, timedelta
import numpy as np


def plot_candlestick_with_bollinger(
    symbol='SPY', 
    start_date=None, 
    end_date=None, 
    window=20, 
    num_std=2, 
    use_trendline=False, 
    use_bollinger=True
):
    """
    주식 심볼의 캔들스틱 차트에 볼린저 밴드 및 추세선을 추가하여 시각화하는 함수입니다.

    Parameters:
    - symbol: 주식 심볼, 기본값은 'SPY' (SPY500 ETF)
    - start_date: 시작 날짜 (YYYY-MM-DD 형식의 문자열)
    - end_date: 종료 날짜 (YYYY-MM-DD 형식의 문자열)
    - window: 볼린저 밴드의 이동 평균 기간, 기본값은 20일
    - num_std: 볼린저 밴드의 표준 편차 배수, 기본값은 2
    - use_trendline: 추세선 추가 여부, 기본값은 False
    - use_bollinger: 볼린저 밴드 추가 여부, 기본값은 True
    """
    # 시작 날짜와 종료 날짜 처리
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        # 기본값으로 6개월 전 데이터부터 시작
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')

    # window 기간만큼 이전 데이터를 추가로 가져와서 볼린저 밴드 계산을 위한 충분한 데이터 확보
    calc_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=window + 5)).strftime('%Y-%m-%d')
    
    # 데이터 가져오기
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=calc_start_date, end=end_date)
    
    # 볼린저 밴드 계산
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['Upper'] = data['MA20'] + (data['Close'].rolling(window=window).std() * num_std)
    data['Lower'] = data['MA20'] - (data['Close'].rolling(window=window).std() * num_std)

    # 추세선 계산 (Close 데이터의 선형 회귀)
    if use_trendline:
        x = range(len(data))
        y = data['Close'].values
        fit = np.polyfit(x, y, 1)
        data['Trendline'] = fit[0] * x + fit[1]

    # 실제 표시할 구간만 필터링
    plot_data = data[data.index >= start_date].copy()

    # 추가 플롯 설정
    add_plots = []
    if use_bollinger:
        add_plots += [
            mpf.make_addplot(plot_data['MA20'], color='green', width=1.5),
            mpf.make_addplot(plot_data['Upper'], color='orange', width=1.2),
            mpf.make_addplot(plot_data['Lower'], color='orange', width=1.2)
        ]
    if use_trendline:
        add_plots.append(mpf.make_addplot(plot_data['Trendline'], color='blue', linestyle='-', width=1.5))
    
    # 차트 스타일 설정
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up='red', down='blue',
            edge='inherit',
            wick='inherit',
            volume='in'
        ),
        gridstyle=':', 
        y_on_right=True
    )
    
    # 캔들스틱 차트와 볼린저 밴드 시각화
    fig, axes = mpf.plot(
        plot_data, 
        type='candle', 
        style=style,
        title=f'{symbol} Candlestick with Bollinger Bands\n{start_date} to {end_date}',
        ylabel='Price',
        volume=True, 
        addplot=add_plots,
        returnfig=True
    )
    
    # x축 레이블 조정
    axes[0].set_xlabel('Date')
    plt.show()

    return fig, axes

def plot_candlestick_with_bollinger_minimal_with_volume_fixed(
    symbol='SPY', 
    start_date=None, 
    end_date=None, 
    window=20, 
    num_std=2, 
    use_trendline=False, 
    use_bollinger=True,
    show=True
):
    """
    주식 심볼의 캔들스틱 차트와 거래량, 볼린저 밴드를 최소한의 형태로 시각화하며,
    y축을 오른쪽으로 이동시킨 함수.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=180)).strftime('%Y-%m-%d')

    calc_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=window + 5)).strftime('%Y-%m-%d')
    
    # 데이터 가져오기
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=calc_start_date, end=end_date)
    
    # 볼린저 밴드 계산
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['Upper'] = data['MA20'] + (data['Close'].rolling(window=window).std() * num_std)
    data['Lower'] = data['MA20'] - (data['Close'].rolling(window=window).std() * num_std)

    if use_trendline:
        x = range(len(data))
        y = data['Close'].values
        fit = np.polyfit(x, y, 1)
        data['Trendline'] = fit[0] * x + fit[1]

    plot_data = data[data.index >= start_date].copy()

    # 추가 플롯 설정
    add_plots = []
    if use_bollinger:
        add_plots += [
            mpf.make_addplot(plot_data['MA20'], color='green', width=1.5),
            mpf.make_addplot(plot_data['Upper'], color='orange', width=1.2),
            mpf.make_addplot(plot_data['Lower'], color='orange', width=1.2)
        ]
    if use_trendline:
        add_plots.append(mpf.make_addplot(plot_data['Trendline'], color='blue', linestyle='-', width=1.5))
    
    # 차트 스타일 설정 (y축 오른쪽으로 이동)
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up='red', down='blue',
            edge='inherit',
            wick='inherit',
            volume='in'
        ),
        gridstyle=':', 
        y_on_right=True  # y축을 오른쪽으로 설정
    )
    
    # 캔들스틱 차트와 거래량 포함 시각화
    fig, axes = mpf.plot(
        plot_data, 
        type='candle', 
        style=style,
        title='',  # 제목 제거
        ylabel='',  # y축 레이블 제거
        volume=True,  # 거래량 포함
        addplot=add_plots,
        returnfig=True
    )
    
    # x축, y축, 제목 등 제거
    for ax in axes:
        ax.set_xlabel('')  # x축 레이블 제거
        ax.set_ylabel('')  # y축 레이블 제거
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # x축 눈금 제거
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # 왼쪽 y축 제거
        ax.tick_params(axis='y', which='both', right=True, labelright=False)  # 오른쪽 y축 추가

    
    if show:
        plt.tight_layout()
        plt.show()

    return fig, axes


if __name__ =='__main__':
    # 예제 사용
    # plot_candlestick(symbol='SPY', interval='1d', start_date='2024-11-01', end_date='2024-11-13')

    # 예제 사용
    # plot_bollinger_bands(symbol='SPY', start_date='2023-01-01', end_date='2023-12-31')

    # 예제 사용
    # plot_candlestick_with_bollinger(symbol='IONQ', start_date='2024-10-01', end_date='2024-11-17',use_trendline=False,use_bollinger=False)

    fig, axes = plot_candlestick_with_bollinger_minimal_with_volume_fixed(
        symbol='IONQ',
        start_date='2024-10-01',
        end_date='2024-11-17',
        use_trendline=True,
        use_bollinger=True
    )
