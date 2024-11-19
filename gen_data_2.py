import numpy as np
import pandas as pd
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_candlestick_with_sampling(
    symbol='SPY', 
    start_date=None, 
    end_date=None, 
    window=20, 
    num_std=2, 
    use_trendline=False,
    use_bollinger=True,
    sampling_mode='random',  # 'random' or 'last'
    num_samples=1,          # random 모드에서 사용될 샘플 개수
    is_target=False,        # target 이미지인지 여부
    show=True,
    figsize=(8, 6)         # 이미지 크기 조정을 위한 새로운 매개변수
):
    """
    주식 심볼의 캔들스틱 차트와 거래량을 시각화하며, sampling_mode에 따라 데이터를 비우는 함수.
    
    Parameters:
    - sampling_mode: 'random' (랜덤 위치 n개) 또는 'last' (마지막 인덱스)
    - num_samples: random 모드에서 비운 데이터 포인트 개수
    - is_target: True면 모든 데이터 표시, False면 일부 데이터 비움
    - figsize: 이미지 크기를 조정하는 툴 (width, height)
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
    
    # input 이미지 생성 시 데이터 비우기
    if not is_target:
        if sampling_mode == 'random':
            # 랜덤하게 n개의 인덱스 선택
            total_points = len(plot_data)
            indices_to_remove = random.sample(range(total_points), min(num_samples, total_points))
            
            # 선택된 인덱스의 데이터 비우기
            for idx in indices_to_remove:
                plot_data.iloc[idx, :] = np.nan
                
        elif sampling_mode == 'last':
            # 마지막 인덱스의 데이터 비우기
            plot_data.iloc[-1, :] = np.nan

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
    
    # 캔들스틱 차트와 거래량 포함 시각화 (figsize 적용)
    fig, axes = mpf.plot(
        plot_data, 
        type='candle', 
        style=style,
        title='',
        ylabel='',
        volume=True,
        addplot=add_plots,
        returnfig=True,
        figsize=figsize  # 이미지 크기 설정
    )
    
    # x축, y축, 제목 등 제거
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(axis='y', which='both', right=True, labelright=False)
    
    if show:
        plt.tight_layout()
        plt.show()

    return fig, axes, data

def driver_save_candlestick_charts(
    dataset_symbols,
    samples_per_symbol=400,
    base_folder='data',
    use_trendline=False,
    use_bollinger=False,
    sampling_mode='random',
    num_samples=1,
    days_per_sample=14,  # 각 샘플의 데이터 기간
    image_size=256      # 이미지 크기를 지정하는 새로운 매개변수
):
    """
    주식 심볼 그룹별로 대량의 차트 데이터를 생성하여 저장하는 함수.
    
    Parameters:
    - dataset_symbols: 심볼 그룹을 포함하는 딕셔너리
    - samples_per_symbol: 각 심볼당 생성할 샘플 수
    - days_per_sample: 각 샘플의 데이터 기간 (기본값: 14일)
    - image_size: 생성할 이미지의 크기 (정사각형, 픽셀 단위)
    """
    # figsize 계산 (dpi=100 기준으로 계산)
    figsize = (image_size/100, image_size/100)
    
    # 기본 폴더 생성
    input_base = os.path.join(base_folder, 'inputs')
    target_base = os.path.join(base_folder, 'targets')
    
    # 모든 심볼 목록 생성
    all_symbols = []
    for group_symbols in dataset_symbols.values():
        all_symbols.extend(group_symbols)
    
    # 모드별 폴더 및 심볼별 하위 폴더 생성
    modes = ['random', 'last']
    for mode in modes:
        for symbol in all_symbols:
            os.makedirs(os.path.join(input_base, mode, symbol), exist_ok=True)
            os.makedirs(os.path.join(target_base, mode, symbol), exist_ok=True)
    
    # 오늘 날짜 기준으로 데이터 생성
    today = datetime.now()
    
    # 진행 상황 추적
    total_symbols = len(all_symbols)
    total_samples = total_symbols * samples_per_symbol
    current_sample = 0

    # CSV 데이터 저장을 위한 리스트 초기화
    csv_data = []

    for group_name, symbols in dataset_symbols.items():
        print(f"\nProcessing {group_name} group...")
        
        for symbol in symbols:
            print(f"\nGenerating {samples_per_symbol} samples for {symbol}...")
            
            for sample_idx in range(samples_per_symbol):
                random_start_date = today - timedelta(days=random.randint(days_per_sample, 600))
                start_date = random_start_date.strftime('%Y-%m-%d')
                end_date = (random_start_date + timedelta(days=days_per_sample)).strftime('%Y-%m-%d')
                        
                try:
                    filename = f'{symbol}_{start_date}_to_{end_date}_sample{sample_idx+1:04d}.png'
                    
                    # Random mode
                    fig, axes, data = plot_candlestick_with_sampling(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        use_trendline=use_trendline,
                        use_bollinger=use_bollinger,
                        sampling_mode='random',
                        num_samples=num_samples,
                        is_target=False,
                        show=False,
                        figsize=figsize
                    )
                    
                    input_path = os.path.join(input_base, 'random', symbol, filename)
                    fig.savefig(input_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Random mode target
                    fig, axes, _ = plot_candlestick_with_sampling(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        use_trendline=use_trendline,
                        use_bollinger=use_bollinger,
                        is_target=True,
                        show=False,
                        figsize=figsize
                    )
                    
                    target_path = os.path.join(target_base, 'random', symbol, filename)
                    fig.savefig(target_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Random mode up_or_down and change_rate
                    up_or_down = []
                    change_rate = []
                    for idx in range(num_samples):
                        sample_index = -1 - idx
                        if sample_index - 1 >= -len(data):
                            close_today = data.iloc[sample_index]['Close']
                            close_yesterday = data.iloc[sample_index - 1]['Close']
                            up_or_down.append(1 if close_today > close_yesterday else 0)
                            change_rate.append((close_today - close_yesterday) / close_yesterday)
                    
                    # Add random mode data to csv_data
                    csv_data.append([symbol, input_path, target_path, 'random', ';'.join(map(str, up_or_down)), ';'.join(map(str, change_rate))])
                    
                    # Last mode
                    fig, axes, data = plot_candlestick_with_sampling(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        use_trendline=use_trendline,
                        use_bollinger=use_bollinger,
                        sampling_mode='last',
                        num_samples=1,
                        is_target=False,
                        show=False,
                        figsize=figsize
                    )
                                        
                    input_path = os.path.join(input_base, 'last', symbol, filename)
                    fig.savefig(input_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Last mode target
                    fig, axes, _ = plot_candlestick_with_sampling(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        use_trendline=use_trendline,
                        use_bollinger=use_bollinger,
                        sampling_mode='last',
                        num_samples=1,
                        is_target=True,
                        show=False,
                        figsize=figsize
                    )
                    target_path = os.path.join(target_base, 'last', symbol, filename)
                    fig.savefig(target_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    # Last mode up_or_down and change_rate
                    up_or_down = []
                    change_rate = []
                    if len(data) >= 2:
                        close_today = data.iloc[-1]['Close']
                        close_yesterday = data.iloc[-2]['Close']
                        up_or_down.append(1 if close_today > close_yesterday else 0)
                        change_rate.append((close_today - close_yesterday) / close_yesterday)
                    
                    # Add last mode data to csv_data
                    csv_data.append([symbol, input_path, target_path, 'last', ';'.join(map(str, up_or_down)), ';'.join(map(str, change_rate))])
                    
                    current_sample += 1
                    if sample_idx % 10 == 0:
                        progress = (current_sample / total_samples) * 100
                        print(f"Progress: {progress:.1f}% - Generated sample {sample_idx+1}/{samples_per_symbol} for {symbol}")
                    
                except Exception as e:
                    print(f"Error generating chart for {symbol} (sample {sample_idx+1}): {e}")
                    continue
            
            print(f"Completed generating {samples_per_symbol} samples for {symbol}")
    
    # Save csv_data to CSV
    csv_df = pd.DataFrame(csv_data, columns=['symbol', 'input_path', 'target_path', 'mode', 'up_or_down', 'change_rate'])
    csv_df.to_csv(os.path.join(base_folder, 'candlestick_data.csv'), index=False)
    
    print("\nData generation completed!")
    return csv_df

def get_dataset_info(base_folder='data'):
    """
    데이터셋의 구조와 각 폴더별 이미지 개수를 출력하는 함수
    
    Returns:
    - dict: 데이터셋 정보를 담은 딕셞너리
    """
    dataset_info = {
        'inputs': {'random': {}, 'last': {}},
        'targets': {'random': {}, 'last': {}}
    }
    
    # inputs와 targets 순회
    for data_type in ['inputs', 'targets']:
        base_path = os.path.join(base_folder, data_type)
        
        # random과 last 모드 순회
        for mode in ['random', 'last']:
            mode_path = os.path.join(base_path, mode)
            
            # 각 심볼 폴더 순회
            if os.path.exists(mode_path):
                for symbol in os.listdir(mode_path):
                    symbol_path = os.path.join(mode_path, symbol)
                    if os.path.isdir(symbol_path):
                        num_images = len([f for f in os.listdir(symbol_path) if f.endswith('.png')])
                        dataset_info[data_type][mode][symbol] = num_images
    
    # 정보 출력
    print("\nDataset Information:")
    print("===================")
    
    for data_type in ['inputs', 'targets']:
        print(f"\n{data_type.upper()}:")
        for mode in ['random', 'last']:
            print(f"\n  {mode.upper()} mode:")
            for symbol, count in dataset_info[data_type][mode].items():
                print(f"    - {symbol}: {count} images")
    
    return dataset_info

if __name__ == '__main__':
    dataset_symbols = {
        'big_tech_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ORCL', 'IBM', 'AMD'],
        'my_favors': ['SOXL', 'IONQ', 'AVGO', 'CRM', 'PLTR', 'DPST', 'SHOC', 'MU', 'QQQ', 'SPY', 'QCOM'],
        'drops': ['SMCI', 'INTC'],
        'inverses': ['SOXS', 'NVD']
    }

    info = driver_save_candlestick_charts(
        dataset_symbols=dataset_symbols,
        samples_per_symbol=200,  # 각 심볼당 2개 샘플
        use_bollinger=False,
        num_samples=2,  # random 모드에서 비울 데이터 포인트 개수
        image_size=128  # 256x256 픽셀 이미지 생성
    )

    get_dataset_info()