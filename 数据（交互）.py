import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots  # 添加这一行
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 初始化 Dash 应用
app = dash.Dash(__name__, title="专业股票分析系统")
server = app.server

# ==================== 布局设计 ====================
app.layout = html.Div([
    html.Div([
        html.H1("专业级股票分析系统", style={'color': 'white'}),
        html.Div([
            dcc.Input(
                id='stock-input',
                type='text',
                value='600519',
                placeholder='输入股票代码',
                style={'width': '150px', 'margin-right': '10px'}
            ),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=datetime(2010, 1, 1),
                max_date_allowed=datetime.today(),
                start_date=datetime.today() - timedelta(days=365),
                end_date=datetime.today(),
                display_format='YYYY-MM-DD'
            ),
            dcc.Dropdown(
                id='indicator-selector',
                options=[
                    {'label': 'MACD', 'value': 'MACD'},
                    {'label': 'RSI', 'value': 'RSI'},
                    {'label': '布林线', 'value': 'BOLL'},
                ],
                value=['MACD'],
                multi=True,
                style={'width': '300px', 'margin-left': '10px'}
            )
        ], style={'padding': '20px', 'backgroundColor': '#1a1a1a'})
    ]),

    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dcc.Graph(id='main-chart', style={'height': '600px'}),
            dcc.Interval(id='interval-component', interval=60 * 1000, n_intervals=0)
        ]
    )
], style={'backgroundColor': '#1a1a1a', 'height': '100vh'})


# ==================== 数据处理函数 ====================
def get_stock_data(symbol, start_date, end_date):
    """获取股票数据并计算技术指标"""
    try:
        symbol_clean = symbol.split(".")[0]
        df = ak.stock_zh_a_hist(
            symbol=symbol_clean,
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust="hfq"
        )

        if df.empty:
            return pd.DataFrame()

        # 重命名列
        df = df.rename(columns={
            "日期": "Date",
            "开盘": "Open",
            "最高": "High",
            "最低": "Low",
            "收盘": "Close",
            "成交量": "Volume"
        })

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 计算技术指标
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()

        # 计算MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # 计算RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        return df.dropna()

    except Exception as e:
        logging.error(f"数据获取失败: {str(e)}")
        return pd.DataFrame()


# ==================== 回调函数 ====================
@app.callback(
    Output('main-chart', 'figure'),
    [Input('stock-input', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('indicator-selector', 'value')]
)
def update_chart(symbol, start_date, end_date, indicators):
    # 获取数据
    df = get_stock_data(symbol,
                        datetime.fromisoformat(start_date[:10]),
                        datetime.fromisoformat(end_date[:10]))

    if df.empty:
        return go.Figure()

    # 创建子图
    fig = go.Figure()
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # 主图：K线图 + 均线
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='K线',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ), row=1, col=1)

    # 添加均线
    for ma in ['MA5', 'MA10', 'MA20']:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[ma],
            name=ma,
            line=dict(width=1),
            visible=True
        ), row=1, col=1)

    # 成交量
    colors = ['#00b0ff' if x >= 0 else '#ff4444' for x in df['Close'].diff().fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='成交量',
        marker_color=colors
    ), row=2, col=1)

    # MACD
    if 'MACD' in indicators:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='#00ffff', width=1)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Signal'],
            name='Signal',
            line=dict(color='#ff88ff', width=1)
        ), row=3, col=1)

        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Hist'],
            name='Hist',
            marker_color=['#00ff00' if x >= 0 else '#ff0000' for x in df['Hist']]
        ), row=3, col=1)

    # RSI
    if 'RSI' in indicators:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='#ffd700', width=1)
        ), row=4, col=1)

        fig.add_shape(
            type="line",
            x0=df.index[0], y0=70,
            x1=df.index[-1], y1=70,
            line=dict(color="#ff0000", width=1, dash="dot"),
            row=4, col=1
        )

        fig.add_shape(
            type="line",
            x0=df.index[0], y0=30,
            x1=df.index[-1], y1=30,
            line=dict(color="#00ff00", width=1, dash="dot"),
            row=4, col=1
        )

    # 布局设置
    fig.update_layout(
        template="plotly_dark",
        margin=dict(r=10, t=40, b=40, l=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=1000
    )

    # 轴标签设置
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)

    return fig


# ==================== 启动应用 ====================
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)