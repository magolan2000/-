# -*- coding: utf-8 -*-
import os
import pandas as pd
import akshare as ak
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import mplfinance as mpf
import mplcursors
# 在文件头部设置后端后不再修改
import matplotlib
matplotlib.use('Agg')  # 必须在所有matplotlib导入前设置
from matplotlib import pyplot as plt
#内存管理
import gc
gc.collect()  # 在plt.close()后执行垃圾回收

# ==================== 用户配置部分 ====================
# A股股票代码列表（示例：贵州茅台、宁德时代）
a_stock_symbols = ["600519", "300750","601899"]  # 支持数字或带后缀的代码（如600519.SH）

# 时间范围
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

# 数据保存路径
save_dir = "./stock_data"  # 数据存储主目录
a_stock_dir = os.path.join(save_dir, "A股数据")  # A股数据子目录
plot_dir = os.path.join(save_dir, "可视化图表")  # 可视化图表目录

# 创建目录（如果不存在）
os.makedirs(a_stock_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(save_dir, "stock_data.log")),  # 日志文件
        logging.StreamHandler()  # 控制台输出
    ]
)

# 设置中文字体
try:
    # Windows 系统使用黑体
    #plt.rcParams["font.sans-serif"] = ["SimHei"]
    # macOS 或 Linux 系统使用 Arial Unicode MS 或文泉驿字体
    # plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # macOS
    # plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]  # Linux
    #plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    # 设置全局字体（示例为Windows）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 优先使用雅黑
    plt.rcParams["axes.unicode_minus"] = False
except Exception as e:
    logging.error(f"字体设置失败: {e}")


# ======================================================

def fetch_a_stock_data(symbol, retries=3):
    """通过AKShare获取单只A股数据（后复权）"""
    for attempt in range(retries):
        try:
            # 去掉股票代码的后缀（如 .SH 或 .SZ）
            symbol_clean = symbol.split(".")[0]

            # 获取数据
            df = ak.stock_zh_a_hist(symbol=symbol_clean, period="daily", adjust="hfq")
            if df.empty:
                logging.warning(f"[A股] 数据为空（{symbol_clean}）")
                return None

            # 重命名列
            df.rename(columns={
                "日期": "Date",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low",
                "成交量": "Volume"
            }, inplace=True)

            # 转换日期格式并设置为索引
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            logging.info(f"[A股] 数据获取成功（{symbol_clean}）")
            return df
        except Exception as e:
            if attempt < retries - 1:
                logging.warning(f"[A股] 数据获取失败（{symbol_clean}），第 {attempt + 1} 次重试...")
            else:
                logging.error(f"[A股] 数据获取失败（{symbol_clean}）: {e}")
                return None


def clean_data(df):
    """数据清洗：处理缺失值和异常值"""
    if df is None or df.empty:
        return None

    # 1. 删除完全缺失的行
    df_clean = df.dropna(how="all")

    # 2. 处理零成交量（视为停牌/无效数据）
    df_clean = df_clean[df_clean["Volume"] > 0]

    # 3. 处理价格异常值（如价格<=0或单日涨跌幅超过50%）
    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        df_clean = df_clean[df_clean[col] > 0]  # 删除价格为负或零的异常记录

    # 检查清洗后的数据是否为空
    if df_clean.empty:
        logging.warning("数据清洗后为空，可能全部为异常值")
        return None

    return df_clean


def save_to_csv(df, symbol):
    """保存清洗后的数据到CSV文件"""
    if df is None or df.empty:
        logging.warning(f"无有效数据可保存（{symbol}）")
        return

    # 保存路径
    save_path = os.path.join(a_stock_dir, f"{symbol}.csv")
    df.to_csv(save_path, encoding="utf-8-sig")  # 兼容中文路径
    logging.info(f"数据已保存至：{save_path}")


def plot_stock_data(df, symbol):
    """专业级股票行情走势图"""
    if df is None or df.empty:
        logging.warning(f"无有效数据可绘制（{symbol}）")
        return

    try:
        # ==================== 样式配置 ====================
        plt.style.use('ggplot')
        colors = {
            'background': '#1a1a1a',
            'grid': '#404040',
            'price': '#00ff88',
            'volume_up': '#ff4242',
            'volume_down': '#00b0ff',
            'ma5': '#ffd700',
            'ma10': '#00ffff',
            'ma20': '#ff88ff'
        }

        # ==================== 创建画布 ====================
        fig = plt.figure(figsize=(16, 10), facecolor=colors['background'])
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)

        # 主图区域（价格走势）
        ax_price = fig.add_subplot(gs[0], facecolor=colors['background'])

        # 成交量区域
        ax_volume = fig.add_subplot(gs[1], sharex=ax_price, facecolor=colors['background'])

        # MACD区域
        ax_macd = fig.add_subplot(gs[2], sharex=ax_price, facecolor=colors['background'])

        # RSI区域
        ax_rsi = fig.add_subplot(gs[3], sharex=ax_price, facecolor=colors['background'])

        # ==================== 数据处理 ====================
        # 计算技术指标
        df['Return'] = df['Close'].pct_change()
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

        # ==================== 绘制价格走势 ====================
        # 主价格线
        ax_price.plot(df.index, df['Close'],
                      color=colors['price'],
                      linewidth=1.5,
                      label='收盘价')

        # 均线系统
        ax_price.plot(df.index, df['MA5'],
                      color=colors['ma5'],
                      linewidth=1,
                      linestyle='--',
                      label='5日均线')
        ax_price.plot(df.index, df['MA10'],
                      color=colors['ma10'],
                      linewidth=1,
                      label='10日均线')
        ax_price.plot(df.index, df['MA20'],
                      color=colors['ma20'],
                      linewidth=1.2,
                      label='20日均线')

        # ==================== 绘制成交量 ====================
        # 计算涨跌颜色
        colors_volume = [colors['volume_up'] if x > 0 else colors['volume_down']
                         for x in df['Return']]

        ax_volume.bar(df.index, df['Volume'] / 1e4,
                      color=colors_volume,
                      width=0.8,
                      edgecolor='none')

        # ==================== 绘制MACD ====================
        ax_macd.plot(df.index, df['MACD'],
                     color=colors['ma5'],
                     linewidth=0.8,
                     label='MACD')
        ax_macd.plot(df.index, df['Signal'],
                     color=colors['ma10'],
                     linewidth=0.8,
                     label='Signal')

        # 绘制MACD柱状图
        macd_colors = ['#00ff00' if x >= 0 else '#ff0000' for x in df['Hist']]
        ax_macd.bar(df.index, df['Hist'],
                    color=macd_colors,
                    width=0.8,
                    edgecolor='none')

        # ==================== 绘制RSI ====================
        ax_rsi.plot(df.index, df['RSI'],
                    color=colors['ma20'],
                    linewidth=1,
                    label='RSI')
        ax_rsi.axhline(70, color='#ff0000', linestyle='--', linewidth=0.5)
        ax_rsi.axhline(30, color='#00ff00', linestyle='--', linewidth=0.5)

        # ==================== 全局样式设置 ====================
        for ax in [ax_price, ax_volume, ax_macd, ax_rsi]:
            ax.grid(True, color=colors['grid'], linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')

            # 隐藏底部子图的x轴标签（除最下方）
            if ax != ax_rsi:
                plt.setp(ax.get_xticklabels(), visible=False)

        # 价格轴格式
        ax_price.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'¥{x:.2f}'))

        # 成交量轴格式
        ax_volume.set_ylabel('成交量(万手)', color='white')

        # 日期格式旋转
        plt.xticks(rotation=45, ha='right')

        # ==================== 图例和标题 ====================
        ax_price.legend(loc='upper left',
                        frameon=False,
                        labelcolor='white',
                        prop={'size': 8})

        fig.suptitle(f'{symbol} 专业行情分析',
                     color='white',
                     fontsize=16,
                     y=0.92)

        # ==================== 保存图表 ====================
        plot_path = os.path.join(plot_dir, f"{symbol}_专业行情图.png")
        fig.savefig(plot_path,
                    facecolor=colors['background'],
                    bbox_inches='tight',
                    dpi=300)
        plt.close(fig)
        logging.info(f"专业级图表已保存至：{plot_path}")

    except Exception as e:
        logging.error(f"绘制专业图表失败: {e}")
        plt.close('all')


def process_single_stock(symbol):
    """处理单只股票数据（获取、清洗、保存、可视化）"""
    logging.info(f"正在处理A股：{symbol}")
    raw_data = fetch_a_stock_data(symbol)
    cleaned_data = clean_data(raw_data)
    save_to_csv(cleaned_data, symbol)
    plot_stock_data(cleaned_data, symbol)


def batch_process_a_stocks():
    """批量处理所有A股数据（多线程）"""
    logging.info("\n" + "=" * 30 + " 开始处理A股数据 " + "=" * 30)

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_stock, symbol): symbol for symbol in a_stock_symbols}

        # 等待所有任务完成
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                future.result()  # 获取任务结果（确保异常被捕获）
            except Exception as e:
                logging.error(f"处理股票 {symbol} 时发生错误: {e}")


if __name__ == "__main__":
    batch_process_a_stocks()
    logging.info("\n所有数据处理完成！")