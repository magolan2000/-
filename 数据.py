# -*- coding: utf-8 -*-
import os
import pandas as pd
import akshare as ak
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # macOS 或 Linux 系统使用 Arial Unicode MS 或文泉驿字体
    # plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # macOS
    # plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]  # Linux
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
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
    """绘制股票价格走势图"""
    if df is None or df.empty:
        logging.warning(f"无有效数据可绘制（{symbol}）")
        return
    if "Close" not in df.columns or df["Close"].dropna().empty:
        logging.warning(f"股票 {symbol} 缺少收盘价数据，无法绘图")
        return

    try:
        font_path = None
        for font in fm.findSystemFonts():
            if "SimHei" in font or "Hei" in font:
                font_path = font
                break
        if font_path:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
        else:
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]

        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        logging.error(f"字体加载失败: {e}")
        return

    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Close"], color="red", linewidth=2, label="收盘价")  # 确保 label 设置正确
    plt.title(f"股票 {symbol} 价格走势图", fontsize=18, fontproperties=font_prop)
    plt.xlabel("日期", fontsize=14, fontproperties=font_prop)
    plt.ylabel("股价 (元)", fontsize=14, fontproperties=font_prop)
    plt.legend(prop=font_prop, fontsize=12)  # 确保 legend() 在 plot 之后调用

    plot_path = os.path.join(plot_dir, f"{symbol}_价格走势图.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    logging.info(f"图表已保存至：{plot_path}")




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