#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
正交试验设计 - 行李箱操作人机工程仿真

使用L50标准正交表生成实验设计方案。
在保证实验覆盖面的同时确保了设计的正交性。

此脚本仅生成设计方案CSV文件，模拟计算由 综合计算.py 完成。
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path

# 获取当前脚本文件所在的目录
SCRIPT_DIR = Path(__file__).resolve().parent

# 定义基于脚本位置的输出目录
# 这是主要的修改点，确保 RESULTS_DIR 总是相对于脚本位置
RESULTS_DIR = SCRIPT_DIR / "结果" / "正交设计"
# RESULTS_DIR.mkdir(parents=True, exist_ok=True) # 这行可以在main函数或save函数中调用，避免脚本加载时就创建目录

def generate_timestamp():
    """生成时间戳，格式为：yyyy.mm.dd_HH.MM.SS"""
    now = datetime.now()
    return now.strftime("%Y.%m.%d_%H.%M.%S")

# L50 (2^1 * 5^11) array data from University of York
# https://www.york.ac.uk/depts/maths/tables/l50.htm
# We use X1 (2-level) for gender and X2-X5 (5-level) for percentile, surface, posture, weight
l50_data_columns_X1_X2_X3_X4_X5 = [
    [1, 1, 1, 1, 1],
    [1, 1, 2, 2, 2],
    [1, 1, 3, 3, 3],
    [1, 1, 4, 4, 4],
    [1, 1, 5, 5, 5],
    [1, 2, 1, 2, 3],
    [1, 2, 2, 3, 4],
    [1, 2, 3, 4, 5],
    [1, 2, 4, 5, 1],
    [1, 2, 5, 1, 2],
    [1, 3, 1, 3, 5],
    [1, 3, 2, 4, 1],
    [1, 3, 3, 5, 2],
    [1, 3, 4, 1, 3],
    [1, 3, 5, 2, 4],
    [1, 4, 1, 4, 2],
    [1, 4, 2, 5, 3],
    [1, 4, 3, 1, 4],
    [1, 4, 4, 2, 5],
    [1, 4, 5, 3, 1],
    [1, 5, 1, 5, 4],
    [1, 5, 2, 1, 5],
    [1, 5, 3, 2, 1],
    [1, 5, 4, 3, 2],
    [1, 5, 5, 4, 3],
    [2, 1, 1, 5, 5],
    [2, 1, 2, 1, 1],
    [2, 1, 3, 2, 2],
    [2, 1, 4, 3, 3],
    [2, 1, 5, 4, 4],
    [2, 2, 1, 4, 3],
    [2, 2, 2, 5, 4],
    [2, 2, 3, 1, 5],
    [2, 2, 4, 2, 1],
    [2, 2, 5, 3, 2],
    [2, 3, 1, 2, 4],
    [2, 3, 2, 3, 5],
    [2, 3, 3, 4, 1],
    [2, 3, 4, 5, 2],
    [2, 3, 5, 1, 3],
    [2, 4, 1, 1, 1],
    [2, 4, 2, 2, 2],
    [2, 4, 3, 3, 3],
    [2, 4, 4, 4, 4],
    [2, 4, 5, 5, 5],
    [2, 5, 1, 3, 2],
    [2, 5, 2, 4, 3],
    [2, 5, 3, 5, 4],
    [2, 5, 4, 1, 5],
    [2, 5, 5, 2, 1]
]

# 因子水平映射关系

# 性别：从L50 X1(2水平)映射到 '男' 和 '女'
gender_map = {1: '男', 2: '女'}

# 百分位：从L50 X2(5水平)映射到5个百分位
percentile_map = {1: 'P5', 2: 'P10', 3: 'P50', 4: 'P90', 5: 'P95'}

# 路面类型：从L50 X3(5水平)映射到3个实际水平
# 分布为 2:2:1 即 (L1, L1, L2, L2, L3)
surface_map = {
    1: '机场平面', 2: '机场平面',  # 1级和2级映射到'机场平面'
    3: '斜坡', 4: '斜坡',          # 3级和4级映射到'斜坡'
    5: '粗糙地面'                  # 5级映射到'粗糙地面'
}

# 操作姿态：从L50 X4(5水平)映射到5种实际姿态
posture_map = {
    1: '水平推进', 2: '斜坡拉行', 3: '侧向拉提', 4: '转向操作', 5: '越障动作'
}

# 行李重量：从L50 X5(5水平)映射到3个实际重量等级
# 分布为 2:2:1 即 (L1, L1, L2, L2, L3)
weight_map = {
    1: '8kg', 2: '8kg',     # 1级和2级映射到'8kg'
    3: '14kg', 4: '14kg',   # 3级和4级映射到'14kg'
    5: '20kg'               # 5级映射到'20kg'
}

def generate_l50_experimental_design(l50_data):
    """
    基于L50正交表和因子映射关系生成50组实验设计方案
    
    参数:
        l50_data: L50正交表数据（仅包含X1-X5列）
    
    返回:
        DataFrame: 包含所有实验组合的数据框
    """
    experiments = []
    for i, row_levels in enumerate(l50_data):
        exp_id = f"E{i+1:03d}"  # 格式化为E001, E002, ... E050

        gender_level_oa = row_levels[0]        # X1: 性别
        percentile_level_oa = row_levels[1]    # X2: 百分位
        surface_level_oa = row_levels[2]       # X3: 路面类型
        posture_level_oa = row_levels[3]       # X4: 操作姿态
        weight_level_oa = row_levels[4]        # X5: 行李重量

        # 创建实验设置字典
        experiment_details = {
            "实验编号": exp_id,
            "性别": gender_map[gender_level_oa],
            "百分位": percentile_map[percentile_level_oa],
            "路面": surface_map[surface_level_oa],
            "姿态": posture_map[posture_level_oa],
            "重量": weight_map[weight_level_oa]
        }
        experiments.append(experiment_details)

    return pd.DataFrame(experiments)

def save_design_to_csv(df_design, output_dir=None):
    """
    将实验设计方案保存为CSV文件，文件名包含时间戳
    
    参数:
        df_design: 实验设计数据框
        output_dir: 输出目录。如果为None，则使用顶层的RESULTS_DIR。
    
    返回:
        str: 保存的文件路径
    """
    # 如果未指定output_dir，则使用全局定义的RESULTS_DIR（已基于SCRIPT_DIR）
    if output_dir is None:
        output_dir_to_use = RESULTS_DIR
    else:
        output_dir_to_use = Path(output_dir) # 如果传入，确保是Path对象

    # 确保输出目录存在
    output_dir_to_use.mkdir(parents=True, exist_ok=True)

    # 使用generate_timestamp()生成时间戳
    timestamp = generate_timestamp()
    # 统一文件名格式，与综合计算脚本中的find_latest_design_file的预期模式之一匹配
    filename = f"L50_experimental_design_{timestamp}.csv" 
    filepath = output_dir_to_use / filename

    # 保存为CSV，使用utf-8-sig编码以支持中文
    df_design.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"实验设计方案已保存到: {filepath}")
    return str(filepath)

def main():
    """
    主函数：生成并保存L50正交实验设计方案
    """
    print("开始生成L50正交实验设计...")
    
    # 使用L50正交表数据（选取的5列）生成实验设计方案
    df_experimental_design = generate_l50_experimental_design(l50_data_columns_X1_X2_X3_X4_X5)
    
    print("\n生成的实验设计方案预览:")
    print(df_experimental_design.head())
    
    # 保存设计方案到CSV文件，不传递output_dir，将使用save_design_to_csv中的默认逻辑（即顶层的RESULTS_DIR）
    saved_filepath = save_design_to_csv(df_experimental_design)
    
    print(f"\nL50实验设计方案生成完毕，共 {len(df_experimental_design)} 组实验。")
    print(f"文件保存在: {saved_filepath}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户手动中断程序执行。")
    except Exception as e:
        import traceback
        print(f"脚本执行过程中发生未捕获的错误: {e}")
        print("详细追溯信息:")
        traceback.print_exc() 