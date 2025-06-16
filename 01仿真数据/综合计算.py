import sys
import math
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from 数据加载 import DataLoader
import glob
import pandas as pd
import traceback

# 获取当前脚本文件所在的目录
SCRIPT_DIR = Path(__file__).resolve().parent

# --- 辅助向量运算函数 ---
def add_vectors(v1, v2):
    """向量加法"""
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]

def sub_vectors(v1, v2):
    """向量减法"""
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def scale_vector(scalar, v):
    """向量数乘"""
    return [scalar * v[0], scalar * v[1], scalar * v[2]]

def cross_product(v1, v2):
    """向量叉乘"""
    cx = v1[1] * v2[2] - v1[2] * v2[1]
    cy = v1[2] * v2[0] - v1[0] * v2[2]
    cz = v1[0] * v2[1] - v1[1] * v2[0]
    return [cx, cy, cz]

def magnitude(v):
    """计算向量大小（模）"""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

# --- 常量定义 (参考 计算公式.md 和 计算数据.md) ---
GRAVITY_ACCEL = 9.81  # 重力加速度 m/s^2

# 基于脚本目录定义其他路径
RESULTS_BASE_DIR = SCRIPT_DIR / "结果"
CALCULATION_RESULTS_DIR = RESULTS_BASE_DIR / "计算"
GRIP_RESULTS_DIR = RESULTS_BASE_DIR / "握把"
ORTHOGONAL_DESIGN_DIR = RESULTS_BASE_DIR / "正交设计"
DATABASE_DIR_FOR_DATALOADER = SCRIPT_DIR / "数据库"

# --- 手指力分布百分比 (参考 计算数据.md Section 2) ---
# 使用平均值作为默认百分比
FINGER_FORCE_PERC = {
    "拇指": 0.32,
    "食指": 0.28,
    "中指": 0.24,
    "无名指": 0.11,
    "小指": 0.05
}
# 确保总和为1
total_perc = sum(FINGER_FORCE_PERC.values())
if abs(total_perc - 1.0) > 1e-6:
    print(f"警告: 手指力分布百分比总和不为1 ({total_perc})，将按比例调整。")
    scale = 1.0 / total_perc
    FINGER_FORCE_PERC = {finger: perc * scale for finger, perc in FINGER_FORCE_PERC.items()}

# 可以引入一个参数来调整径向握力相对于总外部载荷的比例
# 这是一个经验因子，反映了为了稳定抓握额外需要的握力
RADIAL_GRIP_MULTIPLIER = 1.0 # 径向握力估算乘数：假设总径向握力 = 手部合力大小 * 这个乘数

# --- 初始化数据加载器 ---
# 显式传递数据库目录给 DataLoader
try:
    # 假设 DataLoader 的 __init__ 方法接受 database_dir 参数
    dataloader = DataLoader(database_dir=DATABASE_DIR_FOR_DATALOADER)
except TypeError:
    # 如果 DataLoader 不接受该参数（旧版本），则打印警告并使用默认行为
    # 这将依赖于 DataLoader 内部对路径的正确处理，或者您后续需要更新 DataLoader
    print("警告: 当前版本的 DataLoader 可能不支持显式指定 database_dir。脚本将尝试使用其默认数据库路径。")
    print(f"        如果遇到数据加载错误，请确保 DataLoader ({SCRIPT_DIR / '数据加载.py'}) 配置正确，或更新其以接受 database_dir 参数。")
    dataloader = DataLoader()

# --- 结果保存函数 ---
def generate_timestamp():
    """生成时间戳，格式为：yyyy.mm.dd_HH.MM.SS"""
    now = datetime.now()
    return now.strftime("%Y.%m.%d_%H.%M.%S")

def save_results_to_csv(results_data_list_of_dicts, overall_scenario_types_for_context, filename_prefix=None):
    """
    将结果保存为CSV文件
    
    参数:
        results_data_list_of_dicts: 结果数据列表，每个元素包含一次模拟的结果
        overall_scenario_types_for_context: 模拟的场景类型列表
        filename_prefix: 可选的文件名前缀
    
    返回:
        保存的文件路径
    """
    # 确保结果计算目录存在
    os.makedirs(CALCULATION_RESULTS_DIR, exist_ok=True)
    
    # 生成唯一的文件名
    timestamp = generate_timestamp()
    if filename_prefix:
        csv_filename = f"{filename_prefix}_{timestamp}.csv"
    else:
        csv_filename = f"simulation_results_{timestamp}.csv"
    
    # 使用Path对象构建路径
    csv_path = CALCULATION_RESULTS_DIR / csv_filename
    
    # 定义CSV字段 - 添加新的手指受力列
    fieldnames = [
        "用户ID", "性别", "百分位", "行李质量_kg", "场景类型",
        "斜坡角度_deg", "摩擦系数", "水平加速度_m_s2", "拉行系数", "操作类型",
        "肩关节角度_deg", "肘关节角度_deg", "腕关节角度_deg",
        "手部水平作用力_N", "手部垂直作用力_N", "手部合力_N",
        "腕关节力矩_X_Nm", "腕关节力矩_Y_Nm", "腕关节力矩_Z_Nm", "腕关节力矩_合_Nm",
        "肘关节力矩_X_Nm", "肘关节力矩_Y_Nm", "肘关节力矩_Z_Nm", "肘关节力矩_合_Nm",
        "肩关节力矩_X_Nm", "肩关节力矩_Y_Nm", "肩关节力矩_Z_Nm", "肩关节力矩_合_Nm",
        "MVC百分比", "疲劳惩罚值",
        # 新增手指估算总接触力列
        "拇指估算总接触力_N", "食指估算总接触力_N", "中指估算总接触力_N", "无名指估算总接触力_N", "小指估算总接触力_N"
    ]
    
    # 写入CSV文件，使用 utf-8-sig 编码以支持Excel打开中文
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result_dict in results_data_list_of_dicts:
            if result_dict is None:
                continue
            
            # 直接从 result_dict 获取 gender 和 percentile
            gender = result_dict.get("gender", "未知")
            percentile = result_dict.get("percentile", "未知")

            # 为 "用户ID" 列构造值
            csv_user_id_display = "未知用户"
            if gender != "未知" and percentile != "未知":
                csv_user_id_display = f"{gender}_{percentile}"
            else:
                # 如果 gender 或 percentile 未知，尝试从 "实验编号" 获取一些信息（如果需要）
                # 但通常情况下，如果模拟结果中有 gender 和 percentile, 这里就不会是未知
                # 如果确实需要一个ID，可以考虑使用 result_dict.get("实验编号", "未知实验")
                pass # csv_user_id_display 默认为 "未知用户" 或已构造

            # 场景类型获取逻辑不变
            scenario_type = result_dict.get("场景类型", "未知场景")
            
            # 计算力矩合力
            m_wrist = result_dict.get("M_wrist_Nm", [0, 0, 0])
            m_elbow = result_dict.get("M_elbow_Nm", [0, 0, 0])
            m_shoulder = result_dict.get("M_shoulder_Nm", [0, 0, 0])
            m_wrist_mag = math.sqrt(sum(x*x for x in m_wrist))
            m_elbow_mag = math.sqrt(sum(x*x for x in m_elbow))
            m_shoulder_mag = math.sqrt(sum(x*x for x in m_shoulder))
            
            # 获取手部力
            f_hand = result_dict.get("F_hand_N", [0, 0, 0])
            f_hand_mag = math.sqrt(sum(x*x for x in f_hand))
            
            # 获取关节角度
            joint_angles = result_dict.get("joint_angles_deg", {})
            
            # 获取操作参数
            operation_params = result_dict.get("operation_params", {})
            
            # 计算MVC百分比
            mvc_grip = result_dict.get("mvc_grip_N_used", 0)
            mvc_percent = (f_hand_mag / mvc_grip * 100) if mvc_grip > 0 else 0
            
            # 提取新的手指估算总接触力数据
            finger_forces = result_dict.get("finger_total_forces_estimated_N", {})
            
            # 写入一行 - 添加新的手指受力数据
            row_to_write = {
                "用户ID": csv_user_id_display,
                "性别": gender,
                "百分位": percentile,
                "行李质量_kg": result_dict.get("luggage_mass_kg", 0),
                "场景类型": scenario_type,
                "斜坡角度_deg": operation_params.get("slope_deg", 0),
                "摩擦系数": operation_params.get("rolling_friction_coeff", 0),
                "水平加速度_m_s2": operation_params.get("acceleration_h", operation_params.get("acceleration_h_m_s2",0)),
                "拉行系数": operation_params.get("c_pull_push_factor", 0),
                "操作类型": operation_params.get("operation_type", "unknown"),
                "肩关节角度_deg": joint_angles.get("shoulder", 0),
                "肘关节角度_deg": joint_angles.get("elbow", 0),
                "腕关节角度_deg": joint_angles.get("wrist", 0),
                "手部水平作用力_N": f_hand[0] if len(f_hand)>0 else 0,
                "手部垂直作用力_N": f_hand[2] if len(f_hand)>2 else 0,
                "手部合力_N": f_hand_mag,
                "腕关节力矩_X_Nm": m_wrist[0] if len(m_wrist)>0 else 0,
                "腕关节力矩_Y_Nm": m_wrist[1] if len(m_wrist)>1 else 0,
                "腕关节力矩_Z_Nm": m_wrist[2] if len(m_wrist)>2 else 0,
                "腕关节力矩_合_Nm": m_wrist_mag,
                "肘关节力矩_X_Nm": m_elbow[0] if len(m_elbow)>0 else 0,
                "肘关节力矩_Y_Nm": m_elbow[1] if len(m_elbow)>1 else 0,
                "肘关节力矩_Z_Nm": m_elbow[2] if len(m_elbow)>2 else 0,
                "肘关节力矩_合_Nm": m_elbow_mag,
                "肩关节力矩_X_Nm": m_shoulder[0] if len(m_shoulder)>0 else 0,
                "肩关节力矩_Y_Nm": m_shoulder[1] if len(m_shoulder)>1 else 0,
                "肩关节力矩_Z_Nm": m_shoulder[2] if len(m_shoulder)>2 else 0,
                "肩关节力矩_合_Nm": m_shoulder_mag,
                "MVC百分比": mvc_percent,
                "疲劳惩罚值": result_dict.get("P_fatigue", 0),
                # 提取并写入新的手指估算总接触力数据
                "拇指估算总接触力_N": finger_forces.get("拇指", 0),
                "食指估算总接触力_N": finger_forces.get("食指", 0),
                "中指估算总接触力_N": finger_forces.get("中指", 0),
                "无名指估算总接触力_N": finger_forces.get("无名指", 0),
                "小指估算总接触力_N": finger_forces.get("小指", 0)
            }
            writer.writerow(row_to_write)
    
    print(f"结果已保存到: {csv_path}")
    
    # --- 新增：保存手指力数据到握把目录 ---
    # 确保握把结果目录存在
    os.makedirs(GRIP_RESULTS_DIR, exist_ok=True)
    
    # 构建握把专用结果文件名
    grip_csv_filename = f"finger_forces_{timestamp}.csv"
    grip_csv_path = GRIP_RESULTS_DIR / grip_csv_filename
    
    # 定义握把CSV字段
    grip_fieldnames = [
        "用户ID", "性别", "百分位", "行李质量_kg", "场景类型", 
        "手部合力_N", "拇指力_N", "食指力_N", "中指力_N", "无名指力_N", "小指力_N",
        "拇指占比", "食指占比", "中指占比", "无名指占比", "小指占比"
    ]
    
    # 写入握把专用CSV文件
    with open(grip_csv_path, 'w', newline='', encoding='utf-8-sig') as grip_csvfile:
        grip_writer = csv.DictWriter(grip_csvfile, fieldnames=grip_fieldnames)
        grip_writer.writeheader()
        
        for result_dict in results_data_list_of_dicts:
            if result_dict is None:
                continue
            
            gender = result_dict.get("gender", "未知")
            percentile = result_dict.get("percentile", "未知")
            csv_user_id_display = f"{gender}_{percentile}" if gender != "未知" and percentile != "未知" else "未知用户"
            scenario_type = result_dict.get("场景类型", "未知场景")
            
            # 计算手部合力
            f_hand = result_dict.get("F_hand_N", [0, 0, 0])
            f_hand_mag = math.sqrt(sum(x*x for x in f_hand))
            
            # 获取手指力数据
            finger_forces = result_dict.get("finger_total_forces_estimated_N", {})
            
            # 补充缺失的手指力（如果有的话）
            for finger in ["拇指", "食指", "中指", "无名指", "小指"]:
                if finger not in finger_forces:
                    finger_forces[finger] = 0
            
            # 计算各手指占比
            finger_percentages = {}
            if f_hand_mag > 0:
                for finger, force in finger_forces.items():
                    finger_percentages[f"{finger}占比"] = force / f_hand_mag
            else:
                for finger in ["拇指", "食指", "中指", "无名指", "小指"]:
                    finger_percentages[f"{finger}占比"] = 0
            
            # 写入握把专用行
            grip_row = {
                "用户ID": csv_user_id_display,
                "性别": gender,
                "百分位": percentile,
                "行李质量_kg": result_dict.get("luggage_mass_kg", 0),
                "场景类型": scenario_type,
                "手部合力_N": f_hand_mag,
                "拇指力_N": finger_forces.get("拇指", 0),
                "食指力_N": finger_forces.get("食指", 0),
                "中指力_N": finger_forces.get("中指", 0),
                "无名指力_N": finger_forces.get("无名指", 0),
                "小指力_N": finger_forces.get("小指", 0),
                "拇指占比": finger_percentages.get("拇指占比", 0),
                "食指占比": finger_percentages.get("食指占比", 0),
                "中指占比": finger_percentages.get("中指占比", 0),
                "无名指占比": finger_percentages.get("无名指占比", 0),
                "小指占比": finger_percentages.get("小指占比", 0)
            }
            grip_writer.writerow(grip_row)
    
    print(f"手指力数据已保存到: {grip_csv_path}")
    # --- 新增结束 ---
    
    return csv_path

# --- 计算函数 ---

def calculate_external_load(luggage_mass_kg, slope_deg, rolling_friction_coeff, 
                            acceleration_h_m_s2, c_pull_push_factor, operation_type="pulling", 
                            k_impact_factor=0, k_vibration_factor=0):
    """
    计算外部载荷 F_hand (参考 计算公式.md Section 1)
    c_pull_push_factor: 拉行时为正 (如0.3)，推动时为负 (如-0.2)
    """
    theta_slope_rad = math.radians(slope_deg)
    
    N_force = luggage_mass_kg * GRAVITY_ACCEL * math.cos(theta_slope_rad)
    F_friction = rolling_friction_coeff * N_force
    F_slope = luggage_mass_kg * GRAVITY_ACCEL * math.sin(theta_slope_rad)
    F_accel = luggage_mass_kg * acceleration_h_m_s2
    
    F_horizontal_req = F_friction + F_slope + F_accel
    F_impact_val = k_impact_factor * luggage_mass_kg * GRAVITY_ACCEL 
    
    F_hand_base_x = F_horizontal_req
    F_hand_base_y = 0.0
    
    if operation_type == "pulling":
        F_hand_base_z = (c_pull_push_factor * F_horizontal_req) + F_impact_val
    elif operation_type == "pushing":
        F_hand_base_z = (c_pull_push_factor * F_horizontal_req) + F_impact_val 
    else: 
        F_hand_base_z = (c_pull_push_factor * F_horizontal_req) + F_impact_val

    Vec_F_hand_base = [F_hand_base_x, F_hand_base_y, F_hand_base_z]
    Vec_F_hand = scale_vector((1 + k_vibration_factor), Vec_F_hand_base)
    return Vec_F_hand

def get_segment_inertial_params(data_loader, gender, percentile_str):
    """
    获取指定用户的肢段惯性参数 (参考 计算公式.md Section 3 和 计算数据.md)
    
    参数:
        data_loader (LuggageDataLoader): 数据加载器实例。
        gender (str): 性别 ('男' 或 '女').
        percentile_str (str): 百分位字符串 (例如 'P50').
    """
    # DataLoader.get_anthropometry_for_simulation 期望中文性别 "男" 或 "女".
    # gender 参数从调用处 (run_single_simulation 或 run_batch_simulations_from_design)
    # 传入时已经是中文的 "男" 或 "女".
    # 因此，直接使用 gender 参数调用 data_loader 方法。
    # 移除以下转换逻辑:
    # gender_for_loader = gender
    # if gender == "男":
    #     gender_for_loader = "male"
    # elif gender == "女":
    #     gender_for_loader = "female"
    
    user_specific_data = data_loader.get_anthropometry_for_simulation(gender, percentile_str) # 直接使用 gender
    
    if user_specific_data is None:
        print(f"错误: 无法获取 {gender} {percentile_str} 的人体测量数据")
        return None
    
    # user_data 现在直接是对应用户的字典，不需要再通过 user_id_str 索引
    # 例如: user_specific_data['total_weight_kg']

    total_weight = user_specific_data.get("total_weight_kg")
    if total_weight is None:
        print(f"错误: {gender} {percentile_str} 的数据中缺少 total_weight_kg")
        return None
        
    params = {}
    # 注意: CSV列名中的百分号可能已被替换为下划线或文字
    # 需要与DataLoader加载后的实际DataFrame列名一致
    segments_config = [
        # (内部名, CSV中长度键, CSV中质量百分比键, CSV中质心近端百分比键)
        ("hand", "L_hand_m", "mass_perc_hand", "com_perc_hand_proximal"),
        ("forearm", "L_forearm_m", "mass_perc_forearm", "com_perc_forearm_proximal"),
        ("upperarm", "L_upperarm_m", "mass_perc_upperarm", "com_perc_upperarm_proximal")
    ]
    
    for seg_name, len_key, mass_perc_key, com_perc_key in segments_config:
        if not all(k in user_specific_data for k in [len_key, mass_perc_key, com_perc_key]):
            print(f"错误: {gender} {percentile_str} 的数据中缺少肢段 {seg_name} 的必要参数 ({len_key}, {mass_perc_key}, or {com_perc_key})")
            return None

        mass_kg = user_specific_data[mass_perc_key] * total_weight
        length_m = user_specific_data[len_key]
        com_dist_proximal_m = user_specific_data[com_perc_key] * length_m
        weight_vector_N = [0, 0, -mass_kg * GRAVITY_ACCEL]
        
        params[seg_name] = {
            "mass_kg": mass_kg,
            "length_m": length_m,
            "com_dist_proximal_m": com_dist_proximal_m,
            "weight_vector_N": weight_vector_N
        }
    
    # 确保 hand_length_m 存在于 user_specific_data 中 (或 L_hand_m)
    hand_length_key = "L_hand_m" # 或其他在DataLoader中定义的对应手长的键
    if hand_length_key not in user_specific_data:
        print(f"错误: {gender} {percentile_str} 的数据中缺少 {hand_length_key}")
        return None
    params["hand"]["hp_dist_proximal_m"] = 0.5 * user_specific_data[hand_length_key] # 使用实际手长

    return params

def calculate_position_vectors(joint_angles_deg, segment_data):
    """
    计算关键点的位置向量 (参考 计算公式.md Section 5, 简化为2D矢状面)
    joint_angles_deg: {'shoulder_flex': deg, 'elbow_flex': deg, 'wrist_flex': deg}
    segment_data: 包含各肢段长度和质心距离的字典 (来自 get_segment_inertial_params)
    """
    sh_flex_rad = math.radians(joint_angles_deg["shoulder_flex"])
    el_flex_rad = math.radians(joint_angles_deg["elbow_flex"])
    wr_flex_rad = math.radians(joint_angles_deg.get("wrist_flex", 0)) 

    alpha_upperarm = sh_flex_rad
    alpha_forearm = alpha_upperarm + el_flex_rad 
    alpha_hand = alpha_forearm + wr_flex_rad
    
    vecs = {}
    L_ua = segment_data["upperarm"]["length_m"]
    com_ua_dist = segment_data["upperarm"]["com_dist_proximal_m"]
    vecs["r_E_S"] = [L_ua * math.cos(alpha_upperarm), 0, L_ua * math.sin(alpha_upperarm)]
    vecs["r_CoM_ua_S"] = [com_ua_dist * math.cos(alpha_upperarm), 0, com_ua_dist * math.sin(alpha_upperarm)]

    L_fa = segment_data["forearm"]["length_m"]
    com_fa_dist = segment_data["forearm"]["com_dist_proximal_m"]
    vecs["r_W_E"] = [L_fa * math.cos(alpha_forearm), 0, L_fa * math.sin(alpha_forearm)]
    vecs["r_CoM_fa_E"] = [com_fa_dist * math.cos(alpha_forearm), 0, com_fa_dist * math.sin(alpha_forearm)]
    
    hp_h_dist = segment_data["hand"]["hp_dist_proximal_m"]
    com_h_dist = segment_data["hand"]["com_dist_proximal_m"]
    vecs["r_HP_W"] = [hp_h_dist * math.cos(alpha_hand), 0, hp_h_dist * math.sin(alpha_hand)]
    vecs["r_CoM_h_W"] = [com_h_dist * math.cos(alpha_hand), 0, com_h_dist * math.sin(alpha_hand)]
    
    return vecs

def calculate_joint_kinetics(F_hand_N, segment_weight_vectors, pos_vectors):
    """
    计算关节反作用力和力矩 (参考 计算公式.md Section 4)
    segment_weight_vectors: {'hand': W_h, 'forearm': W_f, 'upperarm': W_u}
    pos_vectors: 来自 calculate_position_vectors 的结果
    """
    W_hand_N = segment_weight_vectors["hand"]
    W_forearm_N = segment_weight_vectors["forearm"]
    W_upperarm_N = segment_weight_vectors["upperarm"]
    
    kinetics = {}
    
    kinetics["F_wrist"] = scale_vector(-1, add_vectors(F_hand_N, W_hand_N))
    term1_M_wrist = cross_product(pos_vectors["r_HP_W"], F_hand_N)
    term2_M_wrist = cross_product(pos_vectors["r_CoM_h_W"], W_hand_N)
    kinetics["M_wrist"] = scale_vector(-1, add_vectors(term1_M_wrist, term2_M_wrist))
    
    kinetics["F_elbow"] = sub_vectors(kinetics["F_wrist"], W_forearm_N)
    term_elbow_cross1 = cross_product(pos_vectors["r_W_E"], scale_vector(-1, kinetics["F_wrist"]))
    term_elbow_cross2 = cross_product(pos_vectors["r_CoM_fa_E"], W_forearm_N)
    # Calculate original M_elbow value
    original_m_elbow = sub_vectors(kinetics["M_wrist"], add_vectors(term_elbow_cross1, term_elbow_cross2))
    # Store the sign-flipped version as per advice
    kinetics["M_elbow"] = scale_vector(-1, original_m_elbow)

    kinetics["F_shoulder"] = sub_vectors(kinetics["F_elbow"], W_upperarm_N)
    term_shoulder_cross1 = cross_product(pos_vectors["r_E_S"], scale_vector(-1, kinetics["F_elbow"]))
    term_shoulder_cross2 = cross_product(pos_vectors["r_CoM_ua_S"], W_upperarm_N)
    # Calculate original M_shoulder value, using the original_m_elbow for consistency
    original_m_shoulder = sub_vectors(original_m_elbow, add_vectors(term_shoulder_cross1, term_shoulder_cross2))
    # Store the sign-flipped version as per advice
    kinetics["M_shoulder"] = scale_vector(-1, original_m_shoulder)
    
    return kinetics

def calculate_fatigue_penalty(F_hand_N_vec, mvc_grip_N, mvc_threshold_ratio=0.03):
    """
    计算疲劳惩罚项 (参考 计算公式.md Section 6.2)
    """
    f_hand_mag = magnitude(F_hand_N_vec)
    if mvc_grip_N <= 0: 
        return float('inf') if f_hand_mag > 0 else 0 
        
    ratio = f_hand_mag / mvc_grip_N
    fatigue_penalty = max(0, ratio - mvc_threshold_ratio)**2
    return fatigue_penalty

def calculate_finger_forces(F_hand_N_vec):
    """
    估算行李箱通过握柄作用在每个手指上的力。

    这是基于手指力分布百分比的简化载荷分配模型，
    将总手部合力的大小按百分比分配作为估算的总接触力大小。

    参数:
        F_hand_N_vec (list): 手部总的作用力向量 (来自 calculate_external_load)。

    返回:
        dict: 包含每个手指估算总接触力大小的字典，键为手指名称，值为力大小 (N)。
              示例: {"拇指": F_thumb_total_est_N, "食指": F_index_total_est_N, ...}
    """
    # 计算手部总外部力的大小
    F_hand_magnitude = magnitude(F_hand_N_vec)

    # --- 简化载荷分配模型 ---
    # 假设每个手指承受的总接触力大小，是总外部力大小按百分比分配的结果。
    # 这隐含了手指力分布百分比也适用于分配总外部载荷，而不是仅径向握力。
    
    finger_total_estimated_forces = {}
    for finger, perc in FINGER_FORCE_PERC.items():
        # 估算每个手指的总接触力 = 总手部合力大小 * 该手指的百分比
        finger_total_estimated_forces[finger] = F_hand_magnitude * perc

    return finger_total_estimated_forces

def simulate_luggage_operation(data_loader, gender, percentile_str, luggage_mass_kg, 
                               operation_params, joint_angles_deg, scene_type, verbose=True):
    """
    执行一次完整的生物力学计算模拟。
    
    参数:
        data_loader (LuggageDataLoader): 初始化后的数据加载器实例。
        gender (str): 性别 ('男' 或 '女').
        percentile_str (str): 百分位字符串 (例如 'P50').
        luggage_mass_kg (float): 行李箱质量 (kg).
        operation_params (dict): 操作参数字典.
        joint_angles_deg (dict): 关节角度字典.
        scene_type (str): 场景类型.
        verbose (bool): 是否打印详细模拟过程.
    """
    if verbose:
        # 构造一个 user_id_display_str 用于打印，更友好
        gender_display = "男性" if gender == "男" or gender.lower() == "male" else "女性"
        print(f"--- 开始模拟: 用户 {gender_display}{percentile_str}, 行李质量 {luggage_mass_kg}kg ---")

    # 1. 获取指定用户的肢段惯性参数
    #    现在 get_segment_inertial_params 直接使用 data_loader, gender, percentile_str
    segment_inertial_params = get_segment_inertial_params(data_loader, gender, percentile_str)
    if segment_inertial_params is None:
        print(f"错误: 无法为用户 {gender}{percentile_str} 获取肢段惯性参数。")
        return None # 或者返回一个包含错误信息的字典

    segment_weights = {
        "hand": segment_inertial_params["hand"]["weight_vector_N"],
        "forearm": segment_inertial_params["forearm"]["weight_vector_N"],
        "upperarm": segment_inertial_params["upperarm"]["weight_vector_N"]
    }

    # 2. 计算外部载荷 F_hand
    F_hand_N = calculate_external_load(
        luggage_mass_kg,
        operation_params.get("slope_deg", 0),
        operation_params.get("rolling_friction_coeff", 0.035),
        operation_params.get("acceleration_h", 0.5), # 使用不带单位的键名
        operation_params.get("c_pull_push_factor", 0.3),
        operation_params.get("operation_type", "pulling"),
        operation_params.get("k_impact_factor", 0),
        operation_params.get("k_vibration_factor", 0)
    )
    if verbose:
        print(f"  计算得到手部作用力 (F_hand): {['{:.3f}'.format(x) for x in F_hand_N]} N")

    # 3. 计算位置向量
    #    确保 joint_angles_deg 包含 'shoulder_flex', 'elbow_flex', 'wrist_flex'
    #    这些应该从外部传入的 joint_angles_deg 字典中获取，并映射到模型期望的名称
    model_joint_angles = {
        "shoulder_flex": joint_angles_deg.get("shoulder", 0), # 映射 "shoulder" 到 "shoulder_flex"
        "elbow_flex": joint_angles_deg.get("elbow", 0),       # 映射 "elbow" 到 "elbow_flex"
        "wrist_flex": joint_angles_deg.get("wrist", 0)        # 映射 "wrist" 到 "wrist_flex"
    }
    position_vectors = calculate_position_vectors(model_joint_angles, segment_inertial_params)

    # 4. 计算关节动力学
    joint_kinetics = calculate_joint_kinetics(F_hand_N, segment_weights, position_vectors)
    
    # 将力矩从 N·mm 转换为 N·m (除以1000)
    M_wrist_Nm = [x/1000 for x in joint_kinetics.get("M_wrist", [0,0,0])]
    M_elbow_Nm = [x/1000 for x in joint_kinetics.get("M_elbow", [0,0,0])]
    M_shoulder_Nm = [x/1000 for x in joint_kinetics.get("M_shoulder", [0,0,0])]
    
    if verbose:
        print(f"  腕关节力矩 (M_wrist): {['{:.3f}'.format(x) for x in M_wrist_Nm]} N·m")
        print(f"  肘关节力矩 (M_elbow): {['{:.3f}'.format(x) for x in M_elbow_Nm]} N·m")
        print(f"  肩关节力矩 (M_shoulder): {['{:.3f}'.format(x) for x in M_shoulder_Nm]} N·m")
        
    # 5. 计算疲劳惩罚 (如果需要)
    #    使用 DataLoader 的 get_mvc_data 方法获取特定用户的 MVC 数据
    #    gender 参数应为中文 "男" 或 "女"
    #    muscle_group 参数应为功能人因参数CSV中 '参数' 列对应的 '最大握力' 相关条目
    mvc_grip_N_value = data_loader.get_mvc_data(gender, "最大握力", percentile_str) # 传入百分位信息
    
    if mvc_grip_N_value is None:
        print(f"警告: 无法获取用户 {gender}{percentile_str} 的最大握力MVC数据，将使用默认值0进行疲劳计算，可能导致疲劳惩罚不准确或为inf。")
        mvc_grip_N = 0 # 或者一个合理的默认值，或者如果为0则后续疲劳计算应特殊处理
    else:
        mvc_grip_N = float(mvc_grip_N_value) # 确保是浮点数
    
    P_fatigue = calculate_fatigue_penalty(F_hand_N, mvc_grip_N)
    if verbose:
        print(f"  疲劳惩罚项 (P_fatigue): {P_fatigue:.3f} (基于MVC: {mvc_grip_N:.2f} N)")

    # 在关节动力学和疲劳计算后，添加手指力计算
    # --- 新增：计算每个手指的估算总接触力 ---
    finger_total_forces_estimated = calculate_finger_forces(F_hand_N)
    
    if verbose:
        print("  估算的手指总接触力 (N):")
        for finger, force in finger_total_forces_estimated.items():
            print(f"    {finger}: {force:.3f} N")
    # --- 新增结束 ---

    if verbose:
        gender_display_end = "男性" if gender == "男" else "女性"
        print(f"--- 模拟结束: {gender_display_end}{percentile_str} ---")
        
    # --- 修改返回结果字典，包含新的手指受力信息 ---
    result_dict = {
        "F_hand_N": F_hand_N,
        "M_wrist_Nm": M_wrist_Nm,
        "M_elbow_Nm": M_elbow_Nm,
        "M_shoulder_Nm": M_shoulder_Nm,
        "P_fatigue": P_fatigue,
        "luggage_mass_kg": luggage_mass_kg,
        "operation_params": operation_params,
        "joint_angles_deg": joint_angles_deg,
        "gender": gender, 
        "percentile": percentile_str,
        "mvc_grip_N_used": mvc_grip_N,
        "场景类型": scene_type,
        # 新增手指估算总接触力结果
        "finger_total_forces_estimated_N": finger_total_forces_estimated
    }
    # --- 修改返回结果字典结束 ---

    return result_dict

def run_single_simulation(data_loader, gender, percentile, luggage_mass_kg, scenario_type="机场平路"):
    """
    运行单次模拟并打印结果。
    现在也需要 data_loader。
    """
    print(f"\n模拟{percentile}{gender} ({scenario_type}):\n")
    
    # 从 DataLoader 获取操作参数和关节角度
    operation_params = data_loader.get_operation_scenario_params(scenario_type)
    joint_angles_deg = data_loader.get_joint_angles_for_scenario(scenario_type)

    if operation_params is None:
        print(f"错误: 无法从 DataLoader 获取场景 '{scenario_type}' 的操作参数。使用默认值或跳过。")
        # 可以选择提供一个非常基础的默认值，或者直接返回 None
        # 为了简单起见，如果获取失败，我们先打印错误并返回
        # operation_params = {} # 或者一个最小化的默认字典
        return None 

    if joint_angles_deg is None:
        print(f"错误: 无法从 DataLoader 获取场景 '{scenario_type}' 的关节角度。使用默认值或跳过。")
        # joint_angles_deg = {} # 或者一个最小化的默认字典
        return None

    # 调用更新后的 simulate_luggage_operation
    result = simulate_luggage_operation(
        data_loader=data_loader,
        gender=gender,
        percentile_str=percentile,
        luggage_mass_kg=luggage_mass_kg,
        operation_params=operation_params,
        joint_angles_deg=joint_angles_deg,
        scene_type=scenario_type,
        verbose=True
    )

    # 打印完整结果
    if result:
        print(f"  手部作用力 (F_hand): {result.get('F_hand_N')} N")
        print(f"  腕关节力矩 (M_wrist): {result.get('M_wrist_Nm')} N·m")
        print(f"  肘关节力矩 (M_elbow): {result.get('M_elbow_Nm')} N·m")
        print(f"  肩关节力矩 (M_shoulder): {result.get('M_shoulder_Nm')} N·m")
        print(f"  疲劳惩罚项 (P_fatigue): {result.get('P_fatigue')}")
    else:
        print("模拟未能生成有效结果。")
        
    return result

def find_latest_design_file(directory_path_obj, file_type="csv"):
    """
    查找指定目录下最新的正交设计文件
    directory_path_obj: 期望是一个 Path 对象
    """
    # 搜索多种可能的文件名模式
    # 使用 Path 对象的 / 操作符构建模式
    patterns_path_objs = [
        directory_path_obj / f"实验设计方案_*.{file_type}",
        directory_path_obj / f"L50_experimental_design_*.{file_type}",
        directory_path_obj / f"正交设计方案_*.{file_type}"
    ]
    
    all_files_str = []
    for p_obj in patterns_path_objs:
        # glob.glob 期望字符串路径
        all_files_str.extend(glob.glob(str(p_obj)))
    
    if not all_files_str:
        print(f"错误: 在目录 {directory_path_obj} 未找到实验设计方案文件，尝试了以下模式:")
        for p_obj in patterns_path_objs:
            print(f"  - {p_obj}")
        return None
    
    # 返回最新的文件
    latest_file_str = max(all_files_str, key=os.path.getmtime)
    print(f"找到最新的实验设计文件: {latest_file_str}")
    return Path(latest_file_str) # 返回 Path 对象

def load_experiment_design():
    """加载最新的实验设计方案"""
    # 使用 ORTHOGONAL_DESIGN_DIR (已经是Path对象)
    filepath_obj = find_latest_design_file(ORTHOGONAL_DESIGN_DIR, "csv")
    if not filepath_obj:
        print("错误：未能定位最新的实验设计CSV文件。请先运行正交设计脚本。")
        return None
    
    print(f"从以下路径加载实验设计方案: {filepath_obj}")
    try:
        df = pd.read_csv(filepath_obj)
        print(f"成功加载 {len(df)} 个实验设计组合。")
        return df
    except Exception as e:
        print(f"加载实验设计文件 {filepath_obj} 时出错: {e}")
        return None

def run_batch_simulations_from_design(data_loader, design_df):
    """
    根据实验设计方案 (DataFrame) 运行批量模拟。

    Args:
        data_loader (LuggageDataLoader): 已初始化的数据加载器。
        design_df (pd.DataFrame): 包含实验设计方案的DataFrame。
                                   适配L50实验设计CSV格式
                                   期望列: "实验编号", "性别", "百分位", "路面", "姿态", "重量"
                                   
    Returns:
        pd.DataFrame: 包含所有模拟结果的DataFrame。
    """
    if design_df is None or design_df.empty:
        print("实验设计方案为空，无法运行批量模拟。")
        return pd.DataFrame()

    all_results = []
    print(f"\n开始执行基于设计方案的 {len(design_df)} 次模拟...")

    # 路面类型到各参数的映射 - 提高载荷版本
    SCENE_PARAMS = {
        "机场平面": {
            "斜坡角度_deg": 0.0,
            "摩擦系数": 0.015, # 略微提高摩擦系数
            "水平加速度_m_s2": 0.8, # 提高加速度
            "拉行系数": 1.0, # 保持不变
            "k_impact_factor": 0.0,
            "k_vibration_factor": 0.0
        },
        "斜坡": {
            "斜坡角度_deg": 8.0, # 使用更大的坡度角度 (MD中航空坡道上限)
            "摩擦系数": 0.03, # 略微提高摩擦
            "水平加速度_m_s2": 0.7, # 提高加速度
            "拉行系数": 1.3, # 略微提高拉行系数
            "k_impact_factor": 0.0,
            "k_vibration_factor": 0.08 # 略微提高振动
        },
        "粗糙地面": {
            "斜坡角度_deg": 0.0,
            "摩擦系数": 0.05, # 提高摩擦系数
            "水平加速度_m_s2": 1.0, # 显著提高加速度
            "拉行系数": 1.5, # 提高拉行系数
            "k_impact_factor": 0.5, # 引入冲击因子
            "k_vibration_factor": 0.3 # 提高振动因子
        }
    }

    # 姿态类型到关节角度的映射
    # POSTURE_JOINT_ANGLES 字典保持不变，因为它代表了用户在不同姿态下的典型关节角度
    POSTURE_JOINT_ANGLES = {
        "水平推进": {
            "shoulder": 30.0,
            "elbow": 100.0,
            "wrist": 0.0,
            "knee_flex": 25.0
        },
        "斜坡拉行": {
            "shoulder": 40.0,
            "elbow": 90.0,
            "wrist": 15.0,
            "knee_flex": 30.0
        },
        "侧向拉提": {
            "shoulder": 60.0,
            "elbow": 95.0,
            "wrist": 20.0,
            "knee_flex": 20.0
        },
        "转向操作": {
            "shoulder": 35.0,
            "elbow": 85.0,
            "wrist": 25.0,
            "knee_flex": 25.0
        },
        "越障动作": {
            "shoulder": 45.0,
            "elbow": 110.0,
            "wrist": 10.0,
            "knee_flex": 40.0
        }
    }

    # 操作类型映射
    POSTURE_TO_OPERATION = {
        "水平推进": "pushing",
        "斜坡拉行": "pulling",
        "侧向拉提": "lifting",
        "转向操作": "turning",
        "越障动作": "lifting" # 越障动作通常是提拉或向上推动以跨越障碍
    }

    for index, row in design_df.iterrows():
        try:
            experiment_id = row["实验编号"]
            gender = row["性别"] 
            percentile = row["百分位"]
            scene_type = row["路面"]
            posture = row["姿态"]
            weight_str = row["重量"]
            
            # 从重量字符串中提取数值部分
            luggage_mass_kg = float(weight_str.replace("kg", ""))
            
            print(f"\n------ 执行实验 {experiment_id} ------")
            print(f"模拟参数: 性别={gender}, 百分位={percentile}")
            print(f"行李质量={luggage_mass_kg}kg, 场景={scene_type}, 姿态={posture}")
            
            # 根据路面类型获取对应的参数
            scene_params = SCENE_PARAMS.get(scene_type, SCENE_PARAMS["机场平面"])
            
            # 设置操作参数
            operation_params = {
                "slope_deg": scene_params["斜坡角度_deg"],
                "rolling_friction_coeff": scene_params["摩擦系数"],
                "acceleration_h": scene_params["水平加速度_m_s2"],
                "c_pull_push_factor": scene_params["拉行系数"],
                "operation_type": POSTURE_TO_OPERATION.get(posture, "pulling"),
                "k_impact_factor": scene_params["k_impact_factor"],
                "k_vibration_factor": scene_params["k_vibration_factor"]
            }
            
            # 根据姿态类型获取关节角度
            joint_angles_deg = POSTURE_JOINT_ANGLES.get(posture, POSTURE_JOINT_ANGLES["水平推进"])

            print(f"操作参数: {operation_params}")
            print(f"关节角度: {joint_angles_deg}")

            simulation_result = simulate_luggage_operation(
                data_loader=data_loader,
                gender=gender,
                percentile_str=percentile,
                luggage_mass_kg=luggage_mass_kg,
                operation_params=operation_params,
                joint_angles_deg=joint_angles_deg,
                scene_type=scene_type,
                verbose=False
            )

            if simulation_result:
                result_with_id = {"实验编号": experiment_id, **simulation_result}
                all_results.append(result_with_id)
                print(f"实验 {experiment_id} 完成。")
            else:
                print(f"实验 {experiment_id} 未返回有效结果。")

        except Exception as e:
            print(f"实验 {experiment_id} 失败: {e}")
            traceback.print_exc()
            # 选择是否将失败信息也记录到结果中，或者仅跳过
            # all_results.append({"实验编号": experiment_id, "status": "failed", "error": str(e)})

    if not all_results:
        print("\n批量模拟未产生任何结果。")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    
    # 可能需要重新排序列，使 "实验编号" 在前面
    if "实验编号" in results_df.columns:
        cols = ["实验编号"] + [col for col in results_df.columns if col != "实验编号"]
        results_df = results_df[cols]
        
    return results_df

def main():
    # 确保数据加载器已正确初始化
    try:
        # 简单测试数据加载器
        weight = dataloader.get_value('anthropometry', '体重', filters={'性别': '男'}, column='P50')
        if weight is None:
            print("警告: 数据加载器初始化可能有问题，无法读取基本数据")
        else:
            print(f"数据加载器测试: 男性P50体重 = {weight} kg")
    except Exception as e:
        print(f"错误: 数据加载器初始化失败: {e}")
        sys.exit(1)
    
    print("\n=== 使用数据加载器运行模拟 ===")
    
    # 收集单次模拟的结果
    results = []
    scenario_types = []
    
    # 运行一些预设的单次模拟作为快速检查
    test_scenarios = [
        {"gender": "男", "percentile": "P50", "luggage_mass_kg": 6.5, "scene_type": "机场平路"},
        {"gender": "女", "percentile": "P50", "luggage_mass_kg": 6.5, "scene_type": "机场平路"},
        {"gender": "男", "percentile": "P50", "luggage_mass_kg": 6.5, "scene_type": "斜坡"} 
    ]

    for ts_params in test_scenarios:
        print(f"\n模拟{ts_params['percentile']}{ts_params['gender']} ({ts_params['scene_type']}):")
        result = run_single_simulation(
            data_loader=dataloader,
            gender=ts_params['gender'],
            percentile=ts_params['percentile'],
            luggage_mass_kg=ts_params['luggage_mass_kg'],
            scenario_type=ts_params['scene_type']
        )
        if result:
            results.append(result)
            scenario_types.append(ts_params['scene_type'])
        print("------------------------------")

    if results:
        # 保存初始测试结果 (仅CSV)
        # 注意: save_results_to_csv 的第二个参数 scenario_types 之前是 unique_scenes，现在如果每个result里有场景信息，可能不需要
        # 但为了保持函数签名，可以传递一个包含所有场景类型的列表，或者让函数更智能地处理
        initial_scene_types = [res.get("场景类型", "未知场景") for res in results]
        saved_csv_path_initial = save_results_to_csv(results, list(set(initial_scene_types)), filename_prefix="initial_test")
        print(f"\n测试结果已保存到:\n{saved_csv_path_initial}")
    else:
        print("\n初始模拟未产生任何结果进行保存。")

    # --- 批量模拟部分 --- 
    # ... (加载实验设计文件和运行批量模拟的逻辑不变) ...
    design_file_path = find_latest_design_file(ORTHOGONAL_DESIGN_DIR)
    if design_file_path:
        print(f"从以下路径加载实验设计方案: {design_file_path}")
        experimental_design_df = pd.read_csv(design_file_path)
        print(f"成功加载 {len(experimental_design_df)} 个实验设计组合。")
        
        batch_results_df = run_batch_simulations_from_design(dataloader, experimental_design_df)
        
        if batch_results_df is not None and not batch_results_df.empty:
            print(f"\n批量模拟完成，共获得 {len(batch_results_df)} 个结果。")
            results_list_of_dicts = batch_results_df.to_dict(orient='records')
            
            # 提取唯一的场景类型用于文件名或上下文，如果 save_results_to_csv 仍需要
            # 假设 batch_results_df DataFrame 包含 "场景类型" 列
            unique_scenes_for_filename = batch_results_df["场景类型"].unique().tolist()

            saved_csv_path_batch = save_results_to_csv(
                results_list_of_dicts, 
                unique_scenes_for_filename, 
                filename_prefix="orthogonal_simulation_results"
            )
            print(f"批量模拟结果已保存到: {saved_csv_path_batch}")
        else:
            print("\n批量模拟运行结束，但未产生任何有效结果进行保存。")
    else:
        print("未能加载实验设计方案，跳过批量模拟。")

    print("\n程序执行完毕。")

if __name__ == "__main__":
    main()
