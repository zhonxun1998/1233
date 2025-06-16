import pandas as pd
import os
import numpy as np
import math
from pathlib import Path
import traceback
import re # 确保 re 被导入
import io # 确保 io 被导入

# 设置数据库目录路径（相对路径或绝对路径）
DATABASE_DIR = Path(__file__).parent / "csvdatabase"

# 定义文件名到 self.data 键的映射
# (这个全局变量CSV_FILES可能不再需要，如果load_all_data硬编码文件名和键)
CSV_KEY_MAP = {
    "人体测量数据.csv": "anthropometry",
    "肢段惯性参数.csv": "segment_inertia",
    "关节数据.csv": "joint_data",
    "肌肉力学数据.csv": "muscle_mech", # 与load_all_data中的键名一致
    "行李箱场景参数.csv": "luggage_scenario",
    "功能人因参数.csv": "functional_ergonomics", # 与load_all_data中的键名一致
    "动态仿真参数.csv": "dynamic_sim_params" # 与load_all_data中的键名一致
}

class DataLoader:
    """数据加载器类，用于从CSV文件加载数据并提供给计算模块使用"""
    
    def __init__(self, database_dir_override=None):
        """初始化数据加载器"""
        if database_dir_override:
            self.database_dir = Path(database_dir_override)
        else:
            # 默认DATABASE_DIR的计算方式
            # 获取此文件 (数据加载.py) 的绝对路径
            current_script_path = Path(__file__).resolve() 
            # 父目录 (仿真3.0) 再加上 "数据库" (中文目录名)
            self.database_dir = current_script_path.parent / "数据库"
        
        print(f"--- DataLoader Initializing with DATABASE_DIR ---")
        resolved_db_dir = self.database_dir.resolve()
        print(f"Attempting to use database directory: {resolved_db_dir}")
        
        if not resolved_db_dir.exists():
            print(f"错误: 数据库目录不存在或无法访问: {resolved_db_dir}")
            # 在这种情况下，后续的 _load_all_data 会因为找不到文件而出错，这里提前警告
        elif not resolved_db_dir.is_dir():
            print(f"错误: 指定的数据库路径不是一个有效的目录: {resolved_db_dir}")
        else:
            print(f"数据库目录已确认存在且为有效目录: {resolved_db_dir}")
        
        self.data = {}
        self._load_all_data()
        
    def _clean_column_name(self, col_name):
        """清理CSV列名，移除特殊字符，替换空格等"""
        if not isinstance(col_name, str):
            return col_name
        # 移除BOM等不可见字符并去除首尾空格
        cleaned_name = col_name.replace('\ufeff', '').strip()
        
        # --- 简化的清理逻辑 --- 
        # 暂时只做最基本的清理，避免过度处理导致有效列名丢失
        # 后续如果发现特定字符确实引起pandas问题，可以再针对性添加替换规则
        # 例如，如果列名中包含点 '.' 或方括号 '[]' 等可能引起属性访问问题的字符，可以替换
        # cleaned_name = cleaned_name.replace('.', '_') 
        
        # 保留原始列名的大部分特征，而不是强制移除大量字符
        # 原来的re.sub过于激进，导致很多列名变为空字符串
        # cleaned_name = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '', cleaned_name)
        # cleaned_name = re.sub(r'_+', '_', cleaned_name)
        # cleaned_name = cleaned_name.strip('_')
        
        # 一个更保守的策略：如果原始列名在pandas中通常是有效的，
        # 那么除了首尾空格和BOM，可能不需要太多改动。
        # 我们需要确保的是，get_segment_inertial_params 等方法中引用的列名
        # 与这里清理后生成的列名一致。

        # 让我们检查一下最初的打印输出，清理前的列名是什么
        # 在 _load_and_clean_csv 中，pd.read_csv 之后，应用清理前，列名是原始的
        # 然后应用了 self._clean_column_name
        # 假设原始列名本身是 pandas 可接受的，例如：'体段', '质量百分比_体重_百分比'
        # 那么这些就不应该被清理掉。

        # 暂时返回仅处理了BOM和首尾空格的列名
        return cleaned_name

    def _load_and_clean_csv(self, filename, key_in_data_dict):
        """
        加载单个CSV文件，尝试多种编码和分隔符，并清理列名。
        将加载的DataFrame存储在self.data[key_in_data_dict]中。
        """
        file_path = self.database_dir / filename
        if not file_path.exists():
            print(f"警告: 找不到文件 {filename} at {file_path}")
            self.data[key_in_data_dict] = pd.DataFrame() # 存一个空的DataFrame
            return

        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        separators_to_try = [',', ';', '\\t']
        df_loaded = None

        for encoding in encodings_to_try:
            if df_loaded is not None: break
            for sep in separators_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=sep, engine='python')
                    # 简单检查是否成功加载 (例如，列数大于0)
                    if df.shape[1] > 0:
                        df_loaded = df
                        print(f"已加载 {filename} (编码: {encoding}, 分隔符: '{sep}')")
                        break
                except Exception: # pd.errors.ParserError, UnicodeDecodeError, etc.
                    continue
            
            # 如果特定编码的标准分隔符尝试失败，可以尝试更宽松的读取
            if df_loaded is None:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', quoting=3, on_bad_lines='skip')
                    if df.shape[1] > 0:
                        df_loaded = df
                        print(f"已加载 {filename} (编码: {encoding}, 灵活解析模式)")
                        break
                except:
                    continue
        
        if df_loaded is None:
            try: # 最后尝试，手动读取并尝试修复
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                if not lines:
                    print(f"警告: 文件 {filename} 为空或无法读取。")
                    self.data[key_in_data_dict] = pd.DataFrame()
                    return

                first_line = lines[0].strip()
                detected_sep = ',' # default
                if '\\t' in first_line: detected_sep = '\\t'
                elif ';' in first_line: detected_sep = ';'
                
                header = [self._clean_column_name(h) for h in lines[0].strip().split(detected_sep)]
                num_columns = len(header)
                
                data_rows = []
                for line in lines[1:]:
                    parts = line.strip().split(detected_sep)
                    if len(parts) == num_columns:
                        data_rows.append(parts)
                    elif len(parts) > num_columns: #  取前num_columns个
                         data_rows.append(parts[:num_columns])
                    else: # 少了，补空值
                         data_rows.append(parts + [''] * (num_columns - len(parts)))

                if data_rows:
                    df_loaded = pd.DataFrame(data_rows, columns=header)
                    print(f"已加载 {filename} (通过手动解析和修复)")
                else: # 只有表头
                    df_loaded = pd.DataFrame(columns=header)


            except Exception as e:
                print(f"警告: 最终无法加载文件 {filename}: {e}")
                self.data[key_in_data_dict] = pd.DataFrame() # 确保键存在
                return

        if df_loaded is not None:
            # 清理列名
            df_loaded.columns = [self._clean_column_name(col) for col in df_loaded.columns]
            # 移除完全是NA的行和列 (可选)
            # df_loaded.dropna(axis=0, how='all', inplace=True)
            # df_loaded.dropna(axis=1, how='all', inplace=True)
            self.data[key_in_data_dict] = df_loaded
        else:
            print(f"警告: 文件 {filename} 加载失败，即使尝试了多种方法。")
            self.data[key_in_data_dict] = pd.DataFrame()

    def _load_all_data(self):
        """加载所有CSV文件数据"""
        # 使用定义的映射关系
        for filename, key in CSV_KEY_MAP.items():
            self._load_and_clean_csv(filename, key)
            # 确认加载后的DataFrame不为空，并且列名是我们期望的
            if key in self.data and not self.data[key].empty:
                 print(f"加载并清理完成: {filename} as self.data['{key}']. 列名: {self.data[key].columns.tolist()}")
                 # --- 添加的诊断代码开始 ---
                 if key == 'segment_inertia':
                    df_segment_inertia = self.data['segment_inertia']
                    if '体段' in df_segment_inertia.columns:
                        print(f"诊断: 'segment_inertia' DataFrame 中 '体段' 列的前5个唯一值: {df_segment_inertia['体段'].unique()[:5]}")
                    else:
                        print(f"诊断: 'segment_inertia' DataFrame 中未找到名为 '体段' 的列。实际列名: {df_segment_inertia.columns.tolist()}")
                 # --- 添加的诊断代码结束 ---
            elif key in self.data and self.data[key].empty:
                 print(f"加载完成但DataFrame为空: {filename} as self.data['{key}']")
            else:
                 print(f"警告: {filename} (应为 self.data['{key}']) 加载后键不存在于self.data中。")

    def _check_missing_data(self, df, param_name=None):
        """检查DataFrame中是否有缺失数据"""
        if df is None:
            print(f"错误: 数据为空")
            return True
            
        missing = df.isnull().sum().sum()
        if missing > 0:
            if param_name:
                print(f"警告: 参数 '{param_name}' 有 {missing} 个缺失值")
            else:
                print(f"警告: 数据中有 {missing} 个缺失值")
            return True
        return False
    
    # --- 通用查询方法 ---
    
    def get_value(self, dataset_key, param_name, filters=None, column=None, default=None):
        """
        从指定数据集中获取符合条件的特定参数值
        
        参数:
            dataset_key: 数据集键，如 'anthropometry', 'segment_inertia' 等
            param_name: 要查询的参数名称
            filters: 筛选条件字典，如 {'性别': '男', 'P50': None} 
            column: 要返回的列名，如 'P50'。如果为None，返回整行
            default: 如果找不到数据，返回的默认值
            
        返回:
            查询结果或默认值
        """
        if dataset_key not in self.data:
            print(f"错误: 未找到数据集键 {dataset_key}")
            return default
            
        df = self.data[dataset_key]
        
        # 构建筛选条件
        result = df.copy()
        if filters:
            for key, value in filters.items():
                if key in df.columns:
                    if value is None:
                        continue  # 跳过值为None的筛选条件
                    else:
                        # 使用布尔索引而不是query方法
                        result = result[result[key] == value]
        
        # 如果参数名不为None，添加参数名过滤条件
        if param_name is not None:
            # 确定包含参数名的列
            param_col = None
            potential_cols = ['参数', 'DataType', '体段', '肌肉', '参数类别']
            
            for col in potential_cols:
                if col in df.columns:
                    param_col = col
                    break
                    
            if param_col:
                result = result[result[param_col] == param_name]
            
        if result.empty:
            print(f"警告: 未找到符合条件的数据: {param_name}, 筛选条件: {filters}")
            return default
        
        # 如果指定了返回列，返回该列的值，否则返回整行
        if column and column in result.columns:
            return result[column].values[0]
        else:
            return result
    
    # --- 特定数据访问方法 ---
    
    def get_anthropometric_data(self, gender, percentile, param_name=None):
        """
        获取人体测量学数据
        
        参数:
            gender: '男' 或 '女'
            percentile: 'P1', 'P5', 'P10', 'P50', 'P90', 'P95', 'P99'
            param_name: 要查询的具体参数，如'身高', '体重'等。如果为None，返回所有符合条件的行
            
        返回:
            如果param_name为None，返回DataFrame；否则返回特定参数的值
        """
        df = self.data.get('anthropometry')
        if df is None:
            print("错误: 人体测量数据未加载")
            return None
            
        # 筛选性别和参数
        result = df[(df['性别'] == gender)]
        
        if param_name:
            result = result[result['参数'] == param_name]
            
            if result.empty:
                print(f"警告: 未找到参数 '{param_name}' 的数据")
                return None
                
            # 返回特定百分位数的值
            if percentile in result.columns:
                return result[percentile].values[0]
            else:
                print(f"警告: 未找到百分位数 '{percentile}'")
                return None
        else:
            # 返回所有符合条件的行
            return result
            
    def get_segment_inertial_params(self, segment_name=None):
        """
        获取肢段惯性参数
        
        参数:
            segment_name: 肢段名称，如'上臂', '前臂', '手'。如果为None，返回所有肢段数据
            
        返回:
            DataFrame或特定肢段的行
        """
        df = self.data.get('segment_inertia')
        if df is None:
            print("错误: 肢段惯性参数数据未加载")
            return None
            
        if segment_name:
            result = df[df['体段'] == segment_name]
            if result.empty:
                print(f"警告: 未找到肢段 '{segment_name}' 的惯性参数")
                return None
            return result.iloc[0].to_dict()  # 返回字典形式
        else:
            return df
            
    def get_joint_data(self, data_type=None, joint_name=None):
        """
        获取关节数据
        
        参数:
            data_type: 数据类型，如'ROM'(关节活动范围)或'TypicalPosture'(典型姿势)
            joint_name: 关节名称，如'肩关节前屈', '肘关节屈曲'等
            
        返回:
            符合条件的DataFrame或None
        """
        df = self.data.get('joint_data')
        if df is None:
            print("错误: 关节数据未加载")
            return None
            
        result = df.copy()
        if data_type:
            result = result[result['DataType'] == data_type]
        if joint_name:
            result = result[result['参数'] == joint_name]
            
        if result.empty:
            print(f"警告: 未找到符合条件的关节数据: DataType={data_type}, 参数={joint_name}")
            return None
                
        return result
            
    def get_typical_posture(self, posture_name):
        """
        获取典型姿势数据
        
        参数:
            posture_name: 姿势名称，如'水平推进_肩关节前屈'
            
        返回:
            姿势角度值或None
        """
        df = self.data.get('joint_data')
        if df is None:
            print("错误: 关节数据未加载")
            return None
            
        # 筛选出指定姿势
        result = df[(df['DataType'] == 'TypicalPosture') & (df['参数'] == posture_name)]
        
        if result.empty:
            print(f"警告: 未找到姿势 '{posture_name}' 的数据")
            return None
            
        # 从P50列获取角度值
        if 'P50' in result.columns:
            value_str = result['P50'].values[0]
            # 处理形如 "15±3" 的字符串，提取中心值
            if '±' in value_str:
                center_value = float(value_str.split('±')[0])
                return center_value
            else:
                try:
                    return float(value_str)
                except ValueError:
                    print(f"警告: 无法转换角度值 '{value_str}'")
                    return None
        else:
            print("警告: 数据中未找到 'P50' 列")
            return None
            
    def get_luggage_scenario_param(self, param_name):
        """
        获取行李箱场景参数
        
        参数:
            param_name: 参数名称，如'滚动摩擦系数（硬质地面）'
            
        返回:
            参数值或None
        """
        df = self.data.get('luggage_scenario')
        if df is None:
            print("错误: 行李箱场景参数数据未加载")
            return None
            
        # 查找参数
        result = df[df['参数'] == param_name]
        
        if result.empty:
            print(f"警告: 未找到参数 '{param_name}' 的数据")
            return None
            
        # 获取参数值
        value = result['数值/范围'].values[0]
        
        # 处理可能的范围值，如 "0.02–0.05"
        if isinstance(value, str) and '–' in value:
            # 将范围转换为数值，取中间值
            try:
                range_parts = value.split('–')
                min_val = float(range_parts[0])
                max_val = float(range_parts[1])
                return (min_val + max_val) / 2  # 返回范围中间值
            except ValueError:
                print(f"警告: 无法解析范围值 '{value}'")
                return value  # 返回原始字符串
        else:
            try:
                return float(value)  # 尝试转换为浮点数
            except (ValueError, TypeError):
                return value  # 保持原始值
                
    def get_mvc_data(self, gender, muscle_group=None, percentile_str=None):
        """
        获取最大随意收缩(MVC)数据
        
        参数:
            gender: '男' 或 '女'
            muscle_group: 肌肉群名称，如'最大握力'。如果为None，返回所有MVC数据
            percentile_str: 百分位字符串，如'P5', 'P50', 'P95'。如果为None，使用'P50'
            
        返回:
            MVC值(N)或DataFrame
        """
        df = self.data.get('functional_ergonomics')
        if df is None:
            print("错误: 功能人因参数数据未加载")
            return None
        
        # 如果未指定百分位，默认使用P50
        if percentile_str is None:
            percentile_str = 'P50'
        
        # 筛选最大握力数据
        result = df[df['ParameterType'] == 'FunctionalTest']
        
        if gender in ['男', '女']:
            if gender == '男':
                result = result[result['性别'].isin(['男', '男/女'])]
            elif gender == '女':
                result = result[result['性别'].isin(['女', '男/女'])]
            
        if result.empty:
            print(f"警告: 未找到符合条件的MVC数据")
            return None
            
        if muscle_group:
            # 查找特定肌肉群
            mvc_row = result[result['参数'].str.contains(muscle_group, case=False, na=False)]
            
            if mvc_row.empty:
                print(f"警告: 未找到肌肉群 '{muscle_group}' 的MVC数据")
                return None
                
            # 根据指定百分位返回值
            if percentile_str in mvc_row.columns:
                try:
                    return float(mvc_row[percentile_str].values[0])
                except (ValueError, TypeError):
                    print(f"警告: 无法将MVC值转换为浮点数，尝试使用P50")
                    if 'P50' in mvc_row.columns:
                        try:
                            return float(mvc_row['P50'].values[0])
                        except (ValueError, TypeError):
                            return None
                    return None
            else:
                print(f"警告: 未找到百分位 '{percentile_str}' 的MVC数据，使用P50")
                if 'P50' in mvc_row.columns:
                    try:
                        return float(mvc_row['P50'].values[0])
                    except (ValueError, TypeError):
                        return None
                return None
        else:
            return result
    
    def get_anthropometry_for_simulation(self, gender, percentile):
        """
        获取用于模拟的人体测量学数据，整合为综合计算.py所需的格式
        
        参数:
            gender: '男' 或 '女'
            percentile: 百分位数，如'P50'
            
        返回:
            包含人体测量数据的字典 (直接返回参数字典，而不是嵌套在user_id下)
        """
        if gender not in ['男', '女']:
            print(f"错误: 性别参数必须是 '男' 或 '女'，而不是 '{gender}'")
            return None
            
        valid_percentiles = ['P1', 'P5', 'P10', 'P50', 'P90', 'P95', 'P99']
        if percentile not in valid_percentiles:
            print(f"错误: 百分位数参数 '{percentile}' 无效。应为 {valid_percentiles}")
            return None
        
        body_weight = self.get_value('anthropometry', '体重', filters={'性别': gender}, column=percentile)
        if body_weight is None: print(f"错误: 无法获取 {gender} {percentile} 的体重数据"); return None
            
        body_height = self.get_value('anthropometry', '身高', filters={'性别': gender}, column=percentile)
        if body_height is None: print(f"错误: 无法获取 {gender} {percentile} 的身高数据"); return None
        
        upperarm_length = self.get_value('anthropometry', '上臂长', filters={'性别': gender}, column=percentile)
        if upperarm_length is None: print(f"错误: 无法获取 {gender} {percentile} 的上臂长数据"); return None
            
        forearm_length = self.get_value('anthropometry', '前臂长', filters={'性别': gender}, column=percentile)
        if forearm_length is None: print(f"错误: 无法获取 {gender} {percentile} 的前臂长数据"); return None
            
        hand_length = self.get_value('anthropometry', '手长', filters={'性别': gender}, column=percentile)
        if hand_length is None: print(f"错误: 无法获取 {gender} {percentile} 的手长数据"); return None

        segment_mass_perc = {}
        segment_com_perc = {}
        
        for segment_internal_name, display_name in [('upperarm', '上臂'), ('forearm', '前臂'), ('hand', '手')]:
            segment_data_dict = self.get_segment_inertial_params(display_name)
            
            if not segment_data_dict:
                print(f"错误: 无法获取 {display_name} 的惯性参数字典 (来自DataLoader内部方法)")
                return None
            
            try:
                # --- 修改开始: 直接使用CSV中定义的、清理后DataFrame中实际存在的列名 ---
                mass_perc_key = "质量百分比_体重_百分比"
                com_perc_key = "质心位置_肢段长_距近端_百分比"
                # --- 修改结束 ---

                mass_perc_val = segment_data_dict.get(mass_perc_key)
                if mass_perc_val is not None:
                    if isinstance(mass_perc_val, str) and '%' in mass_perc_val: # 假设原始数据带百分号
                        segment_mass_perc[segment_internal_name] = float(mass_perc_val.replace('%', '')) / 100
                    else: # 假设已经是小数形式或纯数字
                        try:
                            segment_mass_perc[segment_internal_name] = float(mass_perc_val)
                        except ValueError:
                            print(f"警告: {display_name} 的 {mass_perc_key} 值 '{mass_perc_val}' 无法转换为浮点数，使用默认值。")
                            default_mass_perc = {'hand': 0.006, 'forearm': 0.016, 'upperarm': 0.027}
                            segment_mass_perc[segment_internal_name] = default_mass_perc.get(segment_internal_name, 0)
                else:
                    print(f"警告: {display_name} 缺少 '{mass_perc_key}' 数据，使用默认值。")
                    default_mass_perc = {'hand': 0.006, 'forearm': 0.016, 'upperarm': 0.027}
                    segment_mass_perc[segment_internal_name] = default_mass_perc.get(segment_internal_name, 0)

                com_perc_val = segment_data_dict.get(com_perc_key)
                if com_perc_val is not None:
                    if isinstance(com_perc_val, str) and '%' in com_perc_val: # 假设原始数据带百分号
                        segment_com_perc[segment_internal_name] = float(com_perc_val.replace('%', '')) / 100
                    else: # 假设已经是小数形式或纯数字
                        try:
                            segment_com_perc[segment_internal_name] = float(com_perc_val)
                        except ValueError:
                            print(f"警告: {display_name} 的 {com_perc_key} 值 '{com_perc_val}' 无法转换为浮点数，使用默认值。")
                            default_com_perc = {'hand': 0.500, 'forearm': 0.430, 'upperarm': 0.436}
                            segment_com_perc[segment_internal_name] = default_com_perc.get(segment_internal_name, 0.5)
                else:
                    print(f"警告: {display_name} 缺少 '{com_perc_key}' 数据，使用默认值。")
                    default_com_perc = {'hand': 0.500, 'forearm': 0.430, 'upperarm': 0.436}
                    segment_com_perc[segment_internal_name] = default_com_perc.get(segment_internal_name, 0.5)
            except Exception as e:
                 print(f"错误: 处理 {display_name} 的惯性参数字典时发生错误: {e}")
                 traceback.print_exc()
                 return None
        
        mvc_grip = self.get_mvc_data(gender, '最大握力', percentile)
        if mvc_grip is None:
            print(f"警告: 无法获取 {gender} {percentile} 的握力MVC数据，使用默认值")
            mvc_grip = 480 if gender == '男' else 330
        
        return {
            "total_weight_kg": float(body_weight),
            "L_hand_m": float(hand_length) / 1000,
            "L_forearm_m": float(forearm_length) / 1000,
            "L_upperarm_m": float(upperarm_length) / 1000,
            "mass_perc_hand": segment_mass_perc.get('hand', 0.006),
            "mass_perc_forearm": segment_mass_perc.get('forearm', 0.016),
            "mass_perc_upperarm": segment_mass_perc.get('upperarm', 0.027),
            "com_perc_hand_proximal": segment_com_perc.get('hand', 0.500),
            "com_perc_forearm_proximal": segment_com_perc.get('forearm', 0.430),
            "com_perc_upperarm_proximal": segment_com_perc.get('upperarm', 0.436),
            "mvc_grip_N": float(mvc_grip),
            "gender": gender,
            "percentile": percentile
        }

    def get_operation_scenario_params(self, scenario_type="机场平路"):
        """
        获取特定场景的操作参数
        
        参数:
            scenario_type: 场景类型，如 "机场平路", "斜坡", "粗糙地面" 等
            
        返回:
            包含场景参数的字典
        """
        scenario_params = {}
        
        # 1. 根据场景类型设置默认参数
        if scenario_type == "机场平路":
            scenario_params = {
                "slope_deg": 0,
                "rolling_friction_coeff": 0.02,
                "acceleration_h_m_s2": 0.5,
                "c_pull_push_factor": 0.3,
                "operation_type": "pulling",
                "k_impact_factor": 0,
                "k_vibration_factor": 0
            }
        elif scenario_type == "斜坡":
            scenario_params = {
                "slope_deg": 5,  # 典型值
                "rolling_friction_coeff": 0.03,
                "acceleration_h_m_s2": 0.6,
                "c_pull_push_factor": 0.3,
                "operation_type": "pulling",
                "k_impact_factor": 0,
                "k_vibration_factor": 0
            }
        elif scenario_type == "粗糙地面":
            scenario_params = {
                "slope_deg": 0,
                "rolling_friction_coeff": 0.04,
                "acceleration_h_m_s2": 0.5,
                "c_pull_push_factor": 0.3,
                "operation_type": "pulling",
                "k_impact_factor": 0,
                "k_vibration_factor": 0.1
            }
        elif scenario_type == "侧向提拉":
            scenario_params = {
                "slope_deg": 0,
                "rolling_friction_coeff": 0.02,
                "acceleration_h_m_s2": 0.4,
                "c_pull_push_factor": 0.2,
                "operation_type": "lateral",
                "k_impact_factor": 0,
                "k_vibration_factor": 0
            }
        else:
            print(f"警告: 未定义场景类型 '{scenario_type}'，使用默认参数")
            scenario_params = {
                "slope_deg": 0,
                "rolling_friction_coeff": 0.02,
                "acceleration_h_m_s2": 0.5,
                "c_pull_push_factor": 0.3,
                "operation_type": "pulling",
                "k_impact_factor": 0,
                "k_vibration_factor": 0
            }
        
        # 2. 从数据库获取并覆盖默认参数
        try:
            # 坡度
            if scenario_type == "斜坡":
                slope_range = self.get_value('luggage_scenario', '坡度角度', column='数值/范围')
                if slope_range and isinstance(slope_range, str) and '≤' in slope_range:
                    max_slope = float(slope_range.replace('≤', ''))
                    scenario_params["slope_deg"] = max_slope
            
            # 摩擦系数
            friction_param = '滚动摩擦系数（硬质地面）'
            if scenario_type == "粗糙地面":
                friction_param = '滚动摩擦系数（地毯）'
                
            friction_value = self.get_value('luggage_scenario', friction_param, column='数值/范围')
            if friction_value is not None:
                # 处理可能的范围值 "0.02–0.05"
                if isinstance(friction_value, str) and '–' in friction_value:
                    values = friction_value.split('–')
                    min_val = float(values[0])
                    max_val = float(values[1])
                    scenario_params["rolling_friction_coeff"] = (min_val + max_val) / 2
                else:
                    try:
                        scenario_params["rolling_friction_coeff"] = float(friction_value)
                    except (ValueError, TypeError):
                        pass
            
            # 拉行/推动系数
            pull_push_param = '拉行垂直比例因子'
            if "push" in scenario_params["operation_type"]:
                pull_push_param = '推动垂直比例因子'
                
            pull_push_value = self.get_value('luggage_scenario', pull_push_param, column='数值/范围') 
            if pull_push_value is not None:
                try:
                    scenario_params["c_pull_push_factor"] = float(pull_push_value)
                except (ValueError, TypeError):
                    pass
            
            # 振动因子
            vibration_value = self.get_value('luggage_scenario', '地形振动调整因子', column='数值/范围')
            if vibration_value is not None and isinstance(vibration_value, str):
                # 处理如 "0（机场） / 0.1（街道） / 0.2（粗糙）" 的字符串
                if "机场" in vibration_value and scenario_type == "机场平路":
                    parts = vibration_value.split('/')
                    for part in parts:
                        if "机场" in part:
                            value = part.split('（')[0].strip()
                            scenario_params["k_vibration_factor"] = float(value)
                elif "街道" in vibration_value and scenario_type in ["斜坡", "城市街道"]:
                    parts = vibration_value.split('/')
                    for part in parts:
                        if "街道" in part:
                            value = part.split('（')[0].strip()
                            scenario_params["k_vibration_factor"] = float(value)
                elif "粗糙" in vibration_value and scenario_type == "粗糙地面":
                    parts = vibration_value.split('/')
                    for part in parts:
                        if "粗糙" in part:
                            value = part.split('（')[0].strip()
                            scenario_params["k_vibration_factor"] = float(value)
        
        except Exception as e:
            print(f"警告: 加载场景参数时出错: {e}")
        
        return scenario_params

    def get_joint_angles_for_scenario(self, scenario_type="机场平路"):
        """
        获取特定场景的关节角度
        
        参数:
            scenario_type: 场景类型，如 "机场平路", "斜坡拉行", "侧向提拉" 等
            
        返回:
            包含关节角度的字典
        """
        joint_angles = {}
        scenario_mapping = {
            "机场平路": "水平推进",
            "斜坡": "斜坡拉行",
            "粗糙地面": "斜坡拉行",  # 使用相同的姿势
            "侧向提拉": "侧向提拉",
            "越障动作": "越障动作",
            "转向操控": "转向操控"
        }
        
        scenario_prefix = scenario_mapping.get(scenario_type, "水平推进")
        
        try:
            # 根据场景获取典型姿势数据
            df = self.data.get('joint_data')
            if df is None:
                print("错误: 关节数据未加载")
                return {"shoulder_flex": -8, "elbow_flex": 90, "wrist_flex": 0}  # 默认值
                
            # 筛选出该场景的所有姿势
            posture_data = df[(df['DataType'] == 'TypicalPosture') & 
                             (df['参数'].str.startswith(scenario_prefix))]
            
            if posture_data.empty:
                print(f"警告: 未找到场景 '{scenario_type}' 的姿势数据")
                return {"shoulder_flex": -8, "elbow_flex": 90, "wrist_flex": 0}  # 默认值
            
            # 提取关节角度
            for _, row in posture_data.iterrows():
                posture_name = row['参数']
                angle_value = row['P50']
                
                # 处理形如 "15±3" 的字符串，提取中心值
                if isinstance(angle_value, str) and '±' in angle_value:
                    center_value = float(angle_value.split('±')[0])
                else:
                    try:
                        center_value = float(angle_value)
                    except (ValueError, TypeError):
                        continue
                
                # 根据姿势名称映射到关节参数
                if "肩关节前屈" in posture_name:
                    joint_angles["shoulder_flex"] = center_value
                elif "肩关节后伸" in posture_name:
                    joint_angles["shoulder_flex"] = -center_value  # 注意这里用负值表示后伸
                elif "肩关节外展" in posture_name:
                    joint_angles["shoulder_abduction"] = center_value
                elif "肘关节屈曲" in posture_name:
                    joint_angles["elbow_flex"] = center_value
                elif "肘关节伸展" in posture_name:
                    joint_angles["elbow_flex"] = 180 - center_value  # 转换为屈曲角度
                elif "腕关节掌屈" in posture_name:
                    joint_angles["wrist_flex"] = center_value
                elif "腕关节背屈" in posture_name:
                    joint_angles["wrist_flex"] = -center_value  # 注意这里用负值表示背屈
                elif "腕关节中立位" in posture_name:
                    joint_angles["wrist_flex"] = 0
                elif "膝关节屈曲" in posture_name:
                    joint_angles["knee_flex"] = center_value
            
            # 如果没有找到某些关节的角度，设置默认值
            if "shoulder_flex" not in joint_angles:
                if scenario_type in ["斜坡", "粗糙地面"]:
                    joint_angles["shoulder_flex"] = -8  # 后伸
                else:
                    joint_angles["shoulder_flex"] = 15  # 前屈
                    
            if "elbow_flex" not in joint_angles:
                if scenario_type in ["斜坡", "粗糙地面", "侧向提拉"]:
                    joint_angles["elbow_flex"] = 90
                else:
                    joint_angles["elbow_flex"] = 20  # 接近伸直
                    
            if "wrist_flex" not in joint_angles:
                joint_angles["wrist_flex"] = 0
        
        except Exception as e:
            print(f"警告: 获取关节角度时出错: {e}")
            joint_angles = {"shoulder_flex": -8, "elbow_flex": 90, "wrist_flex": 0}
            
        return joint_angles

# 示例用法
if __name__ == "__main__":
    # 创建数据加载器实例
    loader = DataLoader()
    
    # 示例1: 获取男性P50体重
    weight = loader.get_value('anthropometry', '体重', filters={'性别': '男'}, column='P50')
    print(f"男性P50体重: {weight} kg")
    
    # 示例2: 获取用于模拟的人体测量数据
    male_p50_data = loader.get_anthropometry_for_simulation('男', 'P50')
    print("\n男性P50人体测量数据:")
    for key, value in male_p50_data.items():
        print(f"  {key}: {value}")
    
    # 示例3: 获取场景参数
    scenario_params = loader.get_operation_scenario_params("机场平路")
    print("\n机场平路场景参数:")
    for key, value in scenario_params.items():
        print(f"  {key}: {value}")
    
    # 示例4: 获取关节角度
    joint_angles = loader.get_joint_angles_for_scenario("机场平路")
    print("\n机场平路关节角度:")
    for key, value in joint_angles.items():
        print(f"  {key}: {value}°") 