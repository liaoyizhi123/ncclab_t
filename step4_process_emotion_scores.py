import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端以支持无显示
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
from pathlib import Path


# 配置路径
BASE_DIR = str(Path(__file__).resolve().parent)
DATA_DIR = os.path.join(BASE_DIR, "results", "prompt_v2")
RAW_SCORES_DIR = os.path.join(BASE_DIR, "results", "raw_scores")
SMOOTH_SCORES_DIR = os.path.join(BASE_DIR, "results", "smooth_scores")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

# 确保输出目录存在
os.makedirs(RAW_SCORES_DIR, exist_ok=True)
os.makedirs(SMOOTH_SCORES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def parse_response_content(content):
    """
    解析API返回的response字符串，提取每秒的6维情绪评分。
    返回: list of [高兴, 惊讶, 悲伤, 愤怒, 厌恶, 恐惧]
    """
    lines = content.strip().split('\n')
    scores_list = []
    emotions = ["高兴", "惊讶", "悲伤", "愤怒", "厌恶", "恐惧"]
    assert len(lines) == 8
    for line in lines:
        line = line.strip()
        if not line or "第" not in line:
            continue
            
        current_scores = []
        for emotion in emotions:
            # 查找形如 "高兴: 0.5" 的模式
            # 使用简单的字符串查找或正则
            try:
                # 寻找 emotion 后面的冒号和数字
                # 格式可能是 "高兴: 0.5;" 或 "高兴: 0.5"
                pattern = rf"{emotion}:\s*(\[?[\d\.]+\]?)"
                match = re.search(pattern, line)
                if match:
                    val_str = match.group(1)
                    # 处理可能存在的方括号
                    val_str = val_str.replace('[', '').replace(']', '')
                    score = float(val_str)
                else:
                    score = 0.0 # 默认值，或者抛出警告
                    print(f"Warning: Could not find score for {emotion} in line: {line}")
                current_scores.append(score)
            except Exception as e:
                print(f"Error parsing {emotion} in line: {line}. Error: {e}")
                current_scores.append(0.0)
        
        if len(current_scores) == 6:
            scores_list.append(current_scores)
            
    return np.array(scores_list)

def read_and_merge_scores(data_dir, video_id):
    """
    读取并合并指定视频ID的所有评分文件
    """
    # files = [f for f in os.listdir(data_dir) if f.startswith(video_id) and f.endswith('.json')]
    files = sorted(from_episode_to_clip(video_id))

    # 按照起始时间排序
    # 文件名格式假设: neg_s_2_0_10_mode1.json
    # 提取第4个部分作为起始时间 (index 3)
    # def get_start_time(filename):
    #     parts = filename.split('_')
    #     # neg_s_2_0_10_mode1.json -> parts: ['neg', 's', '2', '0', '10', 'mode1.json']
    #     # start time is at index 3
    #     try:
    #         return int(parts[3])
    #     except:
    #         return 0
            
    # files.sort(key=get_start_time)
    
    all_scores = []
    
    print(f"Found {len(files)} files for video {video_id}")
    
    for filename in files:
        file_path = os.path.join(data_dir, filename+"_mode_1.json")
        # print(f"Processing {filename}...")
        assert os.path.exists(file_path), f"File not found: {file_path}"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        response_text = data.get('response', '')
        if not response_text:
            continue
            
        scores = parse_response_content(response_text)
        assert scores.size>0
        if scores.size > 0:
            all_scores.append(scores)
            
    if not all_scores:
        return np.array([]), np.array([])
        
    # 垂直堆叠
    merged_scores = np.vstack(all_scores)
    # 生成时间轴 (假设每行代表1秒)
    raw_times = np.arange(merged_scores.shape[0])
    
    return raw_times, merged_scores

def smooth_scores_window(raw_times, raw_scores, window_size=10, step_size=2):
    """
    使用中心滑动窗口平滑数据
    window_size: 窗口总大小 (前后各 window_size/2)
    step_size: 步长
    """
    if raw_scores.size == 0:
        return np.array([]), np.array([])

    half_win = window_size / 2
    max_time = raw_times[-1]
    
    target_times = np.arange(0, max_time + 1, step_size)
    smoothed_scores = []
    
    for t in target_times:
        # 定义窗口范围 [t - half_win, t + half_win]
        # 注意: 这里包含边界
        mask = (raw_times >= (t - half_win)) & (raw_times <= (t + half_win))
        
        # 提取窗口内的分数
        scores_in_window = raw_scores[mask]
        
        if scores_in_window.size > 0:
            # 计算均值
            mean_scores = np.mean(scores_in_window, axis=0)
        else:
            # 如果窗口内没有数据（理论上不应该发生，除非数据缺失），沿用上一个或全0
            mean_scores = np.zeros(6)
            
        smoothed_scores.append(mean_scores)
        
    return target_times, np.array(smoothed_scores)

def save_scores(output_dir, video_id, times, scores, suffix):
    """
    保存分数为CSV文件
    """
    if scores.size == 0:
        print("No data to save.")
        return
        
    df = pd.DataFrame(scores, columns=["Happy", "Surprise", "Sad", "Anger", "Disgust", "Fear"])
    df.insert(0, "Time", times)
    
    output_path = os.path.join(output_dir, f"{video_id}_{suffix}.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {suffix} scores to: {output_path}")

def plot_scores(times, scores, video_id):
    """
    绘图并保存
    """
    if scores.size == 0:
        return

    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    emos = ["Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear"]
    
    plt.figure(figsize=(15, 12))
    
    for j in range(6):
        plt.subplot(6, 1, j+1)
        plt.plot(times, scores[:, j], color=colors[j], linewidth=1.5)
        plt.xlim([0, max(times) if len(times) > 0 else 10])
        plt.ylim([0, 7])
        plt.ylabel('Scores')
        plt.xlabel('Time/s')
        plt.title(emos[j])
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f"{video_id}_smoothed_plot.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")
    # plt.show() # 在非交互式环境中不需要

def main():
    episodes_li = [_.stem for _ in Path('/home/liaoyizhi/codes/ncclab_t/Raw_danmu').glob("*.csv")]
    
    for episode in episodes_li:
        print(f"Start processing video: {episode}")

        # 1. 读取并合并
        raw_times, raw_scores = read_and_merge_scores(DATA_DIR, episode)
        print(f"Merged raw data shape: {raw_scores.shape}")
        
        if raw_scores.size == 0:
            print("Error: No data found.")
            return

        # 2. 保存原始数据
        save_scores(RAW_SCORES_DIR, episode, raw_times, raw_scores, "raw")
        
        # 3. 平滑处理
        # 时间窗8秒，平移2秒
        smooth_times, smooth_data = smooth_scores_window(raw_times, raw_scores, window_size=8, step_size=2)
        print(f"Smoothed data shape: {smooth_data.shape}")
        
        # 4. 保存平滑数据
        save_scores(SMOOTH_SCORES_DIR, episode, smooth_times, smooth_data, "smooth")
        
        # 5. 绘图
        plot_scores(smooth_times, smooth_data, episode)
        
        print("Done.")


def from_clip_to_episode(clip_name):
    # 通过clip名称映射回episode名称
    parts = clip_name.split('_')[1]
    episodes_list = [
        '000000_000001-000268_000269',
        '000270_000271-000538_000539',
        '000540_000541-000808_000809',
        '000810_000811-001078_001079',
        '001080_001081-001348_001349',
        '001350_001351-001618_001619',
        '001620_001621-001888_001889',
        '001890_001891-002158_002159',
        '002160_002161-002428_002429',
        '002430_002431-002698_002699',
        '002700_002701-002968_002969',
        '002970_002971-003238_003239',
        '003240_003241-003508_003509',
        '003510_003511-003778_003779',
        '003780_003781-004048_004049',
        '004050_004051-004318_004319',
        '004320_004321-004588_004589',
        '004590_004591-004858_004859',
        '004860_004861-005128_005129',
        '005130_005131-005398_005399',
        '005400_005401-005668_005669',
        '005670_005671-005938_005939',
        '005940_005941-006208_006209',
        '006210_006211-006478_006479',
        '006480_006481-006748_006749',
        '006750_006751-007018_007019',
        '007020_007021-007288_007289',
        '007290_007291-007558_007559',
        '007560_007561-007828_007829',
        '007830_007831-008098_008099',
    ]
    for ep in episodes_list:
        start_clip = ep.split('_')[0]
        end_clip = ep.split('_')[2]
        if int(start_clip) <= int(parts) <= int(end_clip):
            return ep
        

def from_episode_to_clip(episode_name):
    start_clip = episode_name.split('_')[0]
    end_clip = episode_name.split('_')[2]
    return [f'{i:06d}_{i+1:06d}' for i in range(int(start_clip), int(end_clip)+1, 2)]


if __name__ == "__main__":
    main()
