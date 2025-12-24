#  api_key = 'sk-a929ba61bc01427c87dbfbff0c4f8953',

from openai import OpenAI
import os
import base64


# 编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 将xxxx/test.mp4替换为你本地视频的绝对路径
base64_video = encode_video("/home/liaoyizhi/codes/ncclab_t/data/CineBrain_8s/000002_000003.mp4")
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key='sk-a929ba61bc01427c87dbfbff0c4f8953',
    # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

prompt_txt = '''
你是一位情感计算与情绪理解领域的专家。
请仔细观看视频，并描述该场景中表达出的情绪。
请重点关注可观察到的情绪线索，包括但不限于：
面部表情
身体语言与姿态
动作与移动节奏
场景氛围与整体基调
光线、色彩与视觉情绪
角色之间或角色与环境的互动方式

请以自然、细致、开放的方式描述视频所传达的情绪状态。
情绪可能是混合的、微妙的、随时间变化的，或存在一定模糊性。

不要给出数值评分。
不要局限于预设的情绪类别。

请输出一段连贯的文字，用一个完整段落描述视频所呈现的情绪。
'''
completion = client.chat.completions.create(
    model="qwen3-vl-plus",  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    # 直接传入视频文件时，请将type的值设置为video_url
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                },
                {"type": "text", "text": f'{prompt_txt}'},
            ],
        }
    ],
)
print(completion.choices[0].message.content)



# 你是一位情感计算与情绪理解领域的专家。
# 请仔细观看视频，并描述该场景中表达出的情绪。
# 请重点关注可观察到的情绪线索，包括但不限于：
# 面部表情
# 身体语言与姿态
# 动作与移动节奏
# 场景氛围与整体基调
# 光线、色彩与视觉情绪
# 角色之间或角色与环境的互动方式

# 请以自然、细致、开放的方式描述视频所传达的情绪状态。
# 情绪可能是混合的、微妙的、随时间变化的，或存在一定模糊性。

# 不要给出数值评分。
# 不要局限于预设的情绪类别。

# 请输出一段连贯的文字，用一个完整段落描述视频所呈现的情绪。