 

import bs4
import requests


def fetch_page(url: str) -> tuple[str, str]:
    """
    抓取页面
    :param url: 统一资源定位符
    :return: 网页标题和主体内容构成的二元组
    """
    resp = requests.get(
        url=url,
        headers={
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
        },
    )
    if resp.status_code == 200:
        soup = bs4.BeautifulSoup(resp.text, 'html.parser')
        # 移除body标签下不包含文本内容的标签
        for elem in soup.body(['script', 'link', 'nav', 'header', 'footer', 'form', 'input', 'button',
                               'img', 'audio', 'video', 'area', 'canvas', 'map', 'object', 'embed']):
            elem.decompose()
        # 获取网页标题和主体内容返回二元组
        return soup.title.text, soup.body.get_text(separator='\n', strip=True)
    return '', ''


 
def get_llm_response(client,
                     *, 
                     system_prompt='',
                     user_prompt='',
                     model='agnes-2.5-flash',
                     temperature=0.2,
                     frequency_penalty=0,
                     max_tokens=1024,
                     thinking=False,
                     stream=False):
    """
    获取大模型响应
    :param client: OpenAI对象
    :param system_prompt: 系统提示词
    :param user_prompt: 用户提示词
    :param model: 模型名称
    :param temperature: 温度参数
    :param frequency_penalty: 频率惩罚参数
    :param max_tokens: 最大token数量
    :param thinking: 是否开启思考模式
    :param stream: 是否开启流模型
    :return: 大模型的响应内容或Stream对象
    """
    messages = []

    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if user_prompt:
        messages.append({'role': 'user', 'content': user_prompt})
        
    config = {
        'model': model,
        'temperature': temperature,
        'frequency_penalty': frequency_penalty,
        'max_tokens': max_tokens,
        'messages': messages,
        'stream': stream,
    }
    if not thinking:
        config['extra_body'] = {'thinking': {'type': 'disabled'}}

    resp = client.chat.completions.create(**config)
    if not stream:
        return resp.choices[0].message.content
    return resp