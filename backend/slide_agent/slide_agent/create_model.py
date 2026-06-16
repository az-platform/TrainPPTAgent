#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/6/18 14:44
# @File  : create_model.py.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : little llm 不要设置timeout，超过一定时间会断
import os
import re
import litellm
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
# 关闭LLM的debug模式（本地模型无需费用计算，开启会产生大量日志）
# litellm._turn_on_debug()

load_dotenv()

# ============================================================================
# new-api 等兼容网关的 Claude 工具名还原
#
# 某些 OpenAI/Anthropic 兼容网关(如 new-api)在【多工具】请求时,会把 Claude 的工具名
# 改写成 "Compat" + 帕斯卡名 + 6 位十六进制(例: DocumentSearch -> CompatDocumentSearchb4efcd),
# 而且响应里也不还原。ADK 按 function_call.name 精确查 tools_dict 来派发工具,改名后会抛
# "Function ... is not found in the tools_dict." 导致 agent 中断。
# 下面这个 LiteLlm 子类在 LLM 响应返回后、派发前,把工具名还原回注册原名;
# 对未被改名的响应是空操作(只匹配 Compat 前缀)。
# ============================================================================
_COMPAT_TOOL_RE = re.compile(r"^Compat(?P<body>.+?)(?P<hash>[0-9a-f]{6})$")


def _pascalize(name: str) -> str:
    """snake_case -> 帕斯卡(与网关改写规则一致): my_tool_2 -> MyTool2; DocumentSearch -> DocumentSearch"""
    return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)


def unmangle_function_calls(llm_response, reg_names):
    """把兼容网关改写过的工具名(Compat<名><6hex>)就地还原回注册原名。"""
    rev = {_pascalize(n): n for n in reg_names}  # 帕斯卡名 -> 原名
    content = getattr(llm_response, "content", None)
    if content and getattr(content, "parts", None):
        for part in content.parts:
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", None):
                m = _COMPAT_TOOL_RE.match(fc.name)
                if m and m.group("body") in rev:
                    fc.name = rev[m.group("body")]
    return llm_response


class CompatLiteLlm(LiteLlm):
    """LiteLlm 子类:还原兼容网关给工具名加的 Compat... 前缀。"""

    async def generate_content_async(self, llm_request, stream=False):
        # 收集本次请求注册的工具原名(优先 tools_dict,回退 config.tools)
        reg_names = list((llm_request.tools_dict or {}).keys())
        if not reg_names and getattr(llm_request, "config", None) and getattr(llm_request.config, "tools", None):
            for t in llm_request.config.tools:
                for fd in getattr(t, "function_declarations", None) or []:
                    if getattr(fd, "name", None):
                        reg_names.append(fd.name)
        async for resp in LiteLlm.generate_content_async(self, llm_request, stream=stream):
            unmangle_function_calls(resp, reg_names)
            yield resp


def create_model(model:str, provider: str):
    """
    创建模型，返回字符串或者LiteLlm
    LiteLlm(model="deepseek/deepseek-chat", api_key="xxx", api_base="")
    :return:
    """
    if provider == "google":
        # google的模型直接使用名称
        assert os.environ.get("GOOGLE_API_KEY"), "GOOGLE_API_KEY is not set"
        return model
    elif provider == "claude":
        # Claude 模型需要使用 LiteLlm，并遵循 LiteLLM 的模型命名规范
        assert os.environ.get("CLAUDE_API_KEY"), "CLAUDE_API_KEY is not set"
        # 正确的做法是使用 "anthropic/" 前缀
        if not model.startswith("anthropic/"):
            model = "anthropic/" + model

        return LiteLlm(
            model=model,  # 例如: "anthropic/claude-3-opus-20240229"
            api_key=os.environ.get("CLAUDE_API_KEY"),
            num_retries=3,
        )
    elif provider == "openai":
        # openai的模型需要使用LiteLlm
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("OPENAI_API_KEY"), api_base="https://api.openai.com/v1", num_retries=3)
    elif provider == "deepseek":
        # deepseek的模型需要使用LiteLlm
        assert os.environ.get("DEEPSEEK_API_KEY"),  "DEEPSEEK_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("DEEPSEEK_API_KEY"), api_base="https://api.deepseek.com/v1",num_retries=3)
    elif provider == "glm":
        # GLM的模型需要使用LiteLlm
        assert os.environ.get("GLM_API_KEY"),  "GLM_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("GLM_API_KEY"), api_base="https://open.bigmodel.cn/api/paas/v4",num_retries=3)
    elif provider == "local_google":
        assert os.environ.get("GOOGLE_API_KEY"),  "GOOGLE_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("GOOGLE_API_KEY"), api_base="http://localhost:6688",num_retries=3)
    elif provider == "local_deepseek":
        # deepseek的模型需要使用LiteLlm
        assert os.environ.get("DEEPSEEK_API_KEY"),  "DEEPSEEK_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("DEEPSEEK_API_KEY"), api_base="http://localhost:6688",num_retries=3)
    elif provider == "ali":
        # huggingface的模型需要使用LiteLlm
        assert os.environ.get("ALI_API_KEY"), "ALI_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("ALI_API_KEY"), api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",num_retries=3)
    elif provider == "local_ali":
        assert os.environ.get("ALI_API_KEY"), "ALI_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("ALI_API_KEY"), api_base="http://localhost:6688",num_retries=3)
    elif provider == "doubao":
        # huggingface的模型需要使用LiteLlm
        assert os.environ.get("DOUBAO_API_KEY"), "DOUBAO_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("DOUBAO_API_KEY"), api_base="https://ark.cn-beijing.volces.com/api/v3",num_retries=3)
    elif provider == "kimi":
        # huggingface的模型需要使用LiteLlm
        assert os.environ.get("KIMI_API_KEY"), "KIMI_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("KIMI_API_KEY"), api_base="https://api.moonshot.cn/v1",num_retries=3)
    elif provider == "vllm":
        # huggingface的模型需要使用LiteLlm
        assert os.environ.get("VLLM_API_KEY"), "VLLM_API_KEY is not set"
        assert os.environ.get("VLLM_API_URL"), "VLLM_API_URL is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("VLLM_API_KEY"), api_base=os.environ.get("VLLM_API_URL"),num_retries=3)
    elif provider == "silicon":
        # huggingface的模型需要使用LiteLlm
        assert os.environ.get("SILICON_API_KEY"), "SILICON_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("SILICON_API_KEY"), api_base="https://api.siliconflow.cn/v1",num_retries=3)
    elif provider == "modelscope":
        # modelscope的模型需要使用LiteLlm
        assert os.environ.get("MODELSCOPE_API_KEY"), "MODEL_SCOPE_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("MODELSCOPE_API_KEY"),
                       api_base="https://api-inference.modelscope.cn/v1")
    elif provider == "ollama":
        # huggingface的模型需要使用LiteLlm
        assert os.environ.get("OLLAMA_API_KEY"), "OLLAMA_API_KEY is not set"
        assert os.environ.get("OLLAMA_API_URL"), "OLLAMA_API_URL is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("OLLAMA_API_KEY"), api_base=os.environ.get("OLLAMA_API_URL"),num_retries=3)
    elif provider == "local_openai":
        assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set"
        if not model.startswith("openai/"):
            # 表示兼容openai的模型请求
            model = "openai/" + model
        return LiteLlm(model=model, api_key=os.environ.get("OPENAI_API_KEY"), api_base="http://localhost:6688",num_retries=3)
    elif provider == "local":
        assert os.environ.get("LOCAL_API_KEY"), "LOCAL_API_KEY is not set"
        assert os.environ.get("LOCAL_API_URL"), "LOCAL_API_URL is not set"
        api_key = os.environ.get("LOCAL_API_KEY")
        api_base = os.environ.get("LOCAL_API_URL")
        # Claude 必须走 Anthropic 原生协议:兼容网关(new-api 等)不做 OpenAI→Anthropic 的
        # 工具格式转换,type:function 工具透传给 Anthropic 会 400 "Input tag 'function' ..."。
        # 用 anthropic/ 前缀让 LiteLLM 把 type:function 自动转成 type:custom。
        if model.lower().startswith("claude"):
            if not model.startswith("anthropic/"):
                model = "anthropic/" + model
            # LiteLLM 的 anthropic provider 会自动在 api_base 后追加 /v1/messages,
            # 因此必须去掉结尾的 /v1,避免拼成 /v1/v1/messages (404)。
            api_base = api_base.rstrip("/").removesuffix("/v1")
            # 用 CompatLiteLlm 还原网关给工具名加的 "Compat..." 前缀,保证工具派发成功
            return CompatLiteLlm(model=model, api_key=api_key, api_base=api_base, num_retries=3)
        if not model.startswith("openai/"):
            model = "openai/" + model
        return LiteLlm(model=model, api_key=api_key, api_base=api_base, num_retries=3)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
