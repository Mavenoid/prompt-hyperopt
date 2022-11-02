import functools
import gc
from typing import List, Optional

import transformers
import openai


# @TODO are more device control


gpt_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
ENDOFANSWER = "<|answer|>"  # @TODO use rare tokens?
# @TODO check that these are complete
OPENAI_API_ENGINE_NAMES = [
    "ada",
    "babbage",
    "curie",
    "curie-instruct-beta",
    "davinci",
    "davinci-instruct-beta",
    "text-ada-001",
    "text-babbage-001",
    "text-curie-001",
    "text-davinci-001",
    "text-davinci-002",
    "code-cushman-001",
    "code-davinci-001",
    "code-davinci-002",
]


@functools.lru_cache(128)
def get_hf_gpt_model(
    model_name: str, device: str = "cpu"
) -> transformers.AutoModelForCausalLM:
    """Utility method to load Hugging Face GPT models"""
    gc.collect()
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    if device.startswith("cuda"):
        model.half()
    model.to(device)
    return model


@functools.lru_cache(512000)
def get_hf_gpt_logprobs(
    engine: str, prompt: str, device: str = "cpu"
) -> List[Optional[float]]:
    """Utility method to get logprobs generated via Hugging Face GPT models"""
    gpt_model = get_hf_gpt_model(engine, device=device)
    tokenization = gpt_tokenizer(prompt, return_tensors="pt").to(gpt_model.device)
    # @TODO
    # return [None] + gpt_model(**tokenization).logits.max(axis=-1).values.tolist()[0]
    res = gpt_model(**tokenization).logits.max(axis=-1).values.tolist()[0]
    assert len(res) == len(gpt_tokenizer.encode(prompt))
    return gpt_model(**tokenization).logits.max(axis=-1).values.tolist()[0]


@functools.lru_cache(512000)
def get_openai_logprobs(engine: str, prompt: str) -> List[Optional[float]]:
    """Utility method to get logprobs generated via the OpenAI API"""
    resp = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1,
        echo=True,
        logprobs=0,
    )
    return resp["choices"][0]["logprobs"]["token_logprobs"]


def get_model_logprobs(
    engine: str, prompt: str, device: str = "cpu"
) -> List[Optional[float]]:
    """Utility method to get logprobs for various GPT models"""
    if engine in OPENAI_API_ENGINE_NAMES:
        return get_openai_logprobs(engine=engine, prompt=prompt)
    return get_hf_gpt_logprobs(engine=engine, prompt=prompt, device=device)
