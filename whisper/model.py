import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass 
class ModelDimensions:
    """
    这个装饰器的作用主要是定义一个数据类，用来存储数据。它会自动生成__init__,__repr__,和__eq__函数。
    """
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    """
    层的正则化
    输入为一个Tensor
    输出.type(x.dtype)保证输出也是一个Tensor
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """
    L = w*x + b
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    """
    一维卷积
    """

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0  # 确保channels为偶数
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)  # 对数
    inv_timescales = torch.exp(
        -log_timescale_increment * torch.arange(channels // 2)
    )  # 指数
    scaled_time = (
        torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    )  # np.newaxis添加新的一维
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        """
        生成QKV三个矩阵
        """
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        """
        查询张量 q，键张量 k，值张量 v 和可选的掩码张量
        该函数执行注意力机制，并返回值张量 v 和注意力权重 qk 的加权和。
        """
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        # view用来将Tensor改成m*n维度，permute将tensor的维度换位：
        # 例如某个Tensor的size是（28，29，3，4）就可以利用Tensor.permute(0, 2, 1, 3)
        # 得到一个size为（28，3，29,4）的tensor
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # dot product
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        # flatten 展平维度，开始维度为2，压缩后面的维度
        # 例子——从第二位开始压缩，4*5 = 20
        # x = torch.randn(2, 3, 4, 5)
        # y = x.flatten(start_dim=2)
        # print(y.shape)
        # torch.Size([2, 3, 20])
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    """
    残差注意力块：x加上注意力块的输出，x加上交叉注意力块的输出，x加上MLP层的输出
    总之核心在于：x + f(x)
    """

    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        """
        self.attn: MultiHeadAttention类的实例,代表自注意力机制。
        self.attn_ln:LayerNorm类的实例,表示自注意力机制之后的层归一化。
        self.cross_attn和self.cross_attn_ln:用于交叉注意力的可选组件。
        如果cross_attention为True，它们将分别被初始化为MultiHeadAttention和
        LayerNorm的实例。如果cross_attention为False，它们将被设置为None。
        self.mlp:nn.Sequential的实例,表示多层感知机（MLP）。
        它由两个线性层组成,中间使用了GELU激活函数。
        self.mlp_ln:应用于MLP之后的层归一化的LayerNorm实例。
        """

        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
        前向传播过程包括以下步骤：
        1. x通过自注意力机制（self.attn）以及层归一化（self.attn_ln）传递。
        输出与输入张量x相加。
        2. 如果self.cross_attn不为None，x通过交叉注意力机制（self.cross_attn）
        以及层归一化（self.cross_attn_ln）传递。输出与之前的结果相加。
        3. 输出通过MLP（self.mlp）以及层归一化（self.mlp_ln）传递。
        4. 返回最终输出。
        """
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """
    AudioEncoder类实现了一个用于音频编码的神经网络模型。
    它通过卷积层和残差注意力块来提取音频的特征，并应用层归一化操作以优化特征表示。
    这样的编码器可以用于音频处理任务，如语音识别、音乐生成等
    """

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        self.conv1：卷积层，将输入的mel频谱图从n_mels维度转换为n_state维度。
        卷积核大小为3，填充为1。
        self.conv2：卷积层，将n_state维度的特征图进行下采样，步长为2，将特征图的大小减半。
        self.register_buffer：将一个名为"positional_embedding"的缓冲区注册到模型中。
        这个缓冲区存储了位置嵌入（positional embedding），用于为输入的音频序列的每个位置
        提供位置信息。sinusoids函数用于生成位置嵌入。
        self.blocks：一个nn.ModuleList，其中包含了多个ResidualAttentionBlock的实例。
        这些块将在编码器中被重复应用，用于提取音频的特征。
        self.ln_post：层归一化（LayerNorm）层，用于在编码器的输出上应用归一化。

        """
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer(
            "positional_embedding", sinusoids(n_ctx, n_state)
        )  # 存了模型的一些状态和参数

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        forward方法定义了音频编码器的前向传播过程。它接受一个名为x的输入张量，
        代表音频的mel频谱图。

        前向传播过程包括以下步骤：
        1. 将输入的mel频谱图通过第一个卷积层（self.conv1）进行卷积操作，并通过GELU激活函数进行非线性映射。
        2. 将第一个卷积层的输出通过第二个卷积层（self.conv2）进行卷积操作，并通过GELU激活函
        数进行非线性映射。这一步骤将特征图的大小减半，实现了下采样。
        3. 将特征图的维度进行转置，以便与位置嵌入的维度相匹配。
        4. 将转置后的特征图与位置嵌入相加，并转换为与输入相同的数据类型。
        5. 通过循环遍历self.blocks中的每个ResidualAttentionBlock实例，将特征图作为输入进行残差注意力块的前向传播。
        6. 将输出的特征图通过层归一化（self.ln_post）进行归一化操作，并返回最终的编码结果。
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        self.token_embedding:嵌入层,用于将文本的标记(token)转换为向量表示。
        它将输入的标记索引映射为n_state维的嵌入向量。
        self.positional_embedding:位置嵌入(positional embedding),是一个可学习的参数，
        用于为输入的标记序列的每个位置提供位置信息。
        self.blocks:一个nn.ModuleList,其中包含了多个ResidualAttentionBlock的实例。
        这些块将在解码器中被重复应用，用于进行注意力计算和特征融合。
        self.ln:层归一化(LayerNorm)层，用于在解码器的输出上应用归一化。
        self.mask:一个掩码张量，形状为(n_ctx, n_ctx)。它是一个上三角矩阵，其中上三角部分的值设为负无穷，
        用于屏蔽未来位置的注意力。
        """

        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        forward方法定义了文本解码器的前向传播过程。
        它接受两个输入张量 x 和 xa,分别代表文本的标记序列和待注意的音频特征。

        前向传播过程包括以下步骤：
        1. 将输入的token序列通过token嵌入层(self.token_embedding)进行嵌入操作，并将结果与位置嵌
        入(self.positional_embedding)相加。这样，每个标记都会被嵌入到一个向量表示，并包含位置信息。
        将嵌入结果转换为与音频特征相同的数据类型，并存储在变量 x 中。
        2. 通过循环遍历self.blocks中的每个ResidualAttentionBlock实例,将 x 和 xa 作为输入进行残差注意
        力块的前向传播。在每个块中，注意力机制会计算 x 和 xa 之间的注意力权重，并将权重应用到 x 上以融合音频特征信息。
        3. 将输出的特征向量通过层归一化(self.ln)进行归一化操作,并将其与文本嵌入矩阵的
        转置相乘(x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1))
        得到解码结果的logits。
                """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        """
        初始化了音频编码器(self.encoder)和文本解码器(self.decoder),并注册了一个缓存用于存储对齐头部的设置。
        """
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        """
        用于设置对齐头部。它将输入的字节字符串解码、解压缩，并转换为布尔型数组，然后将其作为对齐头部的掩码注册到模块的缓存中。


        base64.b85decode(dump) 解码
        gzip.decompress() 解压缩
        np.frombuffer() 从buffer-like object(the raw bytes in the buffer)创建一个array
        """
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()

        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        """
        用于将音频特征嵌入为文本表示。它调用音频编码器（self.encoder）来对输入的mel张量进行嵌入操作，并返回嵌入结果
        """
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        """
        用于计算文本解码的logits。它调用文本解码器（self.decoder）来对输入的标记序列和音频特征进行解码，并返回解码结果的logits。
        """
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        它接受音频特征的mel张量和标记序列的标记张量作为输入，并调用decoder和encoder来进行文本解码和音频特征嵌入，最终返回解码结果。
        """
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        用于安装键和值缓存的钩子函数。它接受一个可选的缓存字典作为输入，并返回一个包含缓存和钩子函数的字典和列表。
        钩子函数的作用在于在模型的前向传播过程中捕获中间结果，并将其保存到缓存字典中。具体来说，为了在
        MultiHeadAttention模块中安装钩子函数，以便在模型的前向传播过程中保存中间计算结果，以便后续重复使用。

        返回值：
        返回一个包含缓存和钩子函数的字典和列表。
        cache是一个可选的缓存字典，用于存储键和值的计算结果。
        如果cache为None，则会创建一个空字典。hooks是一个空列表，用于存储钩子函数。
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            """
            save_to_cache函数是一个回调函数，它在每次前向传播过程中被调用。它接受三个参数：module表示当前模块，
            _表示输入参数（这里用不到），output表示模块的输出结果。该函数首先判断当前模块是否在缓存字典中，如果不在
            或者输出结果的维度大于n_text_ctx，则直接将输出结果保存到缓存中；否则，将输出结果与缓存中的结果进行
            拼接，并将拼接后的结果保存到缓存中。
            """
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            """
            install_hooks函数用于遍历解码器中的所有模块，并判断是否为MultiHeadAttention类型的模块。如果是，
            则为该模块的key和value属性注册前向传播的钩子函数，该钩子函数会调用save_to_cache函数将中间结果保存到缓存中。
            """
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
