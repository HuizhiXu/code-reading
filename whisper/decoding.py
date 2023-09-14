from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .model import Whisper


@torch.no_grad()
def detect_language(
    model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None
) -> Tuple[Tensor, List[dict]]:
    """
    检测音频的语音，返回最有可能的语言的token的id以及他们的概率分布。
    如何做到：
    先得到model对应的Tokenizer，然后根据语音特征计算概率。
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )
    # 看mel是否有2维，如果是，增加一维
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # 看mel的size是不是符合model的音频上下文和状态。如果不符合，把mel扔进model回炉重造。
    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # 这段代码的目的是使用单个令牌(即开始转录令牌tokenizer.sot)对模型进行前向传递
    # sot表示start of transcript，tokenizer.sot令牌表示转录或序列的开始。
    # x中的每个元素都用tokenizer.sot令牌的值进行初始化。
    # 然后，将张量x与编码的音频特征(mel)一起传递给model.logits函数。
    # 该函数的目的是计算输入序列中每个令牌的未归一化对数概率（logits）。
    # 选择与x中的第一个（也是唯一一个）token对应的logits。
    # 通过使用单个令牌（即开始转录令牌）并将其与编码的音频特征一起传递给模型，
    # 实际上是要求模型根据给定的音频特征预测下一个令牌。
    # 这可以看作是一项语言建模任务，其中模型经过训练以生成连贯的令牌序列。
    # 在语言检测的上下文中，从此操作中获得的logits可以用于基于模型的预测来确定最可能的语言。
    # 代码中的后续步骤涉及使用获得的logits收集检测到的语言及其概率。
    # 总之，通过使用带有开始转录令牌的x并将其传递给model.logits，
    # 模型可以基于给定的音频特征生成预测结果，这可以用于语言检测的目的。
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]


    # mask是一个和logits shape相同的一维数组，把non-language的token对应的下标的值都设为False
    # 创建掩码张量mask并将与tokenizer词汇表中的非语言令牌对应的索引值设置为False的目的是抑制logits中非语言令牌的概率。
    # 在进行语言检测时，模型需要根据从model.logits函数获得的logits来预测最可能的语言。
    # 然而，tokenizer的词汇表中可能包含表示非语言元素或噪音的令牌。
    # 通过创建掩码张量并将与非语言令牌对应的索引值设置为False，代码在语言检测过程中有效地屏蔽或忽略了这些非语言令牌。
    # 这意味着与这些非语言令牌相关的概率将被抑制或认为是不重要的，在最终的预测中不会起到显著作用。
    # 通过抑制非语言令牌的概率，模型可以进行更可靠的预测，并将更高的概率分配给语言检测任务中感兴趣的部分。

    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf  # 设置为最小值
    # 得到最有可能的
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        """
        接受两个参数：model（类型为 "Whisper"）和 initial_token_length（初始标记长度）。
        """
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []
        # key_modules存储了模型解码器中每个块的注意力机制的键（key）模块。
        key_modules = [block.attn.key for block in self.model.decoder.blocks]
        # value_modules存储了模型解码器中每个块的注意力机制的值（value）模块
        value_modules = [block.attn.value for block in self.model.decoder.blocks]
        # 是一个合并了 key_modules 和 value_modules 的列表
        self.kv_modules = key_modules + value_modules

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """
        这段代码的目的是在进行模型推理之前，检查是否存在键值缓存（kv_cache）。
        如果缓存不存在，则通过调用模型的 install_kv_cache_hooks() 方法来安装键值缓存的钩子（hooks
        接下来，如果输入的 tokens 张量的最后一个维度的大小大于初始标记长度，
        则只需要使用最后一个标记，除了在首次前向传递时。
        所以将 tokens 张量切片，只保留最后一个标记。

        返回使用模型的解码器（decoder）进行推理的结果，其中传递了 tokens、audio_features 
        和键值缓存（kv_cache）作为参数。
        
        
        """
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        """
        清楚缓存，移除钩子
        """
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        """
        根据给定的源索引对键值缓存进行重新排列
        """
        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()


class SequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    用于选择具有最高对数概率的样本，并使用简单的长度归一化或 Google NMT 论文中的长度惩罚进行惩罚。
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        """
        它定义了一个名为 scores 的内部函数，用于计算每个样本的分数。
        在 scores 函数中，根据是否指定了长度惩罚，计算每个样本的惩罚值，并将对数概率除以该惩罚值得
        到最终的分数。然后，通过遍历 sum_logprobs 和 lengths，分别计算每个样本的长度，并调用 
        scores 函数计算每个样本的分数。最后，返回具有最高分数的样本的索引。


        问：长度惩罚是什么意思？
        答：长度惩罚（length penalty）是指在序列生成任务中，为了平衡生成序列的长度和生成概率之间的关系，
        对生成序列的得分进行调整或惩罚的方法。
        在机器翻译和自然语言生成任务中，生成的序列长度通常对应着生成的文本的质量和流畅度。
        较长的序列可能会包含更多的信息，但也可能导致生成结果过于冗长或不准确。为了避免这种情况，可以使用
        长度惩罚来对生成的序列进行调整。
        长度惩罚通常基于生成序列的长度进行计算，其中一种常见的方法是使用指数函数对长度进行惩罚。
        例如，Google NMT 论文中提出的长度惩罚公式为 $((5 + \text{{length}}) / 6)^\text{{length_penalty}}$。
        这个公式中，长度惩罚值根据序列的长度进行计算，然后与生成的对数概率相除，得到最终的分数。
        """
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """用于初始化解码新序列时的任何状态变量。"""
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """
        根据当前的跟踪（trace）和逻辑回归（logits）来指定如何选择下一个标记（token）。


        tokens 参数是一个张量, 形状为 (n_batch, current_sequence_length)，表示到目前为止上下
        文中的所有标记（包括前缀和 sot_sequence 标记）。
        logits 参数是一个张量,形状为 (n_batch, vocab_size)，表示当前步骤的每个标记的逻辑回归值。
        sum_logprobs 参数是一个张量,形状为 (n_batch)，表示每个序列的累积对数概率。
        方法返回一个元组,包含两个元素：
        tokens 是一个张量,形状为 (n_batch, current_sequence_length + 1)，表示添加了选择的下一个标记后的标记序列。
        completed 是一个布尔值,如果所有序列都到达文本的结尾,则为 True。

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """用于完成搜索并返回最终的候选序列

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """
        用于更新解码器的状态。根据给定的温度值，选择下一个token。
        如果温度为0，则选择具有最高logits值的token；否则，根据logits / temperature的概率分布
        采样下一个token。然后，计算log_softmax，并提取当前token的log概率。
        接下来，根据当前token是否为eot，将log概率加权添加到sum_logprobs中。
        然后，将当前token添加到tokens中。最后，检查是否所有序列都已经完成（即最后一个token为eot），
        并返回更新后的tokens和完成状态。



        在Categorical distribution中，每个可能的取值都与一个概率相关联，这些概率之和为1。
        例如，假设我们有一个随机变量X，它可以取值为{A, B, C}，并且每个取值的概率分别
        为{0.3, 0.4, 0.3}，那么这个随机变量X就服从Categorical distribution。


        在文本分类和图像分类等任务中，Categorical distribution通常用于表示类别分布。
        具体来说，可以使用一个向量来表示类别的概率分布，向量的长度等于类别的个数。
        每个位置上的值表示对应类别的概率。

        例如，在文本分类任务中，假设有3个类别：A、B和C。可以用一个长度为3的向量[0.3, 0.5, 0.2]
        来表示这三个类别的概率分布。其中，第一个位置的值0.3表示类别A的概率为0.3，
        第二个位置的值0.5表示类别B的概率为0.5，第三个位置的值0.2表示类别C的概率为0.2。
        
        """
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        """
        接受两个参数tokens和sum_logprobs，并返回一个元组(Tensor, list)。
        该方法用于最终处理解码结果。将每个序列的最后一个token设置为eot，并将结果返回。
        """
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()
"""
BeamSearchDecoder（束搜索解码器）和GreedyDecoder（贪婪解码器）的区别：


BeamSearchDecoder（束搜索解码器）和GreedyDecoder（贪婪解码器）是在序列生成任务中常用的两种解码策略。
它们的主要区别在于生成输出序列时的决策方式。

**GreedyDecoder（贪婪解码器）**是一种基本的解码策略，它在每个时间步选择当前概率最高的单词作为输出。
这意味着贪婪解码器只考虑每个时间步的局部最优解，并没有考虑整个序列的全局最优解。
因此，贪婪解码器通常会导致生成的序列不够多样化，可能会出现重复或不合理的片段。

**BeamSearchDecoder（束搜索解码器）**是一种更加复杂和计算密集的解码策略。
它在每个时间步维护一个大小为beam_width的候选集合，通过计算每个候选序列的得分来选择最有可能的序列。
BeamSearchDecoder会在每个时间步选择beam_width个候选序列，并在下一个时间步继续扩展这些序列，
直到生成完整的输出序列。通过维护候选序列集合，并在每个时间步考虑多个可能的选择，
束搜索解码器能够更好地探索搜索空间，从而更有可能找到全局最优解。

"""

class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):  
        """
        ，beam_size表示Beam Search算法中保留的候选序列的数量，
        eot表示序列的结束标志，inference表示推理模型，
        patience表示控制Beam Search算法的搜索宽度。
        """
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        """
        用于重置解码器的状态，将已完成的序列设置为None。
        """
        self.finished_sequences = None

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """
        该方法用于更新解码器的状态。首先，检查tokens的形状是否符合要求。
        然后，根据beam_size将输入的tokens和logits进行处理，计算候选序列
        的累积对数概率，并根据概率对候选序列进行排序，保留每个音频的top 
        beam_size个序列。接下来，更新解码器的状态，包括记录已完成的序列和
        重新排列推理模型的键值缓存。最后，检查是否所有音频都有足够数量的样
        本，如果是，则标记为完成。
        """
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        """
        将已完成的序列整理为tokens和sum_logprobs的列表，并根据patience补充未完成的序列。
        """
        
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual, language=language, task=options.task
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size, tokenizer.eot, self.inference, options.patience
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        if self.options.fp16:
            mel = mel.half()

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (
            torch.float16 if self.options.fp16 else torch.float32
        ):
            return TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)

    return result[0] if single else result
