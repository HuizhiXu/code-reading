import base64
import os
import string
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Dict, List, Optional, Tuple

import tiktoken

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


@dataclass
class Tokenizer:
    """
    Tokenizer类是对tiktoken库的一个包装，提供了快速访问特殊token的功能。
    encoding：tiktoken.Encoding类的实例，表示用于标记化的编码方案。
    language：一个可选的字符串属性，表示正在处理的语言。
    task：一个可选的字符串属性，表示正在执行的任务（例如，"transcribe"或"translate"）。
    sot_sequence：一个整数元组，表示转录开始序列。它初始化为空元组()，并在__post_init__方法中更新。
    special_tokens：一个将特殊标记映射到其对应整数值的字典。
    它初始化为空字典，并在__post_init__方法中填充。
  
    """

    encoding: tiktoken.Encoding
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int] = ()
    special_tokens: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """
        该方法在对象初始化后被调用。
        它使用从self.encoding对象获取的特殊标记的整数值填充special_tokens字典。
        它还根据self.language和self.task的值更新self.sot_sequence属性。
        """
        for special in self.encoding.special_tokens_set:
            special_token = self.encoding.encode_single_token(special)
            self.special_tokens[special] = special_token

        sot: int = self.special_tokens["<|startoftranscript|>"]
        translate: int = self.special_tokens["<|translate|>"]
        transcribe: int = self.special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())
        sot_sequence = [sot]
        if self.language is not None:
            sot_sequence.append(sot + 1 + langs.index(self.language))
        if self.task is not None:
            task_token: int = transcribe if self.task == "transcribe" else translate
            sot_sequence.append(task_token)

        self.sot_sequence = tuple(sot_sequence)

    def encode(self, text, **kwargs):
        """
        将给定的text编码为一系列token
        """
        return self.encoding.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        将一系列token转化为字符串
        """
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, token_ids: List[int], **kwargs) -> str:
        """
         将一系列token转化为字符串，同时包含时间戳token。
         时间戳标记是特殊标记，其ID范围高于其他特殊标记，并且被decode方法忽略。
        Timestamp tokens are above other special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        return self.encoding.decode(token_ids, **kwargs)

    """
    @cached_property 是一个装饰器函数，它用于将一个方法转换为一个属性（property），
        并且会缓存该属性的值，以提高性能
    """
    @cached_property
    def eot(self) -> int:
        """
        eot: 表示“结束标记（end of transcript）"，返回eot对应的token
        """
        return self.encoding.eot_token

    @cached_property
    def transcribe(self) -> int:
        """
        transcribe: 表示“转录”，返回<|transcribe|>对应的token
        """
        return self.special_tokens["<|transcribe|>"]

    @cached_property
    def translate(self) -> int:
        """
        translate: 表示“翻译”，返回<|translate|>对应的token
        """
        return self.special_tokens["<|translate|>"]

    @cached_property
    def sot(self) -> int:
        """
        表示转录开始
        """
        return self.special_tokens["<|startoftranscript|>"]

    @cached_property
    def sot_lm(self) -> int:
        """
        表示“语言模型开始（start of language model）”
        """
        return self.special_tokens["<|startoflm|>"]

    @cached_property
    def sot_prev(self) -> int:
        """
        表示“前一个转录开始（start of previous transcript）”
        """
        return self.special_tokens["<|startofprev|>"]

    @cached_property
    def no_speech(self) -> int:
        """
        表示无语音（no speech）
        """
        return self.special_tokens["<|nospeech|>"]

    @cached_property
    def no_timestamps(self) -> int:
        """
        表示“无时间戳（no timestamps）”
        """
        return self.special_tokens["<|notimestamps|>"]

    @cached_property
    def timestamp_begin(self) -> int:
        """
        表示时间戳的开始。
        """
        return self.special_tokens["<|0.00|>"]

    @cached_property
    def language_token(self) -> int:
        """
        language_token: 表示语言的特殊标记
        """
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError("This tokenizer does not have language token configured")

        if token := self.special_tokens.get(f"<|{self.language}|>", None):
            return token

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @cached_property
    def all_language_tokens(self) -> Tuple[int]:
        """
        返回所有语言标记的元组。它遍历special_tokens字典中的每个标记和标记ID，
        将标记包含在LANGUAGES列表中的标记ID添加到结果列表中，并将结果作为元组返回。
        """
        result = []
        for token, token_id in self.special_tokens.items():
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @cached_property
    def all_language_codes(self) -> Tuple[str]:
        """
        回所有语言标记所对应的语言代码的元组。
        它使用 decode 方法将每个语言标记解码成对应的语言代码，并移除标记中的 <|> 符号。
        """
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    @cached_property
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        """
        返回包含无时间戳（no timestamps）标记的转录开始（start of transcript）序列的元组。
        它将 sot_sequence 的结果转换为列表，然后将无时间戳标记添加到列表末尾。
        """
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @cached_property
    def non_speech_tokens(self) -> Tuple[int]:
        """
        返回用于在生成文本时抑制的标记的元组。这些标记用于避免生成实际上没有在音频中被说出的文本，
        例如特定的说话者标签或非语音注释。该属性包含一系列标点符号和其他特殊符号的标记。
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += (
            "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        )

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [
                self.encoding.encode(symbol),
                self.encoding.encode(" " + symbol),
            ]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def split_to_word_tokens(self, tokens: List[int]):
        """
        将给定的标记列表拆分为单词标记的列表。根据不同的语言，使用不同的拆分方法。
        对于一些语言（如中文、日文、泰文、老挝文和缅甸文），由于它们通常不使用空格来分隔单词，
        因此使用 Unicode 码点进行拆分。对于其他语言，使用空格进行拆分。
        """
        if self.language in {"zh", "ja", "th", "lo", "my"}:
            # These languages don't typically use spaces, so it is difficult to split words
            # without morpheme analysis. Here, we instead split words at any
            # position where the tokens are decoded as valid unicode points
            return self.split_tokens_on_unicode(tokens)

        return self.split_tokens_on_spaces(tokens)

    def split_tokens_on_unicode(self, tokens: List[int]):
        """
        根据 Unicode 将给定的标记列表拆分为单词列表和对应的标记列表。
        它使用 decode_with_timestamps 方法将标记列表解码成文本，然后根据 Unicode 码点进行拆分。
        拆分后的单词和标记列表会作为结果返回。
        
        """
        decoded_full = self.decode_with_timestamps(tokens)
        replacement_char = "\ufffd"

        words = []
        word_tokens = []
        current_tokens = []
        unicode_offset = 0

        for token in tokens:
            current_tokens.append(token)
            decoded = self.decode_with_timestamps(current_tokens)

            if (
                replacement_char not in decoded
                or decoded_full[unicode_offset + decoded.index(replacement_char)]
                == replacement_char
            ):
                words.append(decoded)
                word_tokens.append(current_tokens)
                current_tokens = []
                unicode_offset += len(decoded)

        return words, word_tokens

    def split_tokens_on_spaces(self, tokens: List[int]):
        """
        这个方法的作用是按照空格将给定的标记列表拆分成单词，
        同时保留每个单词对应的标记。拆分过程中会根据特定的条件将子词合并到前一个单词中。

        步骤如下：
        1. 调用 split_tokens_on_unicode 方法，将给定的标记列表按 Unicode 码点进行拆分，
        得到子词列表和对应的子词标记列表。
        2. 创建空的单词列表 words 和对应的标记列表 word_tokens。
        3. 遍历子词列表和子词标记列表，使用 zip 函数同时迭代它们。
        4. 对于每个子词和子词标记，进行以下判断：

            如果子词标记的第一个标记大于等于 self.eot，或者子词以空格开头，
            或者子词去除首尾空格后在 string.punctuation 中，或者当前单词列表为空，
            那么将该子词添加到单词列表 words 中，并将对应的子词标记列表添加到 word_tokens 中。
            否则，将该子词添加到当前单词列表 words 中的最后一个单词，并将对应的子词标记列表追加到 word_tokens 中的最后一个标记列表。
        5.返回拆分后的单词列表 words 和对应的标记列表 word_tokens。


        问题：为什么需要先用split_tokens_on_unicode方法拆分呢，直接按空格拆分不是很好嘛
        回答：文本被分成多个标记（tokens），并且这些标记不仅仅是基于空格来进行划分的。
        这是因为语言中的单词和短语不仅仅是通过空格来分隔的，还有其他的标点符号和特殊字符。
        因此，为了更好地捕捉文本中的语言结构和语义信息，需要将文本按照 Unicode 码点进行拆分。
        在代码中，使用 split_tokens_on_unicode 方法将给定的标记列表按照 Unicode 码点进行拆分，
        得到子词列表和对应的子词标记列表。然后，再根据一些条件判断，将这些子词合并成单词。



        """
        subwords, subword_tokens_list = self.split_tokens_on_unicode(tokens)
        words = []
        word_tokens = []

        for subword, subword_tokens in zip(subwords, subword_tokens_list):
            special = subword_tokens[0] >= self.eot
            with_space = subword.startswith(" ")
            punctuation = subword.strip() in string.punctuation
            if special or with_space or punctuation or len(words) == 0:
                words.append(subword)
                word_tokens.append(subword_tokens)
            else:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)

        return words, word_tokens


@lru_cache(maxsize=None) # 装饰器来添加缓存功能
def get_encoding(name: str = "gpt2"):
    """
    这段代码的作用是根据给定的 name 参数获取编码信息，
    包括词汇表路径、词汇表大小、正则表达式模式、标记排名和特殊标记等。这些信息将用于后续的编码和解码操作。
    
    vocab_path 是存储词汇表路径的变量，它根据给定的 name 参数构建。
    ranks 是一个字典，它存储了词汇表中每个标记的排名，通过读取词汇表文件并解析每一行来获得。
    n_vocab 是词汇表的大小，即词汇表中标记的数量，它等于 ranks 字典的长度。
    special_tokens 是一个空字典，用于存储特殊标记和它们的标记索引。
    specials 是一个列表，包含了一些特殊标记字符串。
    使用一个循环遍历 specials 列表，将特殊标记和对应的标记索引添加到 special_tokens 字典中，
    同时更新 n_vocab 的值。
    最后，函数返回一个 tiktoken.Encoding 对象，它包含了一些编码相关的信息，
    包括词汇表路径、词汇表大小、正则表达式模式、标记排名、特殊标记等。

    """
    vocab_path = os.path.join(os.path.dirname(__file__), "assets", f"{name}.tiktoken")
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(vocab_path) if line)
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    language: Optional[str] = None,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
) -> Tokenizer:
    """
    根据给定的语言参数和多语言参数，获取一个适当的编码和分词器对象
    调用 get_encoding 函数来获取一个编码对象 encoding，该对象包含了特定编码名称的相关信息。
    函数返回一个 Tokenizer 对象，该对象用于进行编码和分词操作，其中包括编码对象、语言参数和任务参数。

    multilingual：布尔值,表示tokenizer是否支持multiple language
    language: 指定的语言
    task：指定的任务，一种有三种：transcribe, translate和None


    如果提供language参数，那么要看它是否在LANGUAGES里面，LANGUAGES是此模型支持的语言种类
    """
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        encoding_name = "multilingual"
        language = language or "en"
        task = task or "transcribe"
    else:
        encoding_name = "gpt2"
        language = None
        task = None

    encoding = get_encoding(name=encoding_name)

    return Tokenizer(encoding=encoding, language=language, task=task)
