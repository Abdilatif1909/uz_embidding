import re
try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False


# ============================
# UZBEK MORPHOLOGICAL FEATURES
# ============================

# O'zbekcha stopwords
UZBEK_STOPWORDS = {
    "va", "yoki", "lekin", "ammo", "shuning", "uchun", "esa",
    "bo'ldi", "bo'lsa", "bo'lmadi", "ham", "bu", "u", "ular", "men",
    "sen", "biz", "siz", "ularni", "mendan", "senden", "bizdan", "sizdan"
}

# O'zbekcha suffixlar
UZBEK_SUFFIXES = [
    "lar", "ler", "im", "ing", "i", "imiz", "ingiz", "lari",
    "ni", "ga", "dan", "da", "bilan", "cha", "chi", "li", "lik", "siz",
    "dek", "mi", "mu", "mo", "dagi", "ki"
]

# O'zbekcha vowels
VOWELS = set("aeiouoʻ")


# ============================
# ADVANCED TOKENIZER CLASS
# ============================

class UzbekTokenizer:
    """Advanced O'zbekcha tokenizer with morphological features"""
    
    def __init__(self, remove_stopwords=True, apply_stemming=True):
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
    
    def tokenize(self, text):
        """Main tokenization method"""
        text = self._normalize(text)
        tokens = text.split()
        
        processed = []
        for token in tokens:
            if self.remove_stopwords and token in UZBEK_STOPWORDS:
                continue
            
            if self.apply_stemming:
                token = self._stem(token)
            
            if token and len(token) > 1:
                processed.append(token)
        
        return processed
    
    def _normalize(self, text):
        """Text normalization"""
        text = text.lower()
        text = text.replace("ҳ", "x")
        text = text.replace("ғ", "g'")
        text = text.replace("қ", "q")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _stem(self, word):
        """Uzbek-specific stemming with affix removal"""
        sorted_suffixes = sorted(UZBEK_SUFFIXES, key=len, reverse=True)
        
        for suffix in sorted_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                if self._check_suffix_validity(word, suffix):
                    word = word[:-len(suffix)]
                    break
        
        return word
    
    def _check_suffix_validity(self, word, suffix):
        """Check if suffix removal maintains vowel harmony"""
        if len(word) <= len(suffix):
            return False
        return True
    
    def get_affixes(self, word):
        """Extract morphological affixes from word"""
        affixes = []
        original_word = word
        
        for suffix in sorted(UZBEK_SUFFIXES, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                affixes.append(suffix)
                word = word[:-len(suffix)]
        
        return {
            "original": original_word,
            "stem": word,
            "affixes": affixes
        }


# ============================
# BERT-SPECIFIC TOKENIZER
# ============================

class BertUzbekTokenizer:
    """Tokenizer for BERT models with Uzbek-specific preprocessing"""
    
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
    
    def preprocess(self, text):
        """Preprocess text before BERT tokenization"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        if self.remove_stopwords:
            tokens = text.split()
            tokens = [t for t in tokens if t not in UZBEK_STOPWORDS]
            text = ' '.join(tokens)
        
        return text.strip()


# ============================
# BACKWARD COMPATIBILITY
# ============================

class SentencePieceTokenizer:
    """Legacy SentencePiece tokenizer"""
    def __init__(self, model_path):
        if HAS_SENTENCEPIECE:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
        else:
            self.sp = None

    def tokenize(self, text):
        if self.sp:
            return self.sp.encode_as_pieces(text)
        return text.lower().split()


# ============================
# DEFAULT FUNCTIONS
# ============================

_tokenizer = UzbekTokenizer(remove_stopwords=True, apply_stemming=True)
_bert_tokenizer = BertUzbekTokenizer(remove_stopwords=True)


def uz_tokenize(text):
    """Tokenize Uzbek text"""
    return _tokenizer.tokenize(text)


def uz_tokenize_no_stem(text):
    """Tokenize Uzbek text without stemming"""
    tokenizer = UzbekTokenizer(remove_stopwords=True, apply_stemming=False)
    return tokenizer.tokenize(text)


def bert_preprocess(text):
    """Preprocess for BERT"""
    return _bert_tokenizer.preprocess(text)


def get_morphology(word):
    """Get morphological analysis"""
    return _tokenizer.get_affixes(word)