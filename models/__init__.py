from . import FBM_L, FBM_NL, FBM_NP, LSTM, MICN, PDF, TCN, AdaMSHyper, Autoformer, Crossformer, CycleNet, DLinear, \
    ETSformer, FEDformer, FiLM, Fredformer, FreTS, Informer, Koopa, LightTS, MultiPatchFormer, \
    Nonstationary_Transformer, PatchMLP, PatchTST, PAttn, Pyraformer, Reformer, SCINet, SegRNN, SimpleTM, \
    TemporalFusionTransformer, TiDE, TimeBridge, TimeKAN, TimeMixer, TimesNet, TimeXer, TQNet, Transformer, TSMixer, \
    WPMixer, iTransformer, xPatch

MODEL_REQUIRES_CYCLE = [
    'CycleNet',
    'TQNet',
]

MODEL_DICT = {
    'AdaMSHyper': AdaMSHyper,
    'Autoformer': Autoformer,
    'Crossformer': Crossformer,
    'DLinear': DLinear,
    'ETSformer': ETSformer,
    'FBM_L': FBM_L,
    'FBM_NL': FBM_NL,
    'FBM_NP': FBM_NP,
    'FEDformer': FEDformer,
    'FiLM': FiLM,
    'FreTS': FreTS,
    'Fredformer': Fredformer,
    'Informer': Informer,
    'iTransformer': iTransformer,
    'Koopa': Koopa,
    'LightTS': LightTS,
    'LSTM': LSTM,
    'MICN': MICN,
    'MultiPatchFormer': MultiPatchFormer,
    'Nonstationary_Transformer': Nonstationary_Transformer,
    'PatchTST': PatchTST,
    'PAttn': PAttn,
    'Pyraformer': Pyraformer,
    'Reformer': Reformer,
    'SCINet': SCINet,
    'SegRNN': SegRNN,
    'SimpleTM': SimpleTM,
    'TCN': TCN,
    'TemporalFusionTransformer': TemporalFusionTransformer,
    'TiDE': TiDE,
    'TimeKAN': TimeKAN,
    'TimeMixer': TimeMixer,
    'TimesNet': TimesNet,
    'TimeXer': TimeXer,
    'Transformer': Transformer,
    'TSMixer': TSMixer,
    'WPMixer': WPMixer,
    'CycleNet': CycleNet,
    'PDF': PDF,
    'TQNet': TQNet,
    'TimeBridge': TimeBridge,
    'PatchMLP': PatchMLP,
    'xPatch': xPatch,
}