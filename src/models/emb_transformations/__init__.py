def get_emb_transformation(ttype, **kwargs):
    if ttype.upper()=='LINEAR':
        from .linear import Linear
        return Linear(**kwargs)
    elif ttype.upper()=='TCN':
        from .tcn import TCN
        return TCN(**kwargs)
    else:
        raise ValueError
