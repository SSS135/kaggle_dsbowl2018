
def unpad(fmap, unpad):
    neg_unpad = -unpad if unpad != 0 else None
    return fmap[..., unpad: neg_unpad, unpad: neg_unpad]