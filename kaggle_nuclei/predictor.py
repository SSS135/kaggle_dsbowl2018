import numpy as np
from skimage.transform import resize
from torch.autograd import Variable


def predict(model, dataloader, raw_data):
    preds_test_upsampled = []
    model.eval()
    sample_idx = 0
    for data in dataloader:
        data = Variable(data.cuda(), volatile=True)
        out = model(data).data.cpu().numpy()
        for o in out:
            size = raw_data[sample_idx]['img'].shape[:2]
            preds_test_upsampled.append(resize(np.squeeze(o), size,
                                               mode='constant', preserve_range=True))
            sample_idx += 1
    return preds_test_upsampled