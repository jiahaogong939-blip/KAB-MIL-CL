import logging
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#为机器学习或深度学习项目提供实验管理、调试和可视化功能
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level_dict[verbosity])
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    
    logger.addHandler(sh)

    return logger


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def visulization_embed(embed, label, path, id, btach_idx, CLASSES):
    embed = embed.detach().cpu().numpy()
    label = label.cpu().numpy()
    pca_embed = PCA(n_components=2).fit_transform(embed)
    _, ax = plt.subplots()
    scatter = ax.scatter(pca_embed[:, 0], pca_embed[:, 1], c=label)
    _ = ax.legend(scatter.legend_elements()[0], [CLASSES[int(l)] for l in set(label)])
    file_name = path + "[%s-%s]_gene_embed_visulization.png" % (str(id), str(btach_idx))
    plt.savefig(file_name, dpi=300)
    plt.close()


