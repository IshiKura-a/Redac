import logging
import numpy as np

from collections import Counter

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


def log_redac_cluster(_best_k, _best_cluster_index, _best_assignment, _probs, _dataset, _id2label):
    logger.info("----------Cluster Center Info----------")
    logger.info(f'Using {_best_k} clusters')
    logger.info(_best_assignment.unique(return_counts=True))
    logger.info("----------------------------------------")

    logger.info("\n\n\n------------Cluster Info-------------")
    info = []
    for _i in range(_best_k):
        cluster_label = _probs[_best_cluster_index[_i]].argmax(dim=-1).item()
        cluster_context = _dataset["label"][_best_cluster_index[_i]]
        logger.info(f'{_i}th cluster idx {_best_cluster_index[_i]:4} with label (ResNet): '
              f'{_id2label[cluster_label]:10}, context: {cluster_context}')
        _member = (_best_assignment == _i).nonzero(as_tuple=False).squeeze().long()

        if isinstance(_member.tolist(), list):
            logger.info(f'Tot: {len(_member)}')
            labels = _probs[_member].argmax(dim=-1).tolist()
            _contexts = [_dataset["label"][x] for x in _member]
            logger.info(f'Label Stats: {Counter(labels)}')
            _contexts_counter = Counter(_contexts)
            logger.info(f'Context Stats: {_contexts_counter}')
            identical_members = list(
                filter(lambda x: x[0] == cluster_label and x[1] == cluster_context, zip(labels, _contexts)))
            identical_context_members = list(filter(lambda x: x == cluster_context, _contexts))
            logger.info(f'Label & Context Acc: {len(identical_members) / len(_member)}')
            _acc = len(identical_context_members) / len(_member)
            _common_acc = _contexts_counter.most_common(1)[0][1] / len(_member)
            logger.info(f'Context Acc:{_acc}')
            logger.info(f'Common Context Acc:{_common_acc}')
            info.append([cluster_context, len(_member), len(identical_context_members),
                         _contexts_counter.most_common(1)[0][1], _acc, _common_acc])
        else:
            info.append([cluster_context, 1, 1, 1, 1.0, 1.0])
            logger.info(f'This cluster has center only.')
        logger.info("----------------------------------------")

    head = ["context", "total", "same context", "common context", "acc", "common acc"]
    format_head = "{:>20}" * (len(head) + 1)
    format_row = "{:>20}" * (len(head) - 1) + "{:>20.9}" * 2
    logger.info(format_head.format("", *head))
    _acc_list = []
    _common_acc_list = []
    for (i, row) in enumerate(info):
        _acc_list.append(row[4])
        _common_acc_list.append(row[5])
        logger.info(format_row.format(i, *row))
    logger.info(format_row.format("", "", "", "", "", np.mean(_acc_list), np.mean(_common_acc_list)))


def log_kmeans_cluster(info):
    logger.debug('REF: K-Means')
    head = ["context", "common context", "total", "acc"]
    format_head = "{:>20}" * (len(head) + 1)
    format_row = "{:>20}" * (len(head)) + "{:>20.9}"
    logger.debug(format_head.format("", *head))
    _acc_list = []
    for (i, row) in enumerate(info):
        _acc_list.append(row[3])
        logger.debug(format_row.format(i, *row))
    logger.debug(format_row.format("", "", "", "", np.mean(_acc_list)))