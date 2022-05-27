# encoding: utf-8

import logging
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
import torch

def inference(
        cfg,
        model,
        val_loader
):
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    log_period = cfg.SOLVER.LOG_PERIOD
    logger = logging.getLogger("model.inference")
    logger.info("Start inferencing")

    gold_val = []
    gold_exp = []
    gold_valtyp = []
    pred_val = []
    pred_exp = []
    pred_valtyp = []
    model.eval()
    model.to(device)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, value, explanation, value_type = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            out1, out2, out3 = model(inputs)
            _, predicted = torch.max(out2, 1)
            _, predicted1 = torch.max(out3, 1)
            gold_val.append(value.item())
            gold_exp.append(explanation.item())
            gold_valtyp.append(value_type.item())
            pred_val.append(out1.item())
            pred_exp.append(predicted.item())
            pred_valtyp.append(predicted1.item())

            if i % log_period == log_period -1:
                logger.info('Progress [%d/%d]' %
                        (i + 1, len(val_loader)))

    logger.info('| F1 score for explanations: %.3f' % f1_score(gold_exp, pred_exp, average='micro'))
    logger.info('| Pearson for explanations: {:.3f}'.format(pearsonr(gold_exp, pred_exp)[0]))
    round_to_whole = [round(num) for num in pred_val]
    logger.info('| F1 score for values: %.3f' % f1_score(gold_val, round_to_whole, average='micro'))
    logger.info('| Pearson for values: {:.3f}'.format(pearsonr(gold_val, pred_val)[0]))
    logger.info('| F1 score for values: %.3f' % f1_score(gold_valtyp, round_to_whole, average='micro'))
    logger.info('| Pearson for values: {:.3f}'.format(pearsonr(gold_valtyp, pred_valtyp)[0]))
