# encoding: utf-8

import logging
import numpy as np
import torch
import datetime
import glob


def do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        losses,
):

    output_dir = cfg.OUTPUT_DIR
   
    device_type = cfg.MODEL.DEVICE
    device = torch.device(device_type)
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    
    logger = logging.getLogger("model.train")
    logger.info("Start training")
    j = 0
    models = glob.glob(output_dir + '/*.pt')
    if len(models) != 0:
        print(models.sort())    
        logger.info("Loaded last trained model - [%s].  If You wish to start from scratch, please remove models from output folder.", models[-1])
        model.load_state_dict(torch.load(models[-1]))
        j = len(models)

    for epoch in range(j, epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2= 0.0
        running_loss3= 0.0

        partial_loss1 = 0.0
        partial_loss2 = 0.0
        partial_loss3 = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, value, explanation, value_type = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)

            # forward + backward + optimize
            out1, out2, out3 = model(inputs[0])

            explanation = explanation.type(torch.LongTensor).to(device)
            value_type = value_type.type(torch.LongTensor).to(device)
            loss1 = losses[0](out1, value)
            loss2 = losses[1](out2, explanation)
            loss3 = losses[2](out3, value_type)
            loss = loss1 + loss2 + loss3


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            running_loss3 += loss3.item()
            partial_loss1 += loss1.item()
            partial_loss2 += loss2.item()
            partial_loss3 += loss3.item()
            if i % log_period == log_period -1:
                logger.info('EPOCH: [%d/%d] BATCHES [%d/%d] loss summed: %.3f loss MSE: %.3f loss NLLL: %.3f loss NLLL for Value-Type Pairs: %.3f' %
                        (epoch + 1, epochs, i + 1, len(train_loader) ,(partial_loss1 + partial_loss2 + partial_loss3) / log_period, partial_loss1 / log_period, partial_loss2 / log_period, partial_loss3/log_period))
                partial_loss1 = 0.0
                partial_loss2 = 0.0
                partial_loss3 = 0.0


        logger.info('EPOCH: [%d] FINISHED loss summed: %.3f loss MSE: %.3f loss NLLL for Type: %.3f loss NLLL for Value-Type Pairs: %.3f' %
                    (epoch + 1, running_loss / len(train_loader), running_loss1 / len(train_loader), running_loss2 / len(train_loader), running_loss3 / len(train_loader)))
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0

        if((epoch+1) % 5 == 0):
          logger.info('Finished Training')
          logger.info('Saving model ...')
          output_filename = datetime.datetime.now().strftime("%d%m%Y%H%M%S_") + str(epoch+1) + '_model.pt'
          output_dest = output_dir + '/' + output_filename
          torch.save(model.state_dict(), output_dest)
          drive_path = F"/content/content/MyDrive/drive/{output_filename}" 
          torch.save(model.state_dict(), drive_path) 
          logger.info('Model saved as :' + output_filename)
