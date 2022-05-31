"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, writer=None, max_step=math.inf, log_tensorborad=False
                    ):
    # track the training progress
    global_rank = utils.get_rank()
    local_step = 0        
    if log_tensorborad:
        # if global_rank == 0:    
        steps_per_epoch = min(len(data_loader), max_step)
        global_start_step = steps_per_epoch * epoch
        avg_loss = 0.0
        avg_acc1 = 0.0
        avg_acc5 = 0.0
        avg_acc1_head1 = 0.0
        avg_acc1_head2 = 0.0
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if log_tensorborad:
                monitor = log_tensorborad and local_step % ((steps_per_epoch // 1000) + 1) == 0
                outputs = model(samples, monitor, writer, global_start_step + local_step)
            else:
                outputs = model(samples)
            if isinstance(outputs, list):
                loss_list = [criterion(o, targets) / len(outputs) for o in outputs]
                loss = sum(loss_list)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        # visualization - training
        if log_tensorborad and writer is not None and global_rank == 0:
            avg_loss += (loss_value - avg_loss) / (local_step + 1)
            acc1, acc5, acc1_head1, acc1_head2 = utils.acc(outputs, targets)
            avg_acc1 += (acc1.item() - avg_acc1) / (local_step + 1)
            avg_acc5 += (acc5.item() - avg_acc5) / (local_step + 1)
            avg_acc1_head2 += (acc1_head1.item() - avg_acc1_head2) / (local_step + 1)
            avg_acc1_head2 += (acc1_head2.item() - avg_acc1_head2) / (local_step + 1)
            global_step = global_start_step + local_step
            # if global_step == 0:
            #     writer.add_graph(model, input_to_model=samples, verbose=False) 
            writer.add_scalar("Loss", avg_loss, global_step)
            writer.add_scalar("Accuracy1", avg_acc1, global_step)
            writer.add_scalar("Accuracy5", avg_acc5, global_step)
            writer.add_scalar("Accuracy1_Head1", avg_acc1_head1, global_step)
            writer.add_scalar("Accuracy1_Head2", avg_acc1_head2, global_step)
            writer.add_scalars('Summary', {'Loss': avg_loss, 'Accuracy1': avg_acc1, 'Accuracy5': avg_acc5,
                'Accuracy1_Head1': avg_acc1_head1, 'Accuracy1_Head2': avg_acc1_head2}, global_step)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if isinstance(outputs, list):
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        local_step += 1
        if local_step >= max_step:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # evaluate_tmp(output[1], target, model, 'cuda')
            # Conformer
            if isinstance(output, list):
                loss_list = [criterion(o, target) / len(output)  for o in output]
                loss = sum(loss_list)
            # others
            else:
                loss = criterion(output, target)
        if isinstance(output, list):
            # Conformer
            acc1_head1 = accuracy(output[0], target, topk=(1,))[0]
            acc1_head2 = accuracy(output[1], target, topk=(1,))[0]
            acc1_total = accuracy(output[0] + output[1], target, topk=(1,))[0]
        else:
            # others
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        if isinstance(output, list):
            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_0=loss_list[0].item())
            metric_logger.update(loss_1=loss_list[1].item())
            metric_logger.meters['acc1'].update(acc1_total.item(), n=batch_size)
            metric_logger.meters['acc1_head1'].update(acc1_head1.item(), n=batch_size)
            metric_logger.meters['acc1_head2'].update(acc1_head2.item(), n=batch_size)
        else:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    if isinstance(output, list):
        print('* Acc@heads_top1 {heads_top1.global_avg:.3f} Acc@head_1 {head1_top1.global_avg:.3f} Acc@head_2 {head2_top1.global_avg:.3f} '
              'loss@total {losses.global_avg:.3f} loss@1 {loss_0.global_avg:.3f} loss@2 {loss_1.global_avg:.3f} '
              .format(heads_top1=metric_logger.acc1, head1_top1=metric_logger.acc1_head1, head2_top1=metric_logger.acc1_head2,
                      losses=metric_logger.loss, loss_0=metric_logger.loss_0, loss_1=metric_logger.loss_1))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
