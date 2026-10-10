"""Shared CLI/config plumbing for the isolated historical multi-head fusion."""

import math


FUSION_FIELDS = ('use_pixel_reliability_fusion', 'fusion_mode', 'fusion_loss_weight',
                 'reliability_loss_weight', 'fusion_dice_softmax')


def _nonnegative(value):
    import argparse
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('Expected a finite nonnegative loss weight')
    return number


def add_fusion_arguments(parser):
    parser.add_argument('--use_pixel_reliability_fusion', type=int, choices=[0, 1], default=0)
    parser.add_argument('--fusion_mode', default='pixel_reliability',
                        choices=['p1', 'fixed_sum', 'global_scalar', 'pixel_reliability'])
    parser.add_argument('--fusion_loss_weight', type=_nonnegative, default=1.0)
    parser.add_argument('--reliability_loss_weight', type=_nonnegative, default=1.0)
    parser.add_argument('--fusion_dice_softmax', type=int, choices=[0, 1], default=0,
                        help='multiclass auxiliary Dice only: 0 restores try5 raw logits; 1 uses softmax')


def fusion_mode_from_args(args):
    if not getattr(args, 'use_pixel_reliability_fusion', 0):
        return 'p1'
    return getattr(args, 'fusion_mode', 'pixel_reliability')


def restore_fusion_config(args, config, argv):
    saved = {field: config[field] for field in FUSION_FIELDS if field in config}
    # Historical configs had fusion_mode but no numeric master switch.
    if 'use_pixel_reliability_fusion' not in saved:
        saved['use_pixel_reliability_fusion'] = int(config.get('fusion_mode', 'p1') != 'p1')
    for field, configured in saved.items():
        option = '--' + field
        explicit = any(token == option or token.startswith(option + '=') for token in argv)
        if explicit and getattr(args, field) != configured:
            raise RuntimeError('{} conflicts with checkpoint config: requested={} saved={}'.format(
                option, getattr(args, field), configured))
        setattr(args, field, configured)


def inference_logits(model, outputs):
    core = model.module if hasattr(model, 'module') else model
    if getattr(core, 'fusion_mode', 'p1') == 'p1':
        return outputs[-1]
    import torch
    with torch.no_grad():
        return core.fuse_outputs(outputs)


def record_fusion_weights(model, writer, step):
    core = model.module if hasattr(model, 'module') else model
    if getattr(core, 'fusion_mode', 'p1') == 'p1' or (step != 1 and step % 50):
        return
    stats = core.fusion_weight_statistics()
    import logging
    logging.info('FUSION_WEIGHTS step=%s %s', step, stats)
    for name, value in zip(('p4', 'p3', 'p2', 'p1'), stats.get('mean', [])):
        writer.add_scalar('fusion/weight_' + name, value, step)
