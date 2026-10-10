"""Regression checks against the pre-change branch and historical try5 fusion.

Run from the repository root: python tests/check_pixel_fusion.py
Requires the model's existing torch/timm dependencies; no training data needed.
"""

import argparse
import ast
import copy
import itertools
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.networks import EMCADNet
from lib.pixel_fusion import (add_fusion_arguments, fusion_mode_from_args,
                              inference_logits, restore_fusion_config)


def git_source(revision, path):
    return subprocess.check_output(
        ['git', '-c', 'safe.directory=' + ROOT.as_posix(), 'show', revision + ':' + path],
        cwd=ROOT).decode('utf-8')


def load_class(source, name, namespace):
    node = next(n for n in ast.parse(source.lstrip('\ufeff')).body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference>', 'exec'), namespace)
    return namespace[name]


def equal(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0)


DiceLoss = load_class((ROOT / 'utils/utils.py').read_text(encoding='utf-8'),
                      'DiceLoss', {'torch': torch, 'nn': nn})


def mutation(outputs, target):
    ce, dice = nn.CrossEntropyLoss(), DiceLoss(outputs[0].shape[1])
    loss = 0
    for size in range(1, 5):
        for indices in itertools.combinations(range(4), size):
            logits = sum(outputs[index] for index in indices)
            loss = loss + 0.3 * ce(logits, target) + 0.7 * dice(logits, target, softmax=True)
    return loss


def check_disabled():
    import lib.networks as current
    namespace = dict(vars(current))
    Original = load_class(git_source('7871f52', 'lib/networks.py'), 'EMCADNet', namespace)
    original_forward = next(n for n in ast.walk(ast.parse(git_source('7871f52', 'lib/networks.py').lstrip('\ufeff')))
                            if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    new_forward = next(n for n in ast.walk(ast.parse((ROOT / 'lib/networks.py').read_text(encoding='utf-8-sig')))
                       if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    assert ast.dump(original_forward) == ast.dump(new_forward)
    assert git_source('7871f52', 'lib/decoders.py') == (ROOT / 'lib/decoders.py').read_text(encoding='utf-8')
    for caa in ('off', 'caa'):
        kwargs = dict(num_classes=4, encoder='pvt_v2_b0', pretrain=False, caa_mode=caa)
        torch.manual_seed(29)
        old = Original(**kwargs).eval()
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(29)
        new = EMCADNet(**kwargs).eval()
        equal(old_rng, torch.get_rng_state())
        assert list(old.state_dict()) == list(new.state_dict())
        for key, value in old.state_dict().items():
            equal(value, new.state_dict()[key])
        image = torch.randn(1, 3, 64, 64)
        target = torch.randint(0, 4, (1, 64, 64))
        old_outputs, new_outputs = old(image), new(image)
        for a, b in zip(old_outputs, new_outputs):
            equal(a, b)
        assert inference_logits(new, new_outputs) is new_outputs[-1]
        old_loss, new_loss = mutation(old_outputs, target), mutation(new_outputs, target)
        equal(old_loss, new_loss)
        old_loss.backward()
        new_loss.backward()
        for (name, a), (other, b) in zip(old.named_parameters(), new.named_parameters()):
            assert name == other
            if a.grad is None:
                assert b.grad is None
            else:
                equal(a.grad, b.grad)
        torch.optim.AdamW(old.parameters(), lr=1e-4).step()
        torch.optim.AdamW(new.parameters(), lr=1e-4).step()
        for key, value in old.state_dict().items():
            equal(value, new.state_dict()[key])
        print('PASS disabled: CAA={}, initialization/RNG/forward/mutation/gradients/AdamW'.format(caa))


def head_only(cls, channels):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.fusion_mode = 'pixel_reliability'
    model._last_fusion_weights = None
    for name in ('4', '3', '2', '1'):
        setattr(model, 'reliability_head' + name, nn.Conv2d(channels, 1, 1))
    return model


def check_historical():
    Historical = load_class(git_source('1f205a1', 'lib/networks.py'), 'EMCADNet',
                            {'torch': torch, 'nn': nn, 'F': F})
    for channels in (4, 9):
        torch.manual_seed(53)
        old = head_only(Historical, channels)
        new = head_only(EMCADNet, channels)
        new.load_state_dict(old.state_dict(), strict=True)
        old_outputs = [torch.randn(2, channels, 9, 11, requires_grad=True) for _ in range(4)]
        new_outputs = [value.detach().clone().requires_grad_() for value in old_outputs]
        target = torch.randint(channels, (2, 9, 11))
        ce, dice = nn.CrossEntropyLoss(), DiceLoss(channels)
        equal(old.fuse_outputs(old_outputs), new.fuse_outputs(new_outputs))
        old_loss = old.fusion_auxiliary_loss(old_outputs, target, ce, dice)
        new_loss = new.fusion_auxiliary_loss(new_outputs, target, ce, dice)
        equal(old_loss, new_loss)
        old_loss.backward()
        new_loss.backward()
        for a, b in zip(old_outputs, new_outputs):
            equal(a.grad, b.grad)
        for a, b in zip(old.parameters(), new.parameters()):
            equal(a.grad, b.grad)
        clone = head_only(EMCADNet, channels)
        clone.load_state_dict(new.state_dict(), strict=True)
        equal(new.fuse_outputs(new_outputs), clone.fuse_outputs(new_outputs))
        softmax_loss = new.fusion_auxiliary_loss(new_outputs, target, ce, dice, dice_softmax=True)
        assert not torch.equal(softmax_loss, new_loss)
        print('PASS historical try5: C={}, fusion/loss/output and head gradients/checkpoint'.format(channels))


def check_binary():
    model = head_only(EMCADNet, 1)
    reference = copy.deepcopy(model)
    outputs = [torch.randn(2, 1, 9, 11, requires_grad=True) for _ in range(4)]
    reference_outputs = [value.detach().clone().requires_grad_() for value in outputs]
    target = torch.randint(2, (2, 1, 9, 11)).float()
    fused = reference.fuse_outputs(reference_outputs)
    probability = torch.sigmoid(fused)
    dice = 1 - (2 * (probability * target).sum() + 1e-5) / (
        probability.square().sum() + target.square().sum() + 1e-5)
    correctness = torch.cat([((p.detach().sigmoid() >= .5) == (target >= .5)).float()
                             for p in reference_outputs], dim=1)
    expected = .3 * F.binary_cross_entropy_with_logits(fused, target) + .7 * dice
    expected = expected + F.binary_cross_entropy_with_logits(
        reference._reliability_logits(reference_outputs), correctness)
    actual = model.fusion_auxiliary_loss(outputs, target)
    torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-7)
    expected.backward()
    actual.backward()
    for a, b in zip(reference_outputs, outputs):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
    for a, b in zip(reference.parameters(), model.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        amp_loss = model.fusion_auxiliary_loss(outputs, target)
    assert torch.isfinite(amp_loss)
    equal(model.fusion_auxiliary_loss(outputs, target[:, 0]), actual)
    weights = model._last_fusion_weights
    torch.testing.assert_close(weights.sum(dim=1), torch.ones_like(weights[:, 0]))
    assert len(model.fusion_weight_statistics()['mean']) == 4
    print('PASS binary: BCE/sigmoid Dice/correctness/gradients/target shapes/CPU autocast')


def check_enabled_model():
    for channels in (1, 4, 9):
        torch.manual_seed(71)
        model = EMCADNet(num_classes=channels, encoder='pvt_v2_b0', pretrain=False,
                         caa_mode='caa', fusion_mode='pixel_reliability').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        outputs = model(torch.randn(2, 1, 64, 64), mode='train')
        assert len(outputs) == 4 and all(p.shape == (2, channels, 64, 64) for p in outputs)
        if channels == 1:
            target = torch.randint(2, (2, 1, 64, 64)).float()
            base_loss = sum(F.binary_cross_entropy_with_logits(p, target) for p in outputs)
            auxiliary = model.fusion_auxiliary_loss(outputs, target)
        else:
            target = torch.randint(channels, (2, 64, 64))
            base_loss = mutation(outputs, target)
            auxiliary = model.fusion_auxiliary_loss(outputs, target, nn.CrossEntropyLoss(), DiceLoss(channels))
        loss = base_loss + auxiliary
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        initial = model.reliability_head1.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(initial, model.reliability_head1.weight)
        model.eval()
        with torch.no_grad():
            restored = EMCADNet(num_classes=channels, encoder='pvt_v2_b0', pretrain=False,
                                caa_mode='caa', fusion_mode='pixel_reliability').eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            image = torch.randn(1, 3, 64, 64)
            equal(inference_logits(model, model(image)), inference_logits(restored, restored(image)))
        print('PASS enabled full CAA model: C={}, train/backward/update/strict checkpoint/inference'.format(channels))


def check_config_and_wiring():
    def args(argv=()):
        parser = argparse.ArgumentParser()
        add_fusion_arguments(parser)
        return parser.parse_args(argv)
    assert fusion_mode_from_args(args()) == 'p1'
    enabled = args(['--use_pixel_reliability_fusion', '1'])
    saved = vars(enabled).copy()
    restored = args()
    restore_fusion_config(restored, saved, [])
    assert vars(restored) == saved
    restored = args()
    restore_fusion_config(restored, {'fusion_mode': 'pixel_reliability'}, [])
    assert fusion_mode_from_args(restored) == 'pixel_reliability'
    restore_fusion_config(restored, {}, [])
    assert fusion_mode_from_args(restored) == 'p1'
    try:
        restore_fusion_config(args(), saved, ['--use_pixel_reliability_fusion=0'])
    except RuntimeError:
        pass
    else:
        raise AssertionError('conflicting checkpoint option accepted')
    original_argv = sys.argv
    for name in ('acdc', 'polyp', 'synapse'):
        for prefix in ('train_', 'test_'):
            path = ROOT / (prefix + name + '.py')
            nodes = ast.parse(path.read_text(encoding='utf-8-sig')).body
            if name == 'synapse':
                start = next(index for index, node in enumerate(nodes)
                             if isinstance(node, ast.Assign) and any(
                                 isinstance(target, ast.Name) and target.id == 'parser' for target in node.targets))
                end = next(index for index, node in enumerate(nodes[start:], start)
                           if isinstance(node, ast.Assign) and any(
                               isinstance(target, ast.Name) and target.id == 'args' for target in node.targets))
                parser_nodes = nodes[start:end + 1]
            else:
                parser_nodes = [next(node for node in nodes if isinstance(node, ast.FunctionDef)
                                     and node.name == 'parse_args')]
            namespace = {'argparse': argparse, 'add_fusion_arguments': add_fusion_arguments}
            flags = ['--use_pixel_reliability_fusion', '1', '--fusion_mode', 'pixel_reliability',
                     '--fusion_loss_weight', '1', '--reliability_loss_weight', '1',
                     '--fusion_dice_softmax', '0']
            if prefix == 'test_':
                flags += ['--checkpoint', 'unused.pth']
            try:
                sys.argv = [str(path)] + flags
                exec(compile(ast.Module(body=parser_nodes, type_ignores=[]), str(path), 'exec'), namespace)
                parsed = namespace['args'] if name == 'synapse' else namespace['parse_args']()
            finally:
                sys.argv = original_argv
            assert fusion_mode_from_args(parsed) == 'pixel_reliability'
            print('PASS actual CLI parser (isolated from training imports): ' + path.name)
    for path, functions in (
            ('utils/polyp_utils.py', ('structure_loss', 'supervised_structure_loss')),
            ('utils/acdc_utils.py', ('_supervision_groups', 'supervised_loss'))):
        old_nodes = ast.parse(git_source('7871f52', path).lstrip('\ufeff')).body
        new_nodes = ast.parse((ROOT / path).read_text(encoding='utf-8-sig')).body
        for name in functions:
            old = next(n for n in old_nodes if isinstance(n, ast.FunctionDef) and n.name == name)
            new = next(n for n in new_nodes if isinstance(n, ast.FunctionDef) and n.name == name)
            assert ast.dump(old) == ast.dump(new), name + ' original loss changed'
    for name in ('synapse', 'acdc', 'polyp'):
        for prefix in ('train_', 'test_'):
            source = (ROOT / (prefix + name + '.py')).read_text(encoding='utf-8')
            assert 'add_fusion_arguments(parser)' in source
        if name != 'synapse':
            assert 'restore_fusion_config(args, config,' in (ROOT / ('test_' + name + '.py')).read_text(encoding='utf-8')
        source = (ROOT / 'sh' / ('start_train_' + name + '.sh')).read_text(encoding='utf-8')
        assert '--use_pixel_reliability_fusion' in source and '--fusion_dice_softmax' in source
    for name in ('busi', 'isic'):
        for prefix in ('train_', 'test_'):
            source = (ROOT / (prefix + name + '.py')).read_text(encoding='utf-8')
            assert '_base.main()' in source
        source = (ROOT / 'sh' / ('start_train_' + name + '.sh')).read_text(encoding='utf-8')
        assert '--use_pixel_reliability_fusion' in source
    cell = (ROOT / 'sh/start_train_cell.sh').read_text(encoding='utf-8')
    assert 'start_train_polyp.sh' in cell and 'USE_PIXEL_RELIABILITY_FUSION' in cell
    bash = Path('D:/install/Git/Git/bin/bash.exe') if sys.platform == 'win32' else Path('/bin/bash')
    for name in ('synapse', 'acdc', 'polyp', 'busi', 'isic', 'cell'):
        path = ROOT / 'sh' / ('start_train_' + name + '.sh')
        assert b'\r\n' not in path.read_bytes(), str(path) + ' has CRLF'
        subprocess.run([str(bash), '-n', path.as_posix()], check=True)
    for path in ROOT.glob('*.py'):
        compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    print('PASS CLI/config restoration/conflict checks; static shared-entrypoint wiring/Python syntax/six bash launchers LF and syntax')


if __name__ == '__main__':
    torch.set_num_threads(2)
    check_config_and_wiring()
    check_disabled()
    check_historical()
    check_binary()
    check_enabled_model()
    print('All regression checks passed. Dataset training/Dice improvement remain unverified.')
