# Diagnostic (descriptive, not gated): copy a student checkpoint with every output that Track A never supervises set to
# its neutral constant (decoder weight 0, bias = neutral): hand_scale = head_scale = 1; body exp = 0; FLAME eye / pose /
# jaw / eyelid / expression / shape = 0. Source checkpoints are only read.
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
s = torch.load(src, map_location='cpu', weights_only=False)
st = dict(s['student']); n = 0
for name, val in (('smplx_scale_decoder', 1.0), ('smplx_expression_decoder', 0.0), ('flame_poses_decoder', 0.0),
                  ('flame_expression_decoder', 0.0), ('flame_shape_decoder', 0.0)):
    w, b = f'head.{name}.weight', f'head.{name}.bias'
    assert w in st and b in st, name
    st[w] = torch.zeros_like(st[w]); st[b] = torch.full_like(st[b], val); n += 1
torch.save({'student': st, 'step': s.get('step', -1), 'source': src + ' (unsupervised outputs neutral)'}, dst)
print(dst, 'decoders neutralised:', n)
