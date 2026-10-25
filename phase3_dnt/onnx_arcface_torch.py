"""ArcFace (insightface antelopev2 glintr100.onnx: IResNet-100 trained on Glint360K, BN-folded export) executed on the GPU
with PyTorch: a minimal interpreter for the graph's ops (Conv, BatchNormalization, PRelu, Add, Flatten, Gemm), weights read
from the ONNX initializers. Preprocessing = insightface ArcFaceONNX (112 x 112, BGR -> RGB, (x - 127.5) / 127.5) and
face_align.norm_crop (similarity to arcface_dst; least-squares Umeyama instead of skimage's SimilarityTransform, same
solution). Checked against onnxruntime (CPU) by `python phase3_dnt/onnx_arcface_torch.py` (cosine of the embeddings)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
DEFAULT_ONNX = "/home/ubx858/VACE/models/VACE-Annotators/face/models/antelopev2/glintr100.onnx"


class OnnxArcFace(torch.nn.Module):
    def __init__(self, path: str = DEFAULT_ONNX):
        super().__init__()
        import onnx
        from onnx import numpy_helper
        m = onnx.load(path)
        self.init = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in m.graph.initializer}
        self.nodes = [(n.op_type, list(n.input), list(n.output), {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}) for n in m.graph.node]
        self.inp = m.graph.input[0].name; self.out = m.graph.output[0].name
        ops = {n[0] for n in self.nodes}
        assert ops <= {"Conv", "BatchNormalization", "PRelu", "Add", "Flatten", "Gemm"}, ops
        names = [n.name for n in m.graph.node[:8]]
        mx = any(x.startswith(("Sub", "_minus")) for x in names) and any(x.startswith(("Mul", "_mul")) for x in names)
        self.mean, self.std = (0.0, 1.0) if mx else (127.5, 127.5)
        for i, (k, v) in enumerate(self.init.items()):
            self.register_buffer(f"w{i}", v, persistent=False)
        self._names = {k: f"w{i}" for i, k in enumerate(self.init)}

    def W(self, k):
        return getattr(self, self._names[k])

    @torch.no_grad()
    def forward(self, x):
        """x: (N, 3, 112, 112) float, RGB, already normalised. Returns (N, 512) L2-normalised embeddings."""
        env = {self.inp: x}
        for op, ins, outs, at in self.nodes:
            a = env[ins[0]]
            if op == "Conv":
                pads = at.get("pads", [0, 0, 0, 0]); assert pads[0] == pads[2] and pads[1] == pads[3]
                y = F.conv2d(a, self.W(ins[1]), self.W(ins[2]) if len(ins) > 2 else None, stride=tuple(at.get("strides", [1, 1])),
                             padding=(pads[0], pads[1]), dilation=tuple(at.get("dilations", [1, 1])), groups=at.get("group", 1))
            elif op == "BatchNormalization":
                y = F.batch_norm(a, self.W(ins[3]), self.W(ins[4]), self.W(ins[1]), self.W(ins[2]), False, 0.0, at.get("epsilon", 1e-5))
            elif op == "PRelu":
                y = F.prelu(a, self.W(ins[1]).reshape(-1))
            elif op == "Add":
                b = env[ins[1]] if ins[1] in env else self.W(ins[1]); y = a + b
            elif op == "Flatten":
                y = a.flatten(at.get("axis", 1))
            elif op == "Gemm":
                w = self.W(ins[1]); w = w if at.get("transB", 0) else w.t()
                y = at.get("alpha", 1.0) * a @ w.t() + at.get("beta", 1.0) * self.W(ins[2])
            env[outs[0]] = y
        return F.normalize(env[self.out], dim=-1)

    def preprocess(self, imgs_rgb01: torch.Tensor) -> torch.Tensor:
        """(N, 3, 112, 112) RGB in [0, 1] -> network input (insightface: blobFromImage(swapRB) of the BGR crop = RGB)."""
        return (imgs_rgb01 * 255.0 - self.mean) / self.std


def umeyama2d(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src)); S = np.eye(2); S[1, 1] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ S @ Vt; s = (D * np.diag(S)).sum() / (xs ** 2).sum() * len(src); t = mu_d - s * R @ mu_s
    return np.concatenate([s * R, t[:, None]], 1).astype(np.float32)


def align_theta(kps5: np.ndarray, size: int, out: int = 112) -> torch.Tensor:
    """affine_grid theta (2, 3) sampling the 112 x 112 ArcFace crop from a size x size image given its 5 landmarks."""
    M = np.vstack([umeyama2d(np.asarray(kps5, np.float64), ARCFACE_DST.astype(np.float64)), [0, 0, 1]])     # image px -> crop px
    Minv = np.linalg.inv(M)                                                                                  # crop px -> image px
    # normalised coords (align_corners=False): px = ((u + 1) * size - 1) / 2
    A = np.array([[(out) / 2.0, 0, (out - 1) / 2.0], [0, out / 2.0, (out - 1) / 2.0], [0, 0, 1]])           # crop norm -> crop px
    B = np.array([[2.0 / size, 0, 1.0 / size - 1], [0, 2.0 / size, 1.0 / size - 1], [0, 0, 1]])             # image px -> image norm
    return torch.from_numpy((B @ Minv @ A)[:2]).float()


def crops(imgs: torch.Tensor, thetas: torch.Tensor, out: int = 112) -> torch.Tensor:
    """imgs (N, 3, S, S) in [0, 1], thetas (N, 2, 3) -> (N, 3, 112, 112) bilinear (cv2.warpAffine's default), zero border."""
    g = F.affine_grid(thetas.to(imgs.device), (imgs.shape[0], 3, out, out), align_corners=False)
    return F.grid_sample(imgs, g, mode="bilinear", padding_mode="zeros", align_corners=False)


if __name__ == "__main__":
    import onnxruntime as ort, cv2
    net = OnnxArcFace().cuda().eval()
    s = ort.InferenceSession(DEFAULT_ONNX, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    img = (rng.random((512, 512, 3)) * 255).astype(np.uint8); img = cv2.GaussianBlur(img, (9, 9), 3)
    kps = np.array([[230, 240], [282, 238], [256, 270], [236, 300], [278, 298]], np.float32)
    # reference: insightface path (cv2 warp of the BGR image, blobFromImage swapRB)
    M = umeyama2d(kps.astype(np.float64), ARCFACE_DST.astype(np.float64)); bgr = img[:, :, ::-1].copy()
    aimg = cv2.warpAffine(bgr, M, (112, 112), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0 / net.std, (112, 112), (net.mean,) * 3, swapRB=True)
    e_ref = s.run(None, {s.get_inputs()[0].name: blob})[0][0]; e_ref /= np.linalg.norm(e_ref)
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255.0
    c = crops(x, align_theta(kps, 512)[None])
    e = net(net.preprocess(c))[0].cpu().numpy()
    e_same = net(torch.from_numpy(blob).cuda())[0].cpu().numpy()
    print("cos(torch net, ort) same input:", float(e_same @ e_ref), " cos(torch crop + net, ort pipeline):", float(e @ e_ref))
