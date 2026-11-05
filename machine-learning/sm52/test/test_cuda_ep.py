import sys

import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(sys.argv[1], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("session providers:", sess.get_providers())
try:
    out = sess.run(None, {"x": np.array([-1, 0, 1, 2], dtype=np.float32)})
    print("run ok:", out[0].tolist())
except Exception as e:  # on a GPU newer than sm_5x, a sm_52-only build has no kernel image
    print("run error:", str(e).splitlines()[-1][:200])
