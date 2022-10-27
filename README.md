# WideNet_Code
Implementation of AAAI 2022 Paper:  Go wider instead of deeper [arXiv](https://arxiv.org/abs/2107.11817)

You can run our code in this way.
```shell
$ cd WideNet_Code
$ bash submit_tpu.sh
```

We trained our model on Google Cloud TPU v3. You can follow Google Cloud's document to setup the environment. For GPU users, our code would also be feasible by some small modifications. You can also reimplement our code directly by JAX or PyTorch. This implementation should be simple:
1) implementing one MoE layer, this can be supported by [ViT-MoE](https://github.com/google-research/vmoe) in JAX or [DeepSpeed MoE](https://github.com/microsoft/DeepSpeed) in Torch. You can certainly use other MoE implementation.
2) Share the weights of MoE layers and attention layers
3) Unshare the weights of Layer Norm.

If you have any question, please feel free to ping Fuzhao.

## Citing WideNet
If you use WideNet, you can cite our paper.
Here is an example BibTeX entry:

```bibtex
@inproceedings{xue2022go,
  title={Go wider instead of deeper},
  author={Xue, Fuzhao and Shi, Ziji and Wei, Futao and Lou, Yuxuan and Liu, Yong and You, Yang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={8},
  pages={8779--8787},
  year={2022}
}
```
