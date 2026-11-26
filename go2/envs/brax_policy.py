from __future__ import annotations
import enum
import numpy as np


def _silu(x):
    return x/(1.0+np.exp(-np.clip(x,-60,60)))

_ACTIVATIONS={
    "silu":_silu,
    "swish":_silu,
    "relu":lambda x:np.maximum(x,0.0),
    "tanh":np.tanh
}

_FORMAT_VERSION=1


class BraxPolicy:
    def __init__(self,obs_mean,obs_std,weights,biases,action_size,activation="silu",meta=None):
        self.obs_mean=np.asarray(obs_mean,dtype=np.float64)
        self.obs_std=np.asarray(obs_std,dtype=np.float64)
        self.weights=[np.asarray(w,dtype=np.float64) for w in weights]
        self.biases=[np.asarray(b,dtype=np.float64) for b in biases]
        self.action_size=int(action_size)
        if activation not in _ACTIVATIONS:
            raise ValueError("unsupported activation %r" % activation)
        self.activation=activation
        self._act=_ACTIVATIONS[activation]
        self.meta=dict(meta or {})

        if self.weights[0].shape[0]!=self.obs_mean.shape[0]:
            raise ValueError("first layer has %d inputs while normaliser has %d" % (self.weights[0].shape[0],self.obs_mean.shape[0]))
        if self.weights[-1].shape[1]!=2*self.action_size:
            raise ValueError("last layer has %d outputs while expecting %d, including location and scale of a tanh normal" % (self.weights[-1].shape[1],2*self.action_size))

    @property
    def obs_size(self):
        return self.obs_mean.shape[0]

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            version=z["format_version"]
            if version!=_FORMAT_VERSION:
                raise ValueError("%s: policy format is %d while this loader reads %d" % (path,version,_FORMAT_VERSION))
            n=int(z["num_layers"])
            meta={k[len("meta_"):]: z[k].item() for k in z.files if k.startswith("meta_")}

            return cls(
                obs_mean=z["obs_mean"],
                obs_std=z["obs_std"],
                weights=[z["w%d"%i]for i in range(n)],
                biases=[z["b%d"%i]for i in range(n)],
                action_size=int(z["action_size"]),
                meta=meta,
            )

    def save(self,path):
        arrays=dict(
            format_version=np.array(_FORMAT_VERSION),
            num_layers=np.array(len(self.weights)),
            action_size=np.array(self.action_size),
            activation=np.array(self.activation),
            obs_mean=self.obs_mean.astype(np.float32),
            obs_std=self.obs_std.astype(np.float32),
        )

        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            arrays["w%d" % i]=w.astype(np.float32)
            arrays["b%d" % i]=b.astype(np.float32)

        for k,v in self.meta.items():
            arrays["meta_"+k]=np.array(v)
        np.savez(path,**arrays)

    def logits(self,obs):
        x=(np.asarray(obs,dtype=np.float64)-self.obs_mean)/self.obs_std
        last=len(self.weights)-1
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            x=x@w+b
            if i!=last:
                h=self._act(x)
        return x

    def __call__(self,obs):
        loc=self.logits(obs)[...,:self.action_size]
        return np.tanh(loc).astype(np.float32)