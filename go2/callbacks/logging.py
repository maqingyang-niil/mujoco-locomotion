from ast import Try

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

'''
记录每个分项奖励、跟踪误差、步态比例
'''
class RewardTermLoggerCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super().__init__(verbose)
        self._sum=defaultdict(float)
        self._counts=defaultdict(float)
        self._gait_counts=defaultdict(int)

    def _on_step(self)->bool:
        for info in self.locals.get("infos",()):
            for k,v in info.items():
                if k.startswith("rew/") or k in (
                    "tracking_lin_err",
                    "tracking_ang_err",
                    "base_height",
                ):
                    self._sum[k]+=float(v)
                    self._counts[k]+=1
            gait=info.get("gait")
            if gait is not None:
                self._gait_counts[gait]+=1
        return True

    def _on_rollout_end(self)->None:
        for k,v in self._sum.items():
            n=max(1,self._counts[k])
            name=k if k.startswith("rew/") else "diagnostics/"+k
            self.logger.record(name,v/n)

        total_gait=sum(self._gait_counts.values())
        if total_gait:
            for gait,cnt in self._gait_counts.items():
                self.logger.record("gait_fraction/"+gait,cnt/total_gait)

        self._sum.clear()
        self._counts.clear()
        self._gait_counts.clear()

'''
记录episode的长度和平均奖励,并写入日志
'''
class EpisodeStatesCallback(BaseCallback):
    def __init__(self, verbose = 0):
        super().__init__(verbose)

    def _on_step(self)->bool:
        return True

    def _on_rollout_end(self)->None:
        buffer=self.model.ep_info_buffer
        if not buffer:
            return
        self.logger.record("episode/mean_length",float(np.mean([e["l"] for e in buffer])))
        self.logger.record("episode/mean_reward",float(np.mean([e["r"] for e in buffer])))
