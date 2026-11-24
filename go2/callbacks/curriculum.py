import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from go2.envs.commands import CommandCurriculum, CommandRanges

class CommandCurriculumCallback(BaseCallback):
    def __init__(self,
                 initial_ranges:CommandRanges,
                 final_ranges:CommandRanges,
                 threshold:float=0.85,
                 decay_threshold:float=0.65,
                 step:float=0.05,
                 verbose = 0):
        super().__init__(verbose)
        self.curriculum=CommandCurriculum(
            initial_comm=initial_ranges,
            final_comm=final_ranges,
            threshold=threshold,
            decay_threshold=decay_threshold,
            step=step,
        )
        self._scores:list[float]=[]
        self._pushed_level=None

    '''
    force:是否强制发送
    如果不强制推送,并且当前level没有发生变化,就直接退出
    '''
    def _push_ranges(self,force:bool=False)->None:
        level=self.curriculum.level
        if not force and self._pushed_level==level:
            return
        self.training_env.env_method("set_command_ranges",self.curriculum.ranges)
        self._pushed_level=level

    '''
    训练一开始,即使范围一致也强制推送
    '''
    def _on_training_start(self)->None:
        self._push_ranges(True)

    '''
    每次向量环境执行完一个控制步后,从所有环境的info中收集tracking scores
    得到n_env个infos
    '''
    def _on_step(self)->bool:
        for info in self.locals.get("infos",()):
            score=info.get("tracking_score")
            if score is not None:
                self._scores.append(score)
        return True

    def _on_rollout_end(self)->None:
        if not self._scores:
            return

        score=float(np.mean(self._scores))
        self._scores.clear()
        level=self.curriculum.update(score)

        self.logger.record("curriculum/tracking_score",score)
        self.logger.record("curriculum/level",level)
        r=self.curriculum.ranges
        self.logger.record("curriculum/max_vel_x",r.lin_vel_x[1])
        self.logger.record("curriculum/max_vel_y",r.lin_vel_y[1])
        self.logger.record("curriculum/max_yaw_rate",r.ang_vel_yaw[1])

        self._push_ranges()
