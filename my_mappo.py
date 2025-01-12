from dataclasses import MISSING, dataclass
from typing import Dict, Iterable, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import (IndependentNormal, MaskedCategorical,
                             ProbabilisticActor, TanhNormal)
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class Mappo(Algorithm):
    """多智能体PPO算法（MAPPO）

    参数:
        share_param_critic (bool): 是否在智能体组内共享Critic的参数。
        clip_epsilon (float): 在PPO损失函数中用于权重裁剪的阈值。
        entropy_coef (float): 计算总损失时熵项的权重。
        critic_coef (float): 计算总损失时Critic损失项的权重。
        loss_critic_type (str): 用于值函数差异的损失函数类型，可选 "l1"、"l2" 或 "smooth_l1"。
        lmbda (float): GAE（广义优势估计）中的lambda参数。
        scale_mapping (str): 用于标准差的正值映射函数，可选 "softplus"、"exp"、"relu"、"biased_softplus_1"。
        use_tanh_normal (bool): 如果为True，使用TanhNormal作为连续动作分布，其支持范围限制在动作域内；否则使用IndependentNormal。
        minibatch_advantage (bool): 如果为True，优势函数计算在大小为``experiment.config.on_policy_minibatch_size``的小批量上进行，
            而不是在整个``experiment.config.on_policy_collected_frames_per_batch``上进行，这有助于避免内存爆炸。
    """

    def __init__(
        self,
        share_param_critic: bool,
        clip_epsilon: float,
        entropy_coef: bool,
        critic_coef: float,
        loss_critic_type: str,
        lmbda: float,
        scale_mapping: str,
        use_tanh_normal: bool,
        minibatch_advantage: bool,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.share_param_critic = share_param_critic
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.loss_critic_type = loss_critic_type
        self.lmbda = lmbda
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal
        self.minibatch_advantage = minibatch_advantage

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # 定义损失函数
        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coef=self.entropy_coef,
            critic_coef=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
        )
        # 设置损失函数中使用的键
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        # 使用GAE（广义优势估计）作为值函数估计器
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss: ClipPPOLoss) -> Dict[str, Iterable]:
        # 返回损失函数中需要优化的参数
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        # 获取当前组的智能体数量
        n_agents = len(self.group_map[group])
        if continuous:
            # 如果是连续动作空间，计算logits的形状
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2  # 每个动作维度需要输出均值和标准差
        else:
            # 如果是离散动作空间，计算logits的形状
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        # 定义Actor的输入和输出规范
        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        # 获取Actor模型
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,  # 独立式Actor
            share_params=self.experiment_config.share_policy_params,  # 是否共享参数
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            # 如果是连续动作空间，添加NormalParamExtractor模块来提取均值和标准差
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            # 定义概率Actor
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        else:
            # 如果是离散动作空间，定义概率Actor
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                # 如果存在动作掩码，使用MaskedCategorical
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        # MAPPO在数据收集时使用与训练相同的独立式Actor
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        # 处理批次数据，确保包含必要的键
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )

        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        # 如果启用小批量优势计算，将批次数据分割为小批量
        loss = self.get_loss_and_updater(group)[0]
        if self.minibatch_advantage:
            increment = -(
                -self.experiment.config.train_minibatch_size(self.on_policy)
                // batch.shape[1]
            )
        else:
            increment = batch.batch_size[0] + 1
        last_start_index = 0
        start_index = increment
        minibatches = []
        while last_start_index < batch.shape[0]:
            minimbatch = batch[last_start_index:start_index]
            minibatches.append(minimbatch)
            with torch.no_grad():
                loss.value_estimator(
                    minimbatch,
                    params=loss.critic_network_params,
                    target_params=loss.target_critic_network_params,
                )
            last_start_index = start_index
            start_index += increment

        batch = torch.cat(minibatches, dim=0)
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        # 处理损失值，将熵损失合并到目标损失中
        loss_vals.set(
            "loss_objective", loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        )
        del loss_vals["loss_entropy"]
        return loss_vals

    def get_critic(self, group: str) -> TensorDictModule:
        # 获取Critic模型
        n_agents = len(self.group_map[group])
        if self.share_param_critic:
            # 如果共享Critic参数，Critic输出为单个状态值
            critic_output_spec = Composite({"state_value": Unbounded(shape=(1,))})
        else:
            # 如果不共享Critic参数，Critic输出为每个智能体的状态值
            critic_output_spec = Composite(
                {
                    group: Composite(
                        {"state_value": Unbounded(shape=(n_agents, 1))},
                        shape=(n_agents,),
                    )
                }
            )

        if self.state_spec is not None:
            # 如果存在全局状态，使用全局状态作为Critic的输入
            value_module = self.critic_model_config.get_model(
                input_spec=self.state_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=True,  # 集中式Critic
                input_has_agent_dim=False,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
                action_spec=self.action_spec,
            )

        else:
            # 如果不存在全局状态，使用局部观测作为Critic的输入
            critic_input_spec = Composite(
                {group: self.observation_spec[group].clone().to(self.device)}
            )
            value_module = self.critic_model_config.get_model(
                input_spec=critic_input_spec,
                output_spec=critic_output_spec,
                n_agents=n_agents,
                centralised=True,  # 集中式Critic
                input_has_agent_dim=True,
                agent_group=group,
                share_params=self.share_param_critic,
                device=self.device,
                action_spec=self.action_spec,
            )
        if self.share_param_critic:
            # 如果共享Critic参数，扩展状态值以匹配智能体数量
            expand_module = TensorDictModule(
                lambda value: value.unsqueeze(-2).expand(
                    *value.shape[:-1], n_agents, 1
                ),
                in_keys=["state_value"],
                out_keys=[(group, "state_value")],
            )
            value_module = TensorDictSequential(value_module, expand_module)

        return value_module


@dataclass
class MappoConfig(AlgorithmConfig):
    """MAPPO算法的配置类。"""

    share_param_critic: bool = MISSING
    clip_epsilon: float = MISSING
    entropy_coef: float = MISSING
    critic_coef: float = MISSING
    loss_critic_type: str = MISSING
    lmbda: float = MISSING
    scale_mapping: str = MISSING
    use_tanh_normal: bool = MISSING
    minibatch_advantage: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Mappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
