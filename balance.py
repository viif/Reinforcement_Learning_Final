import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """创建世界"""
        # 设置智能体数量、包裹质量等参数
        self.n_agents = kwargs.pop("n_agents", 3)
        self.package_mass = kwargs.pop("package_mass", 5)
        self.random_package_pos_on_line = kwargs.pop("random_package_pos_on_line", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # 确保智能体数量大于1
        assert self.n_agents > 1

        # 设置线的长度和智能体半径
        self.line_length = 0.8
        self.agent_radius = 0.03

        # 设置奖励和惩罚的系数
        self.shaping_factor = 100
        self.fall_reward = -10

        # 设置是否可视化半直径
        self.visualize_semidims = False

        # 创建世界对象，设置重力和y轴半直径
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        # 添加智能体
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
            )
            world.add_agent(agent)

        # 添加目标点
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        # 添加包裹
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # 添加线和地面
        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        self.floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        # 初始化奖励张量
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        """重置世界"""
        # 设置目标点、线和包裹的位置
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (
                        -self.line_length / 2 + self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                    (
                        self.line_length / 2 - self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        # 设置智能体的位置
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -self.agent_radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        # 设置线和包裹的位置
        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )

        # 设置地面的位置
        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - self.agent_radius,
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.compute_on_the_ground()
        # 更新全局形状因子
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def compute_on_the_ground(self):
        """计算智能体和包裹是否在地面上"""
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)

    def reward(self, agent: Agent):
        """计算奖励函数，为每个智能体分配奖励值"""
        # 检查当前智能体是否为第一个智能体，如果是，则计算全局奖励
        is_first = agent == self.world.agents[0]

        # 如果是第一个智能体，执行以下操作
        if is_first:
            # 将位置奖励和地面奖励重置为0
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0

            # 计算智能体和包裹是否在地面上
            self.compute_on_the_ground()

            # 计算包裹与目标点之间的距离
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

            # 如果智能体或包裹在地面上，则给予惩罚
            self.ground_rew[self.on_the_ground] = self.fall_reward

            # 计算当前全局形状因子，即包裹与目标点之间的距离乘以形状因子
            global_shaping = self.package_dist * self.shaping_factor

            # 计算位置奖励，即上一次的全局形状因子减去当前全局形状因子的差值
            self.pos_rew = self.global_shaping - global_shaping

            # 更新全局形状因子
            self.global_shaping = global_shaping

        # 返回地面奖励和位置奖励的总和
        return self.ground_rew + self.pos_rew

    def observation(self, agent: Agent):
        """定义智能体的观测空间，返回一个包含多个观测信息的张量"""
        # 智能体的位置
        agent_pos = agent.state.pos

        # 智能体的速度
        agent_vel = agent.state.vel

        # 智能体与包裹之间的相对位置
        agent_to_package = agent.state.pos - self.package.state.pos

        # 智能体与线之间的相对位置
        agent_to_line = agent.state.pos - self.line.state.pos

        # 包裹与目标点之间的相对位置
        package_to_goal = self.package.state.pos - self.package.goal.state.pos

        # 包裹的速度
        package_vel = self.package.state.vel

        # 线的速度
        line_vel = self.line.state.vel

        # 线的角速度
        line_ang_vel = self.line.state.ang_vel

        # 线的旋转角度（归一化到[0, pi]区间）
        line_rot = self.line.state.rot % torch.pi

        # 将所有观测信息合并成一个张量，dim=-1表示沿着最后一个维度（即特征维度）进行拼接
        return torch.cat(
            [
                agent_pos,
                agent_vel,
                agent_to_package,
                agent_to_line,
                package_to_goal,
                package_vel,
                line_vel,
                line_ang_vel,
                line_rot,
            ],
            dim=-1,
        )

    def done(self):
        # 定义任务完成的条件
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        # 提供额外的信息
        info = {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}
        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        """
        计算启发式策略下的行动：根据观测到的环境状态计算智能体的行动
        """
        # 获取批量维度，即并行模拟的环境数量
        batch_dim = observation.shape[0]
        # 定义包裹相对于目标位置的索引位置
        index_package_goal_pos = 8
        # 提取包裹与目标的位置差
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]
        # 判断包裹与目标在y轴上的距离是否大于等于0
        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            # 如果动作空间是连续的
            # 计算连续行动：创建一个与观测张量形状相同的零张量，然后根据包裹与目标的y轴距离计算行动
            action_agent = torch.clamp(  # 限制行动范围在[u_range, -u_range]之间
                torch.stack(
                    [  # 堆叠两个张量以创建行动张量
                        torch.zeros(
                            batch_dim, device=observation.device
                        ),  # x轴行动，初始化为0
                        -dist_package_goal[:, Y],  # y轴行动，推动包裹向目标移动
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            # 如果包裹已经在目标上方，则不进行y轴行动
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            # 如果动作空间是离散的
            # 计算离散行动：创建一个填充为4的张量，表示默认行动
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            # 如果包裹已经在目标上方，则不执行行动
            action_agent[y_distance_ge_0] = 0
        # 返回计算得到的行动张量
        return action_agent


if __name__ == "__main__":
    # 交互式渲染
    render_interactively(
        __file__,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=True,
        control_two_agents=True,
    )
