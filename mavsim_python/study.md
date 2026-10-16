# `mavsim_python` 代码学习笔记

> 阅读范围：本目录下的 Python 源码；记录时间：2026-09-23。工程源于 Beard & McLain《Small Unmanned Aircraft》的章节式教学仿真。它不是一个已经完整交付的飞控：大量核心函数仍保留 `TODO`、零值返回或注释实现。本文始终区分**架构/目标算法**与**该提交当前真正执行的代码**。

## 1. 总览：从任务到飞行器再到显示

系统要模拟一架 Aerosonde 固定翼无人机，在城市环境中接收航点或目标指令、规划并跟踪路径，同时可加入风、传感器、相机、云台和目标定位。设计上的闭环是：

```text
MsgWorldMap / 任务目标
        │
PathPlanner (simple / RRT) → MsgWaypoints → PathManager → MsgPath
                                                   │
传感器 ← MavDynamicsSensors ← 风 + MavDynamicsControl ← Autopilot ← PathFollower
             │                         │                ↑             │
             └→ Observer / EKF → MsgState ─────────────┘       MsgAutopilot

目标运动 → Camera + Gimbal → 像素 MsgCamera → Geolocation EKF
                                      │
                           ViewManager / viewers / data plot
```

`launch_files/chapXX/mavsim_*.py` 是上述各阶段的可执行教学入口；每个入口都把目录根加入 `sys.path`，实例化模块后用 `SIM.ts_simulation=0.01 s` 的 while 循环推进。接口采用可变的轻量消息对象，而不是 ROS 消息：`MsgState` 是状态，`MsgDelta` 是舵面/油门/云台输入，`MsgSensors` 是测量，`MsgAutopilot` 是高层指令，`MsgPath`/`MsgWaypoints` 是航迹数据。

### 目录和模块职责

| 目录 | 主要文件 | 功能 |
|---|---|---|
| `parameters/` | `aerosonde_parameters.py`、`*_parameters.py` | 飞机尺寸、惯量、气动导数、传感器误差、相机、规划和仿真采样常量。 |
| `message_types/` | `msg_*.py` | 跨模块数据契约；避免模块直接依赖内部数组布局。 |
| `models/` | `mav_dynamics*`、`wind_simulation.py` | 六自由度飞行动力学及其控制/传感器/相机扩展；另含配平、线性化、目标和云台模型。 |
| `controllers/` | `autopilot*.py`、`pi/pid/pd/tf_control.py` | 可复用离散控制块，以及传统串级、TECS、LQR 三种飞控骨架。 |
| `estimators/` | `filters.py`、`observer*.py`、`geolocation.py` | 低通滤波、连续—离散 EKF、姿态/导航估计和基于相机的目标定位。 |
| `planners/` | `path_follower.py`、`path_manager.py`、`rrt_*.py` | 几何路径跟踪、航点状态机、Dubins 曲线及 RRT 全局规划。 |
| `tools/` | `rotations.py`、`transfer_function.py` 等 | 坐标变换、数值雅可比、饱和/角度包裹、测试信号、绘图辅助。 |
| `viewers/`、`plotter/` | `view_manager.py`、`draw_*.py`、`*_viewer.py` | PyQtGraph/OpenGL 3D 场景、相机/传感器/数据图及视频输出；不参与控制计算。 |
| `launch_files/` | `chap02`…`chap13` | 与教材章节对应的组装脚本和若干 `chap*_check.py` 数值检查。 |

## 2. 坐标、状态和数学记号

惯性坐标使用 NED：`[p_n,p_e,p_d]^T` 分别为北、东、下，故高度 `h=-p_d`；机体系为前、右、下，速度为 `[u,v,w]^T`，角速度为 `[p,q,r]^T`。姿态以 3-2-1（偏航 ψ、俯仰 θ、滚转 φ）描述。`tools.rotations.euler_to_rotation` 给出机体到惯性系旋转 `R_b^i=R_z(ψ)R_y(θ)R_x(φ)`，所以

`ṗ^i = R_b^i v^b`，`v_w^b=(R_b^i)^T v_w^i`。

动力学内部状态本应是 13 维：

`x=[p_n,p_e,p_d,u,v,w,e_0,e_1,e_2,e_3,p,q,r]^T`。

这里 `[e0,e1,e2,e3]` 是单位四元数，满足 `e0²+e1²+e2²+e3²=1`。四元数避免欧拉角在 `θ=±π/2` 的奇异性；`mav_dynamics._rk4_step` 每步显式归一化，以压制积分漂移。`MsgState` 则是控制器友好的派生状态，额外给出空速 `V_a`、迎角 `α`、侧滑 `β`、地速 `V_g`、航迹角 `χ`、风和陀螺偏置。

速度关系尤其重要：总风由惯性系稳态风加机体系阵风构成，空气相对速度为 `v_a^b=v^b-v_w^b`，随后

`V_a=||v_a^b||`，`α=atan2(w_a,u_a)`，`β=asin(v_a/V_a)`；

地速 `V_g=||R_b^i v^b||`，`χ=atan2(ṗ_e,ṗ_n)`，`γ=asin(ṗ_d/V_g)`（注意 NED 的下为正，爬升时 γ 为负）。

## 3. `models/`：物理世界

### 3.1 基类 `mav_dynamics.py`

`MavDynamics.update(forces_moments)` 接受 `[F_x,F_y,F_z,ℓ,m,n]^T`，用四阶 Runge–Kutta（RK4）积分：

`x[k+1]=x[k]+T_s/6·(k1+2k2+2k3+k4)`。

标准刚体方程应为：

`ṗ^i=R_b^i v^b`；`v̇^b=[r v-q w, p w-r u, q u-p v]^T+F^b/m`；

`ω̇=J^{-1}(M-ω×Jω)`；四元数运动学为 `ė=1/2 Ω(ω)e`。

`aerosonde_parameters.py` 的 `Jx,Jy,Jz,Jxz` 和预先计算的 `gamma1…gamma8` 用于展开带积惯量 `Jxz` 的滚转/偏航耦合方程。**现状：** `_f()` 整段为 TODO 并返回零导数，因此 chap03 当前不会按施加力矩真实运动；虽然 RK4、四元数归一化和状态消息映射已写好。

### 3.2 控制气动扩展 `mav_dynamics_control.py`

此类继承基类，`update(delta,wind)` 先通过 `_forces_moments` 求合力矩，再积分，并由 `_update_velocity_data` 更新空气数据。设计公式是：

- 重力：`F_g^b=mg[-sinθ, cosθ sinφ, cosθ cosφ]^T`；
- 动压：`q̄=½ρV_a²`；升阻力通常为 `L=q̄S C_L`、`D=q̄S C_D`，再由 α 投到机体系；
- 纵向系数依赖 `α,qc/(2V_a),δ_e`，横侧向 `Y,ℓ,n` 依赖 `β,pb/(2V_a),rb/(2V_a),δ_a,δ_r`；参数表提供 `C_Lα,C_mq,C_ℓδa,C_nδr` 等导数；
- 推进器通过电压 `V_in=δ_t V_max`、电机反电势与桨的 `C_T(J), C_Q(J)` 解桨转速，再得推力和反扭矩。

**现状：** 风变换、空速角、力矩和电机函数均是 TODO/零值。`_update_true_state()` 的派生量逻辑本身完整，但上游物理量不会被真正求出。

### 3.3 传感器、风、相机、云台与目标

- `mav_dynamics_sensors.py` 应在动力学之上模拟陀螺 `ω+b+n`、比力计、磁强计、静压 `ρgh`、差压 `½ρV_a²` 和慢周期 GPS。GPS 偏差按一阶 Gauss–Markov 随机游走，采样间隔来自 `sensor_parameters.ts_gps=0.2 s`。**现状：** 所有测量及 GPS 都硬置为 0；仅保留定时结构。
- `wind_simulation.py` 设计为稳态 NED 风加 Dryden 湍流。三个 `TransferFunction` 以白噪声驱动，输出机体系 `u_g,v_g,w_g`；理论上系数应由飞行速度、低空湍流尺度和标准差确定。**现状：** 分子全 0，故阵风恒为零。
- `mav_dynamics_camera.py` 是实际有功能的附加层：继承传感器飞机后，积分 `aż=k_az u_az`、`el̇=k_el u_el`，并限幅至 ±180°。
- `camera.py` 有效实现针孔投影。相机坐标由 `R_i^c=R_g^c(R_g^b)^T(R_b^i)^T` 得到，对相机系点 `[X,Y,Z]` 输出 `x=fX/Z+n_x`、`y=fY/Z+n_y`；`sigma_pixel=2` 像素。它还生成目标盒子的 8 个顶点供显示。
- `gimbal.py` 的 `pointAtGround/pointAtPosition/pointAlongVector` 是“视线转机体、由 `atan2` 求方位/俯仰、P 控制并限幅”的骨架，但所有关键表达式为 TODO。
- `target_dynamics.py` 是已实现的简化地面匀速模型：目标在道路格点随机选择四个正交方向，以固定速度 15 m/s 前进，在交叉口换向且避免越界。

### 3.4 配平与线性化

`trim.py` 用 SciPy SLSQP 计划寻找平衡状态和输入。约束固定 `||v||=V_a`、`v=0`、单位四元数、无滚转/偏航、`p=q=r=0`；目标通常应最小化与指定航迹角 `γ` 一致的动力学残差，例如 `||f(x,u)-[V_a cosγ,0,-V_a sinγ,0…]^T||²`。**现状：** `trim_objective_fun` 恒为 0，因此约束问题没有物理配平意义。

`compute_models.py` 的目标是围绕 `(x_trim,u_trim)` 以有限差分构造 `A=∂f/∂x`、`B=∂f/∂u`，拆为纵向 `[u,w,q,θ,h]` 与横侧向 `[v,p,r,φ,ψ]` 子系统，并写入 `models/model_coef.py`。还应计算常用近似传函，例如滚转 `φ/δ_a=a_{φ2}/[s(s+a_{φ1})]`。**现状：** 状态变换、雅可比、子矩阵及所有传函系数均未填；仓库中的 `model_coef.py` 因而只应视为占位数据来源。

## 4. `controllers/`：已实现控制积木与未完成飞控

`PIControl` 是可直接使用的离散 PI：梯形积分 `I_k=I_{k-1}+T_s(e_k+e_{k-1})/2`，`u=K_p e+K_i I`，再用回算 `I←I+T_s(u_sat-u)/K_i` 抗积分饱和。`PIDControl` 增加经一阶滤波的误差微分，系数为 `a1=(2σ-T_s)/(2σ+T_s)`、`a2=2/(2σ+T_s)`；也支持直接使用测得速率的 `update_with_rate`。`PDControlWithRate` 是 `u=K_p(y_c-y)-K_d ẏ`。`TFControl` 用双线性/梯形规则把一阶连续传函变为差分方程；`tools.transfer_function.TransferFunction` 则将任意 SISO 有理传函化为可控标准型并 RK4 积分。

三种 `Autopilot` 的意图如下：

| 文件 | 设计控制律 | 当前状态 |
|---|---|---|
| `autopilot.py` | 横向 `χ_c→φ_c→δ_a`，偏航阻尼 `r→δ_r`；纵向 `h_c→θ_c→δ_e`，`V_{a,c}→δ_t`；外环 PI、内环 PD。 | `update` 未填，返回全零舵量/指令状态。且 `TransferFunction` 的 import 被注释，实例化会报 `NameError`。 |
| `autopilot_tecs.py` | TECS 用总能量 `E=gh+V_a²/2` 控油门，用能量分配 `L=gh-V_a²/2`（或等价归一化量）控俯仰。 | 横向同上骨架，TECS 增益和 `update` 均为零/ TODO。 |
| `autopilot_lqr.py` | 对增广线性模型解 CARE：`AᵀP+PA-PBR⁻¹BᵀP+Q=0`，`K=R⁻¹BᵀP`，积分状态消除航向/高度/空速稳态误差。 | Q/R、P/K 都被替换为零，且横向 K 被误覆盖；无实际控制。 |

控制器参数文件 `control_parameters.py` 同样全部为零，说明 chap06 以后是待学生完成的作业框架，而非可得到闭环性能的基线。

## 5. `estimators/`：滤波和导航估计

`filters.py` 是本目录较完整的数学实现。

- `AlphaFilter`：`y_k=αy_{k-1}+(1-α)u_k`，α 越大越平滑但相位滞后越大。
- `ExtendedKalmanFilterContinuousDiscrete`：以 `N` 个小步 Euler 传播 `x̂←x̂+Δt f(x̂,u)`；用数值前向差分求 `A=∂f/∂x`、`G=∂f/∂u`，近似 `A_d≈I+ΔtA+Δt²A²/2`，协方差传播 `P←A_dPA_dᵀ+Δt²(Q+GQ_uGᵀ)`。量测更新为 `K=PHᵀ(HPHᵀ+R)⁻¹`、`x̂←x̂+K(y-h)`；协方差采用 Joseph 形式 `(I-KH)P(I-KH)ᵀ+KRKᵀ`，较普通 `P-KHP` 更能保持半正定。它还通过卡方距离拒绝离群量测。
- `KalmanFilterDiscrete` 是对应的线性离散 KF；工具中 `Jacobian` 也提供独立的前向差分版本。

`observer.py` 是教材常见的“两 EKF”方案：低通后的陀螺减偏置，静/动压反推高度与空速；姿态 EKF 状态 `[φ,θ]` 用重力/加速度更新；位置 EKF 状态 `[p_n,p_e,V_g,χ,w_n,w_e,ψ]` 用 GPS 和风三角伪量测更新。`observer_full.py` 将其扩展为 14 维 `[p^i,v^b,φ,θ,ψ,b_g,w_n,w_e]` 联合 EKF，并有 `to_MsgState` 将机体速度、姿态、风转换成 `V_a,α,β,V_g,χ`。两者中的关键模型函数 `f/h_*` 均残留 TODO，`observer_full.f` 甚至引用尚未定义的 `pos_dot` 等变量，故目前不可用；`observer_old.py`/`geolocation_old.py` 是较早版本供对照，不宜作为主入口。

`geolocation.py` 的目标是“常速度目标 + 视线距离”EKF，状态为 `[p_t^i,v_t^i,L]`。像素反投影先组成归一化相机射线 `ℓ_c∝[pixel_x,pixel_y,f]`，连续旋转至惯性系，再以平地交点求 `L` 与 `p_t=p_m+Lℓ_i`；多帧 EKF 将有噪声的交点平滑成速度连续轨迹。代码中的输入、状态方程和旋转链仍为 TODO，所以这是明确的设计说明而非当前功能。

## 6. `planners/`：几何路径层

`MsgWaypoints` 保存 NED 航点矩阵及空速、航向、父节点、代价和连通终点标记；`MsgPath` 将当前段标准化成直线、圆轨道或螺旋线。`planner_parameters.py` 依据协调转弯关系 `R_min=V_a²/(g tan φ_max)` 取最小转弯半径；这是固定翼规划必须考虑的曲率约束。

`path_planner.py` 是任务层分派器：`simple_straight` 产生四个固定 fillet 航点，`simple_dubins` 产生带航向的四个航点；`rrt_straight` 与 `rrt_dubins` 则把当前状态作为起点、按所在半区选择城市对角作为终点，再调用对应 RRT。因此前两者可正确生成消息，但其后续 PathManager/PathFollower 尚未实现，不能完成跟踪。

- `path_follower.py` 设计成将一段几何路径变为 `MsgAutopilot`。直线路径通常令横向误差 `e_py` 反馈到 `χ_c=χ_q-χ_inf(2/π)atan(k_path e_py)`，高度命令由直线的 NED 斜率插值得到；圆轨道则以半径误差产生航向偏置，并用 `φ_ff=λ atan(V_g²/(gρ))` 前馈。**现状：** 参数和输出均是 0。
- `path_manager.py` 应维护前/当前/后航点指针和半空间切换条件 `(p-r)^Tn≥0`。直线模式在角平分面换段；fillet 模式在航点前后切换直线与半径 R 的圆弧；Dubins 模式按“起始圆—公切线—末端圆”的状态机切换。该文件几乎全部 TODO，且 `_construct_fillet_line` 有 `self.ptr_previous` 拼写错误、`_inHalfSpace` 的 docstring 处有多余 `)`，直接导入会产生语法错误。
- `dubins_parameters.py` 的数学目标是枚举 RSR、RSL、LSR、LSL 四条最短曲率受限路径，计算两个圆心、切点、切换半空间和长度，要求起终点间距至少 `2R`。`compute_points()` 中留有完整绘制步骤注释，但实际返回零点阵。
- `rrt_straight_line.py`、`rrt_dubins.py` 设计为随机采样树：采样 → 最近节点 → 以固定段长扩展 → 沿线/曲线采样碰撞检测 → 累积代价与父指针 → 连上目标 → 回溯最小代价路径 → 直视可达的平滑。`heightAboveGround`、直线碰撞采样、回溯等辅助函数已有实现；树扩展、碰撞/平滑关键部分则仍是 TODO。`path_manager_follow_target.py` 是围绕地面目标生成轨道的专用简化管理器，也留有占位。

## 7. 其它支撑层

`tools.wrap.wrap` 把角度调整到参考角 ±π，避免航向环在 ±π 跳变时产生约 2π 的假误差；`saturate` 限制执行器/指令；`signals.py` 生成阶跃、正弦、方波、双脉冲和随机激励，chap05 以此激发 phugoid、滚转/螺旋和 Dutch-roll 模态。`drawing.py` 负责把顶点旋转、平移并组三角面。

`viewers/` 由 `ViewManager` 汇集 `mav_*_viewer`、`data_viewer`、`sensor_viewer`、`camera_viewer`、`geolocation_viewer` 等；`draw_mav`、`draw_path`、`draw_waypoints`、`draw_map`、`draw_target`、`draw_camera_fov` 是其几何图元。`tools.plotter.py` 与 `plotter/plotter.py` 是滚动时间序列图封装，`video_writer.py` 可按 `SIM.ts_video` 抓取画面。它们主要是观测层，不会修正飞行器状态。

## 8. 各章节入口对应的学习顺序

| 章节脚本 | 意图 |
|---|---|
| chap02 | 手动改变 `MsgState`，验证 3D 飞机/航天器可视化。 |
| chap03 | 直接输入力和力矩，学习四元数 6-DOF 刚体积分。 |
| chap04 | 加入气动、推进和风。 |
| chap05 | 求配平、线性化，使用激励观察飞行模态。 |
| chap06 | 用真实状态闭环验证传统自动驾驶仪。 |
| chap07 | 加传感器模型（脚本实际仍使用真实状态控制）。 |
| chap08 | 加观测器，设计上使用估计状态闭环。 |
| chap10 | 路径跟踪：直线/轨道/螺旋。 |
| chap11 | 航点管理：直线、fillet、Dubins。 |
| chap12 | 城市地图的简单路径或 RRT 路径规划。 |
| chap13 | 云台指向、相机像素与目标定位/绕目标飞行。 |

## 9. 当前可运行性结论与建议

这个仓库最适合作为“按 TODO 完成教材作业”的模板。当前真正较完整、可独立学习/复用的是：消息类、旋转变换、RK4 框架、PI/PID/PD/TF 控制块、通用 EKF、相机针孔投影、目标道路运动、信号与可视化工具。完整的飞机—传感器—估计—控制—规划闭环**当前不能正常产生可信仿真**，原因包括动力学导数、气动力/推进、传感器、配平/线性化、控制参数与飞控、观测模型、路径跟随/管理和 RRT/Dubins 都有占位实现；另有前述 `TransferFunction` 未导入和 `path_manager.py` 语法问题。

推荐按 chap03→04→05→06→07→08→10→11→12→13 的顺序实现和检查，每完成一层就运行同章节 `chap*_check.py`（现有检查主要覆盖 chap03–05、07）。实现时应始终先确认矩阵维度和坐标系：位置 NED、海拔为 `-p_d`、旋转矩阵方向为 body→inertial、风的稳态/阵风坐标系不同；这是本工程最容易造成“数值看似稳定却方向全错”的问题。


## 10. `compute_models.py` 公式与线性模型问答

> 整理时间：2026-10-10。以下收录从 `compute_models.py` 公式解释开始的提问与回答（10.1～10.4 为四轮，10.5 为第 6 章完成后关于设计模型与真值关系的补充），保留公式、数值比较及后续澄清。第 1～9 节是 2026-09-23 的历史笔记，其中关于第 3～5 章未实现的描述不代表当前代码状态；本节说明针对当前实现。

### 10.1 代码与 PDF 数学公式的对应关系

**提问：** 解释一下 [compute_models.py](/home/zr/mavsim_public/mavsim_python/models/compute_models.py) 代码对应pdf中的数学公式

**回答：**

[compute_models.py](/home/zr/mavsim_public/mavsim_python/models/compute_models.py) 的作用是：**在给定配平状态和配平输入附近，把非线性飞机模型转换成控制设计用的线性模型。**

它主要对应第 5 章中的传递函数推导与数值线性化部分，下面的 Slide 编号均指这份资料：[chap5.pdf](/home/zr/mavsim_public/pdf/chap5.pdf)

代码有两条计算路线：

```text
配平状态、配平输入
    ├─ compute_ss_model → 纵向、横向状态空间矩阵
    └─ compute_tf_model → 滚转、俯仰、空速传递函数系数
```

**先明确状态和输入的顺序。**

代码使用的欧拉角状态是：

$$
x_e=
[p_n,p_e,p_d,u,v,w,\phi,\theta,\psi,p,q,r]^T
$$

其中 $u,v,w$ 是机体系速度，$p,q,r$ 是机体系角速度。这里的 $q$ 表示俯仰角速度。

控制输入向量为：

$$
\delta=[\delta_e,\delta_a,\delta_r,\delta_t]^T
$$

分别是升降舵、副翼、方向舵和油门。**PDF Slide 53、54 使用的输入顺序是 $[\delta_t,\delta_e,\delta_a,\delta_r]^T$，代码的顺序不同，所以输入选择矩阵也相应调整了。**

---

**1. `euler_state()`、`quaternion_state()`：对应 Slide 50、51 的坐标转换**

飞机仿真内部使用四元数状态：

$$
x_q=
[p_n,p_e,p_d,u,v,w,e_0,e_1,e_2,e_3,p,q,r]^T
\in\mathbb R^{13}
$$

而线性模型使用欧拉角状态 $x_e\in\mathbb R^{12}$。

[第 151 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:151) 的 `euler_state()` 实现：

$$
x_e=T_e(x_q)
$$

即把四元数替换成 $\phi,\theta,\psi$，其他状态保持原顺序。

[第 160 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:160) 的 `quaternion_state()` 实现反向转换：

$$
x_q=T_q(x_e)
$$

这样，线性化计算可以用欧拉角扰动，底层仍调用已经实现的四元数动力学。

---

**2. `f_euler()`：对应 Slide 52 的欧拉角动力学**

底层飞机模型给出：

$$
\dot x_q=f_q(x_q,\delta)
$$

但需要求导的是欧拉角动力学：

$$
\dot x_e=f_e(x_e,\delta)
$$

根据链式法则：

$$
f_e(x_e,\delta)=
\left.
\frac{\partial T_e}{\partial x_q}
\right|_{x_q=T_q(x_e)}
f_q(T_q(x_e),\delta)
$$

[第 167 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:167) 的实现分为两步。

首先，将欧拉角状态转换成四元数状态，更新空速和气动力，再调用：

```python
f = mav._f(x_quat, forces_moments)
```

得到 13 维四元数状态导数。

随后保留位置、速度和角速度导数，将四元数导数替换为以下三个欧拉角运动学公式：

$$
\begin{aligned}
\dot\phi
&=p+q\sin\phi\tan\theta+r\cos\phi\tan\theta\\
\dot\theta
&=q\cos\phi-r\sin\phi\\
\dot\psi
&=\frac{q\sin\phi+r\cos\phi}{\cos\theta}
\end{aligned}
$$

这正是 PDF Slide 3 中的姿态运动学方程。代码直接使用解析公式，等价于 Slide 52 的链式法则转换，不需要显式构造整个转换雅可比矩阵。

---

**3. `df_dx()`、`df_du()`：对应 Slide 37、48、49、52 的线性化**

在配平点 $(x_e^*,\delta^*)$ 附近做一阶泰勒展开：

$$
f_e(x_e^*+\tilde x_e,\delta^*+\tilde\delta)
\approx
f_e(x_e^*,\delta^*)+
A\tilde x_e+B\tilde\delta
$$

其中：

$$
A=
\left.\frac{\partial f_e}{\partial x_e}\right|_*
\in\mathbb R^{12\times12},
\qquad
B=
\left.\frac{\partial f_e}{\partial\delta}\right|_*
\in\mathbb R^{12\times4}
$$

减去名义配平运动的导数后，得到：

$$
\dot{\tilde x}_e=A\tilde x_e+B\tilde\delta
$$

[第 193 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:193) 的 `df_dx()` 使用前向差分，逐列计算 $A$：

$$
A_{:,j}\approx
\frac{
f_e(x_e^*+\varepsilon\,\mathbf e_j,\delta^*)
-f_e(x_e^*,\delta^*)
}{\varepsilon}
$$

这里 $\mathbf e_j$ 表示第 $j$ 个坐标方向，代码取 $\varepsilon=0.01$。

对应代码：

```python
x_eps[i][0] += eps
df_dxi = (f_at_x_eps - f_at_x) / eps
A[:, i] = df_dxi[:, 0]
```

因此，**$A$ 的第 $j$ 列表示：只改变第 $j$ 个状态时，各状态导数如何变化。**

[第 211 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:211) 的 `df_du()` 同理计算：

$$
B_{:,j}\approx
\frac{
f_e(x_e^*,\delta^*+\varepsilon\,\mathbf e_j)
-f_e(x_e^*,\delta^*)
}{\varepsilon}
$$

例如：

$$
B_{10,0}=\frac{\partial\dot q}{\partial\delta_e}
$$

表示升降舵对俯仰角加速度的影响。这里的下标采用代码中的 **从零开始编号**。

---

**4. `compute_ss_model()`：对应 Slide 53、54 的纵向与横向模型**

[第 122 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:122) 先计算完整 $A,B$，再通过选择矩阵提取两个子模型。

纵向状态和输入为：

$$
x_{\rm lon}=[u,w,q,\theta,h]^T,
\qquad
\delta_{\rm lon}=[\delta_e,\delta_t]^T
$$

其中：

$$
h=-p_d
$$

代码中的 $E_1,E_2$ 满足：

$$
x_{\rm lon}=E_1x_e,
\qquad
\delta_{\rm lon}=E_2\delta
$$

所以：

$$
\boxed{
A_{\rm lon}=E_1AE_1^T,\qquad
B_{\rm lon}=E_1BE_2^T
}
$$

这对应：

```python
A_lon = E1 @ A @ E1.T
B_lon = E1 @ B @ E2.T
```

$E_1$ 最后一行在 $p_d$ 位置放置 `-1`，因此会同时正确处理高度状态及高度导数的符号。

横向状态和输入为：

$$
x_{\rm lat}=[v,p,r,\phi,\psi]^T,
\qquad
\delta_{\rm lat}=[\delta_a,\delta_r]^T
$$

类似地：

$$
\boxed{
A_{\rm lat}=E_3AE_3^T,\qquad
B_{\rm lat}=E_3BE_4^T
}
$$

最终矩阵维度为：

| 矩阵 | 维度 | 含义 |
|---|---:|---|
| $A_{\rm lon}$ | $5\times5$ | 纵向状态之间的影响 |
| $B_{\rm lon}$ | $5\times2$ | 升降舵、油门对纵向运动的影响 |
| $A_{\rm lat}$ | $5\times5$ | 横向状态之间的影响 |
| $B_{\rm lat}$ | $5\times2$ | 副翼、方向舵对横向运动的影响 |

这种提取对应教材在直线配平附近采用的纵向、横向解耦近似。

---

**5. `compute_tf_model()`：对应滚转、俯仰和空速传递函数**

[第 88 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:88) 的这个函数返回的是**传递函数系数**，没有创建传递函数对象。

**滚转部分对应 Slide 22～24。**

忽略耦合扰动后：

$$
\ddot\phi=-a_{\phi1}\dot\phi+a_{\phi2}\delta_a
$$

于是：

$$
\frac{\Phi(s)}{\Delta_a(s)}
=
\frac{a_{\phi2}}{s(s+a_{\phi1})}
$$

在直线配平附近，$\dot\phi\approx p$，所以：

$$
a_{\phi1}
=-\frac{\partial\dot p}{\partial p},
\qquad
a_{\phi2}
=\frac{\partial\dot p}{\partial\delta_a}
$$

这解释了：

```python
a_phi1 = -A.item(9, 9)
a_phi2 = B.item(9, 1)
```

因为 $x_e[9]=p$，而输入第 1 项是副翼。

**俯仰部分对应 Slide 29～31。**

简化模型为：

$$
\ddot\theta
=-a_{\theta1}\dot\theta
-a_{\theta2}\theta
+a_{\theta3}\delta_e+d_{\theta2}
$$

忽略扰动后：

$$
\frac{\Theta(s)}{\Delta_e(s)}
=
\frac{a_{\theta3}}
{s^2+a_{\theta1}s+a_{\theta2}}
$$

代码计算：

$$
\begin{aligned}
a_{\theta1}
&=-\frac{\partial\dot q}{\partial q}\\
a_{\theta2}
&=-\frac{\rho V_a^{*2}Sc}{2J_y}C_{m_\alpha}\\
a_{\theta3}
&=\frac{\partial\dot q}{\partial\delta_e}
\end{aligned}
$$

对应：

```python
a_theta1 = -A.item(10, 10)
a_theta2 = -rho * Va_trim**2 * S * c * C_m_alpha / (2 * Jy)
a_theta3 = B.item(10, 0)
```

**为什么 `a_theta2` 没有直接从 $A$ 提取？**

PDF 推导用到了：

$$
\alpha=\theta-\gamma
$$

把迎角力矩中的 $\theta$ 项留下，将航迹角等影响并入扰动。完整状态空间里，$\theta$ 和机体系速度是独立状态，因此 $\partial\dot q/\partial\theta$ 并不直接等于简化模型中的 $-a_{\theta2}$。

**空速部分对应 Slide 35、36。**

配平附近的扰动方程为：

$$
\dot{\tilde V}_a
=-a_{V1}\tilde V_a
+a_{V2}\tilde\delta_t
-a_{V3}\tilde\theta
$$

系数为：

$$
\boxed{
a_{V1}
=
\frac{\rho V_a^*S}{m}
\left(
C_{D_0}+C_{D_\alpha}\alpha^*
+C_{D_{\delta_e}}\delta_e^*
\right)
-\frac1m
\left.\frac{\partial T_p}{\partial V_a}\right|_*
}
$$

$$
\boxed{
a_{V2}
=\frac1m
\left.\frac{\partial T_p}{\partial\delta_t}\right|_*,
\qquad
a_{V3}=g\cos(\theta^*-\alpha^*)
}
$$

它们分别表示空速自身的响应、油门控制能力，以及俯仰变化带来的重力影响。对应传递函数：

$$
\frac{\tilde V_a(s)}{\tilde\Delta_t(s)}
=\frac{a_{V2}}{s+a_{V1}},
\qquad
\frac{\tilde V_a(s)}{\tilde\Theta(s)}
=-\frac{a_{V3}}{s+a_{V1}}
$$

这里使用教材的线性阻力近似；完整状态空间矩阵直接对非线性模型求导，两种模型的系数不必完全相同。

---

**6. `dT_dVa()`、`dT_ddelta_t()`：求推进模型的局部斜率**

[第 233 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:233) 的两个函数分别计算：

$$
\frac{\partial T_p}{\partial V_a}
\approx
\frac{T_p(V_a+\varepsilon,\delta_t)-T_p(V_a,\delta_t)}
{\varepsilon}
$$

$$
\frac{\partial T_p}{\partial\delta_t}
\approx
\frac{T_p(V_a,\delta_t+\varepsilon)-T_p(V_a,\delta_t)}
{\varepsilon}
$$

它们直接调用当前电机—螺旋桨模型，取返回值中的推力 $T_p$，用于前面的 $a_{V1},a_{V2}$。

**`compute_model()` 则负责汇总和保存。** [第 16 行](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:16) 调用上述两条路线，将配平点、模型矩阵和传递函数系数写入 `model_coef.py`。保存的 $A,B$ 都是**连续时间模型**；文件中的 `Ts` 只是记录采样周期，这里没有进行离散化。

### 10.2 PDF 公式与矩阵提取的差别

**提问：** 直接用pdf中的公式计算和从 $A$ 提取的差别大吗？具体有什么差别？我看代码中有的是公式直接计算，有的是直接从 A 中提取

**回答：**

**当前配平点下，能直接对应的系数几乎完全相同；有些系数需要先改变变量、做简化，不能直接取 $A$ 中的某一个元素。** 代码混用这两种方式，主要是因为它们的数学对应关系不同。

我用当前代码重新计算了 $V_a=25\,\mathrm{m/s}$、水平直线飞行的配平点，迎角约为 $0.04970\,\mathrm{rad}$。

**1. 滚转、俯仰的这四项，两种算法基本完全等价。**

[compute_tf_model](/home/zr/mavsim_public/mavsim_python/models/compute_models.py:88) 中的结果如下。这里 $A$ 是状态导数矩阵，$B$ 是控制输入导数矩阵，索引按 Python 从零开始。

| 系数 | PDF 公式直接计算 | 从矩阵提取 | 数值 |
|---|---|---|---:|
| $a_{\phi1}$ | $-\frac{\rho V_aSb^2}{4}C_{p_p}$ | $-A[9,9]$ | 22.628851 |
| $a_{\phi2}$ | $\frac{\rho V_a^2Sb}{2}C_{p_{\delta_a}}$ | $B[9,1]$ | 130.883678 |
| $a_{\theta1}$ | $-\frac{\rho V_aSc^2}{4J_y}C_{m_q}$ | $-A[10,10]$ | 5.294738 |
| $a_{\theta3}$ | $\frac{\rho V_a^2Sc}{2J_y}C_{m_{\delta_e}}$ | $B[10,0]$ | −36.112390 |

两种算法的绝对差都小于 $6\times10^{-14}$，可以视为相同。

原因是，例如：

$$
A[10,10]=\frac{\partial\dot q}{\partial q},
\qquad
B[10,0]=\frac{\partial\dot q}{\partial\delta_e}.
$$

当前动力学模型的俯仰力矩对 $q$、$\delta_e$ 是线性的，对它们求导就直接得到 PDF 的公式。滚转两项也一样。**这里从矩阵提取只是通过数值求导得到相同的系数。**

上述等价关系针对当前直线配平条件；在角速度不为零等条件下，完整模型中的惯性耦合可能给矩阵元素增加额外贡献。

**2. $a_{\theta2}$ 不能直接取 $-A[10,7]$。**

PDF 的俯仰简化模型先使用：

$$
\alpha=\theta-\gamma,\qquad \dot\theta\approx q,
$$

再将航迹角 $\gamma$ 的影响作为扰动，得到：

$$
\ddot\theta+a_{\theta1}\dot\theta+a_{\theta2}\theta
=a_{\theta3}\delta_e+d_\theta,
$$

其中：

$$
a_{\theta2}
=-\frac{\rho V_a^2Sc}{2J_y}C_{m_\alpha}
\approx99.947422.
$$

这是第五章俯仰传递函数推导中的简化。[chap5.pdf](/home/zr/mavsim_public/pdf/chap5.pdf)

但是完整模型的状态里，$u,w,\theta$ 是独立变量。计算

$$
A[10,7]=\left.\frac{\partial\dot q}{\partial\theta}\right|_{u,w,\ldots\ \text{不变}}
$$

时，只改变姿态角，保持机体系速度不变。此时

$$
\alpha=\operatorname{atan2}(w,u)
$$

没有改变，俯仰气动力矩也没有改变，所以当前模型给出：

$$
A[10,7]=0.
$$

**这里的 $99.947422$ 与 $0$ 表示不同的偏导关系，不能把差异解释为计算误差。**

如果希望从 $A$ 得到迎角对应的系数，需要在保持空速不变时，让 $u,w$ 随迎角一起变化：

$$
u=V_a\cos\alpha,\qquad w=V_a\sin\alpha,
$$

因此：

$$
\frac{\partial\dot q}{\partial\alpha}
=-wA[10,3]+uA[10,5],
$$

$$
\boxed{a_{\theta2}\approx wA[10,3]-uA[10,5]}.
$$

随后再使用 PDF 中 $\alpha=\theta-\gamma$ 的简化，才能把它放入俯仰传递函数。代码直接使用公式，正好省去了这个转换过程。

**3. 空速系数的差异，目前很小，但包含模型近似。**

同样不能把 $a_{V1}$ 直接当成 $-A[3,3]$，因为：

$$
A[3,3]=\frac{\partial\dot u}{\partial u},
\qquad
V_a=\sqrt{u^2+v^2+w^2}.
$$

机体系的 $u$ 与空速 $V_a$ 是不同变量。在当前零风条件下，令速度方向单位向量为：

$$
\boldsymbol n=\frac{[u,v,w]^T}{V_a},
$$

保持迎角、侧滑角不变，正确的比较近似为：

$$
a_{V1}^{\mathrm{完整模型}}
\approx-\boldsymbol n^TA[3:6,3:6]\boldsymbol n,
$$

$$
a_{V2}^{\mathrm{完整模型}}
=\boldsymbol n^TB[3:6,3].
$$

实际计算结果：

| 系数 | 当前 PDF 简化公式 | 完整模型经空速方向转换 | 相对差异 |
|---|---:|---:|---:|
| $a_{V1}$ | 0.296135 | 0.295877 | 约 0.09% |
| $a_{V2}$ | 9.550080 | 9.538287 | 约 0.12% |

差异主要来自：

- PDF 空速简化公式使用线性阻力系数；动力学代码使用包含诱导阻力的非线性表达式。
- 完整模型中，推力沿空速方向的分量包含 $\cos\alpha$；简化公式相当于采用 $\cos\alpha\approx1$。
- $A$ 使用步长 $0.01$ 的前向差分，也有少量数值误差。

当前迎角很小，所以这些差异很小。其他配平点需要重新评估。

**4. 系数一致，也不代表完整模型与简化传递函数的响应完全一致。**

例如 PDF 中：

$$
\frac{\phi(s)}{\delta_a(s)}
=\frac{a_{\phi2}}{s(s+a_{\phi1})}.
$$

即使这两个系数与矩阵提取结果完全相同，完整横向模型仍包含侧向速度 $v$、偏航角速度 $r$ 等耦合。完整模型的传递函数需要通过：

$$
G(s)=C(sI-A)^{-1}B+D
$$

计算，阶数和动态响应可能不同。

所以，**当前代码对滚转和俯仰的四个直接偏导系数采用矩阵提取是合理的；$a_{\theta2}$ 和空速系数采用公式，则是在明确使用 PDF 的简化模型。** 若要让这些简化系数与完整动力学严格一致，需要进行相应的变量转换和降阶，不能统一改成取某个矩阵元素。

### 10.3 矩阵提取是否代表未简化

**提问：** 是不是从矩阵中提取的就是未简化的，而用 PDF 公式中的就是简化的？

**回答：**

**不完全是。关键在于保留了哪些状态和耦合关系，不能只看系数是从矩阵提取，还是用公式计算。**

可以把代码分成三个层次理解：

1. **原始非线性动力学模型**
   包含完整的状态、气动力、推进力和运动耦合。

2. **线性化后的 $A,B$ 模型**
   在配平点附近，用一阶偏导近似原始模型：
   $$
   \Delta\dot x=A\Delta x+B\Delta\delta.
   $$
   它已经忽略了二阶及以上的非线性项，因此也经过了近似，但完整的 $A,B$ 仍保留各状态之间的一阶耦合。

3. **PDF 中的简化传递函数**
   进一步忽略部分状态和耦合，只保留主要动态。即使系数取自完整 $A,B$，把它们放进简化传递函数后，得到的仍是简化模型。

例如，完整横向模型中的滚转角速度方程包含：

$$
\Delta\dot p
=A_{pv}\Delta v+A_{pp}\Delta p+A_{pr}\Delta r
+B_{p\delta_a}\Delta\delta_a+B_{p\delta_r}\Delta\delta_r.
$$

代码提取：

$$
a_{\phi1}=-A_{pp},\qquad a_{\phi2}=B_{p\delta_a},
$$

再用于：

$$
\Delta\dot p=-a_{\phi1}\Delta p+a_{\phi2}\Delta\delta_a,
\qquad \Delta\dot\phi\approx\Delta p.
$$

这里已经舍去了侧向速度、偏航角速度和方向舵等影响。**系数来自完整矩阵，但组装出来的滚转模型仍然是简化的。**

另一方面，PDF 的公式也可能恰好是某个偏导数的解析表达式。例如当前配平条件下：

$$
-\frac{\partial\dot q}{\partial q}
=-\frac{\rho V_aSc^2}{4J_y}C_{m_q}
=a_{\theta1}.
$$

用公式直接计算，与从 $A$ 提取，得到的是同一个量。**公式计算本身并不意味着多做了一层简化。**

因此，我之前那句话更准确的表述是：**这四个系数可以从完整线性化模型中提取，而且在当前配平条件下与 PDF 公式等价；它们随后用于构造 PDF 的简化传递函数。**

### 10.4 为什么 PDF 公式与线性化矩阵的某些元素不一致

**提问：** 那为什么 PDF 推导出来的公式有时候会和线性化中的 $A$ 矩阵不一样，所以不能用 $A$ 矩阵的公式？

**回答：**

**不是不能用 $A$，而是不能总把 PDF 中的系数对应到 $A$ 的某一个元素。** PDF 推导和完整模型线性化，有时改变的是不同变量，有时采用了不同近似。

$A$ 中每个元素的含义很严格：

$$
A_{ij}=\left.\frac{\partial f_i}{\partial x_j}\right|_{\text{其他状态不变}}.
$$

而 PDF 在推导传递函数时，可能先规定某些变量之间的关系，再求系数。这样得到的变化率，不一定是上面这个偏导。

用你关注的 $a_{\theta2}$ 举例最清楚。

**在完整 $A$ 中，只改变 $\theta$，保持 $u,w$ 不变。**

零风时：

$$
\alpha=\operatorname{atan2}(w,u).
$$

因为 $u,w$ 不变，迎角不变，当前模型的俯仰气动力矩也不变，所以：

$$
\frac{\partial\dot q}{\partial\theta}=0.
$$

从物理上看，机体系速度不变、姿态改变，会让惯性系中的飞行方向一起改变。此时航迹角 $\gamma$ 也改变了。

**PDF 的俯仰简化模型，则把航迹角变化作为扰动处理。**

在纵向平面中：

$$
\alpha=\theta-\gamma.
$$

考察俯仰角自身的作用、暂时令 $\Delta\gamma=0$，就有：

$$
\Delta\alpha=\Delta\theta.
$$

这时改变 $\theta$ 会改变迎角，进而改变俯仰力矩。因此：

$$
\Delta\dot q
=M_q\Delta q+M_\alpha\Delta\alpha+M_{\delta_e}\Delta\delta_e
$$

变成：

$$
\Delta\dot q
=M_q\Delta q+M_\alpha\Delta\theta
+M_{\delta_e}\Delta\delta_e-M_\alpha\Delta\gamma.
$$

最后一项归入扰动，得到：

$$
a_{\theta2}=-M_\alpha
=-\frac{\rho V_a^2Sc}{2J_y}C_{m_\alpha}.
$$

**所以两者并没有数学矛盾：一个保持机体系速度不变，另一个考察航迹角不变时的俯仰变化。**

如果要通过 $A$ 算出这个系数，就必须让速度分量按相同条件变化。在空速不变、侧滑为零时：

$$
u=V_a\cos\alpha,\qquad w=V_a\sin\alpha,
$$

因此：

$$
M_\alpha
=\frac{\partial\dot q}{\partial u}\frac{\partial u}{\partial\alpha}
+\frac{\partial\dot q}{\partial w}\frac{\partial w}{\partial\alpha}
=-wA[10,3]+uA[10,5].
$$

于是：

$$
\boxed{a_{\theta2}\approx wA[10,3]-uA[10,5]}.
$$

这就是**用 $A$ 得到 PDF 系数的正确方式**，而不是直接取 $-A[10,7]$。

另外还有两类差异：

- **模型近似不同。** PDF 空速推导使用简化阻力表达式，而当前动力学使用包含诱导阻力的表达式，即使比较相同变量的导数，也可能不同。
- **保留的耦合不同。** 完整 $A$ 保留多个状态之间的耦合；PDF 的低阶传递函数会忽略部分耦合。若要得到完整线性模型的传递函数，应计算 $C(sI-A)^{-1}B+D$，不能只取几个元素组装。

因此，判断能否直接提取，要核对三件事：**是不是同一个变化率、保持不变的条件是否相同、使用的动力学近似是否相同。** 对当前的 $a_{\theta1}$ 等四项，这些条件一致，所以公式和矩阵提取相同；对 $a_{\theta2}$，需要先做上述变量转换。

### 10.5 控制器设计模型与仿真真值之间的三层简化链条（第 6 章完成后的补充）

**提问：** 第 6 章自动驾驶仪的增益是按第 5 章的低阶传递函数算出来的，直接用到非线性仿真里，这样设计模型和仿真真值之间是不是有简化？差了多少？

**回答：**

是的，而且是三层简化链条：控制器的设计模型与仿真真值之间隔了"线性化"和"解耦降阶"两层，每一层都丢了东西。这也不是实现上的偷懒——教材第 6 章的方法论本身就是这样：用低阶设计模型凑出增益，再靠鲁棒性容忍模型误差。

把链条完整写出来：

```text
第 1 层  非线性六自由度模型（mav_dynamics_control.py，chap04 实现）
            │  ① 配平点线性化（compute_models.py，eps=0.01 前向差分）
            ▼
第 2 层  12×12 欧拉状态线性模型 A, B（丢失全部二阶及以上非线性）
            │  ② 纵向/横向解耦 + 降阶（E1/E2/E3/E4 选择矩阵 + 低阶传函）
            ▼
第 3 层  设计模型（chap05 的三个标量传函 + chap06 的增益公式）
            φ/δa = aφ2/[s(s+aφ1)]，θ/δe = aθ3/(s²+aθ1s+aθ2)，
            Va/δt = aV2/(s+aV1)，χ/φ ≈ (g/Va)/s
```

每一层丢掉的内容：

- **① 线性化**：丢掉失速混合函数 `σ(α)`、非线性诱导阻力、电机—螺旋桨的强非线性 `C_T(J)`、大姿态角时的运动学非线性。配平点附近小扰动时误差是二阶小量；
- **② 解耦降阶**：丢掉纵向/横向之间的耦合、Phugoid 与高度通道的耦合、`v/r/ψ` 对滚转的牵连。例如滚转设计模型 `aφ2/[s(s+aφ1)]` 只保留 `∂ṗ/∂p` 和 `∂ṗ/∂δa`，扔掉了 `A_pv`、`A_pr` 与方向舵通道；俯仰二阶模型把 `∂q̇/∂θ`（在完整 `A` 里为 0，见 10.2/10.4 的讨论）按 `α=θ-γ` 重构回 `aθ2`，同时把 `γ` 的变化归入扰动。

第 6 章的增益公式（`control_parameters.py`）全部建立在第 3 层上：

$$
k_{p_\phi}=\frac{\omega_{n_\phi}^2}{a_{\phi2}},\qquad
k_{d_\phi}=\frac{2\zeta_\phi\omega_{n_\phi}-a_{\phi1}}{a_{\phi2}},\qquad
k_{p_\chi}=\frac{2\zeta_\chi\omega_{n_\chi}V_g}{g},\qquad
k_{i_\chi}=\frac{\omega_{n_\chi}^2V_g}{g},
$$

$$
k_{p_\theta}=\frac{\omega_{n_\theta}^2-a_{\theta2}}{a_{\theta3}},\qquad
k_{i_h}=\frac{\omega_{n_h}^2}{K_{\theta_{DC}}V_a},\qquad
k_{p_V}=\frac{2\zeta_V\omega_{n_V}-a_{V1}}{a_{V2}},\qquad
k_{i_V}=\frac{\omega_{n_V}^2}{a_{V2}}.
$$

**这次第 6 章调参恰好给了第 3 层误差的一个定量案例。** 初版高度回路按孤立俯仰传函的"5 倍带宽分离"取 $\omega_{n_h}=2.4\,\mathrm{rad/s}$，线性仿真中俯仰通道以 2.52 rad/s 发散。用第 2 层的完整 $A_{\rm lon}$ 做闭环特征值分析（把高度/空速 PI、$K_{\theta_{DC}}$ 前馈与俯仰 PD 一起闭合，增广积分器状态）扫参得到：

| $\omega_{n_h}$ (rad/s) | 0.5 | 1.0 | 1.5 | 2.4 |
|---|---|---|---|---|
| 最差阻尼比（$\omega_{n_V}=2$） | 0.328 | 0.191 | 0.118 | 0.040 |

也就是说：降阶设计模型预测高度回路任意带宽都稳定（它看不到 Phugoid 耦合），而真实闭环在 $\omega_{n_h}=2.4$ 时阻尼只剩 0.04——**被第 ② 层扔掉的 Phugoid 耦合，恰恰是高度外环要激发的那个动态**。把带宽退到 1.0 rad/s 后非线性仿真恢复稳定。这印证了两件事：带宽分离（每层低 5 倍以上，slide 6 的 "factor of 10 or more"）就是给未建模动态留鲁棒性余量的工程手段；但余量有边界，外环带宽逼近内环动态尺度时会撞上降阶误差。

横侧向的对照也支持同一结论：偏航阻尼器把荷兰滚阻尼比从 0.22 提高到 0.59（$\omega_{n_\chi}=0.8$）或 0.48（$\omega_{n_\chi}=1.6$），线性预测与非线性仿真一致——因为荷兰滚动态大部分保留在 $A_{\rm lat}$ 里，第 ② 层在这里丢得少，设计模型就更准。

所以工程上的读法是：**第 2 层的 $A,B$ 用来验证、诊断和划定带宽上限；第 3 层的低阶传函用来快速给增益初值；最终以第 1 层非线性仿真闭环为准**。三层的先后次序和误差容忍，正是 chap05→chap06 教学设计的主线。

### 10.6 配平的目标函数与约束

**提问：** 这个项目中的配平的目标函数和约束分别是什么？

**回答：**

配平在 [models/trim.py](/home/zr/mavsim_public/mavsim_python/models/trim.py) 的 `compute_trim` / `trim_objective_fun` 里，决策变量是 **17 维向量**：`x[0:13]` 为状态 $[p_n,p_e,p_d,u,v,w,e_0,e_1,e_2,e_3,p,q,r]^T$，`x[13:17]` 为输入 $[\delta_e,\delta_a,\delta_r,\delta_t]^T$，用 SLSQP 求解（含一次二次求解打磨残差）。

**目标函数（slide 20）：**

$$
J(x)=\Big\|\,\dot x^{*}-f\big(x,\delta\big)\,\Big\|_{[2:13]}^2,
\qquad
\dot x^{*}=[\,\cdot,\ \cdot,\ -V_a\sin\gamma,\ 0,\ \cdots,\ 0\,]^T
$$

- `f` 就是第 3/4 章的完整非线性动力学：函数内部设 `mav._state = x[0:13]`、`_update_velocity_data()`（零风）、`_forces_moments(δ)`，再调 `mav._f()` 得到 13 维状态导数；
- 期望导数只有第三项非零：$\dot p_d=-V_a\sin\gamma$——因为 $h=-p_d$，以航迹角 $\gamma$ 定速爬升时 $\dot h=V_a\sin\gamma$；
- 范数只取**下标 2 到 12**（$\dot p_d$ 到 $\dot r$），**故意排除 $\dot p_n$ 和 $\dot p_e$**：配平不关心飞到哪，只关心"除水平位置匀速移动外，其余状态导数全为零"；
- 物理含义：**在给定空速 $V_a$ 和航迹角 $\gamma$ 下，找一组状态和舵量使动力学残差最小**。解中 $\dot p_n=V_a\cos\gamma$、$\dot p_e=0$ 自动成立，不进代价。

**约束（8 个等式约束，slide 21，带解析 Jacobian）：**

| 约束 | 含义 |
|---|---|
| $u^2+v^2+w^2-V_a^2=0$ | 空速大小等于指定值 $V_a$ |
| $v=0$ | 侧向速度为零（无侧滑） |
| $e_0^2+e_1^2+e_2^2+e_3^2-1=0$ | 四元数单位长度 |
| $e_1=0,\quad e_3=0$ | 强制滚转和偏航为零（只剩 $e_0,e_2$ ⇒ 纯俯仰姿态） |
| $p=0,\quad q=0,\quad r=0$ | 角速率为零 |

分工：**约束负责"姿态与角运动定常"**，**目标函数负责"平动定常"**（速度不变、高度按 $\gamma$ 变化），剩余自由度（$\alpha,\theta$、四个舵量等）交给优化器压动力学残差。

两个实践细节：骨架原版 `x0` 是二维列向量，scipy 要求一维，已加 `x0.flatten()`；约束集有一个结构性局限——同时强制 $\varphi=0$ 与 $v=0$，但螺旋桨反扭矩 $Q_p\neq0$ 需要 $\delta_a/\delta_r$ 产生配平力矩，而二者又产生侧力，在 $v=0,\varphi=0$ 下无处平衡，因此 $J$ 存在不可消除的下界（$V_a=25,\gamma=0$ 时 SLSQP 与 trust-constr 都收敛到 $J\approx2.69\times10^{-5}$，残差集中在 $\dot v\approx5\times10^{-3}$、$\dot r\approx10^{-3}$）。真正的平衡态应有微小 $\beta\neq0$，教材的约束公式为教学简洁排除了它——这也是第 5 章仿真"保持配平约 30 s 后缓慢漂移"的根源之一。

### 10.7 TECS 控制器：实现、同风场对比与解耦机制（第 6 章补充）

**提问：** 实现教材的 TECS（总能量控制）控制器，与第 6 章的递次回路控制器在相同场景下对比，并定量解释 TECS 为什么能消除高度-速度耦合。

**回答：**

TECS 正是针对 10.5 中发现的"高度/速度能量耦合"设计的。实现细节与增益全部先经第 5 章纵向线性模型的增广闭环特征值分析校核，再进非线性仿真。

**三份交付物**：[outputs/chap6_tecs_comparison.png](/home/zr/mavsim_public/outputs/chap6_tecs_comparison.png)（同一种子阵风下双控制器对比）、[outputs/chap6_tecs_decoupling.png](/home/zr/mavsim_public/outputs/chap6_tecs_decoupling.png)（解耦机制开环实验）、[controllers/autopilot_tecs.py](/home/zr/mavsim_public/mavsim_python/controllers/autopilot_tecs.py)（在 `mavsim_chap6.py` 里取消第 22 行注释即可切换 TECS 跑 GUI）。

**实现**：横侧向与传统版完全相同；纵向换成能量坐标——总能量误差 `E_err = g·e_h + Va·e_V` 驱动油门 PI（E_kp=4e-3、E_ki=1.5e-3，配平油门前馈），能量分配误差 `L_err = g·e_h − Va·e_V` 驱动俯仰指令 PI（L_kp=1.2e-3，限 ±30°，经原有俯仰 PD 环闭合）；高度误差饱和 ±20 m，两个积分器都带回计算抗饱和。增益用第 5 章 A_lon 增广闭环特征值扫参选定（最差阻尼 0.55）。

**同风场对比**（固定随机种子保证阵风逐点相同、同一方波指令 300 s）：

| 指标 | 经典（递次回路） | TECS |
|---|---|---|
| 稳定段 Va 误差 | **0.014 m/s** | 0.228 m/s |
| 稳定段高度误差 | 0.511 m | 0.797 m |
| **能量域综合误差** | 5.08 J/kg | **3.32 J/kg** |
| 冲突段（爬升+减速同时指令）高度误差 | 7.27 m | **6.26 m** |
| 油门饱和占比 | 13.1% | 18.2% |

**如何消除高度-速度耦合——分三步：**

1. **坐标旋转 45°**。经典控制器的被控量是 $(e_h, e_V)$，控制配对 $\delta_t\to e_V$、$\delta_e\to e_h$——但飞机对象在这组坐标上强耦合（正是长周期模态：推力改变速度、速度改变升力、升力改变轨迹……）。TECS 定义 $E=gh+V_a^2/2$（能量之和）、$L=gh-V_a^2/2$（能量之差），相当于把误差平面旋转 45°：$[E;L]=[[g,V_a],[g,-V_a]]\,[e_h;e_V]$。

2. **对象在 (E, L) 坐标上近似对角**。开环阶跃实验（滚转稳定、无风）定量验证：
   - **只动升降舵 −0.01 rad**：8 秒内 L 上升 +55 J/kg 并保持，E 仅 +36 J/kg（α 增大带来的寄生阻力所致）——升降舵是纯粹的**能量分配阀门**（势能⇄动能互换，不产生能量）；
   - **只动油门 +0.15**：E 单调上升（$\dot E=V_a(T-D)/m$，12 秒 +444 J/kg），L 先降 40 再随长周期缓慢漂移——油门是纯粹的**能量注入阀门**，但配分无人管时会随长周期游荡。

   所以 TECS 的配对 $\delta_t\to E$、$\delta_e\to L$ 让每个执行器只对"自己的"被控量有主导权威——**在对象侧实现了对角化**，而不是像经典法那样在指令侧对角化、再靠带宽分离硬压对象耦合。

3. **长周期模态被结构性"拆掉"**。长周期本质是 E 与 L 交换的无阻尼振荡；两个 PI 独立钉住 E 和 L 后，这个振荡没有存在的自由度——这也是 TECS 增益设计比经典法宽容得多的原因（扫参显示大片稳定区，而经典法高度回路带宽超 1.5 rad/s 即失稳，见 10.5 的表格）。

**调试过程**（记录两次迭代）：初版 E_kp=2e-3、E_ki=4e-4 已稳定且油门饱和时间减半（6.3% vs 13.1%），但 Va 跟踪偏松（0.76 vs 0.37 m/s）；按扫参余量加大到 E_kp=4e-3、E_ki=8e-4 反而变差，诊断（Va/θ/γ 轨迹）显示稳态逐位吻合、慢在能量积分器拉入速度；验证 E_ki=1.5e-3 的线性余量（ζ=0.55）后定为最终值。解耦开环实验本身也迭代了三次：第一次被螺旋模态污染（60 s 滚转发散）→ 加滚转稳定；40 s 快照被长周期振荡污染（采样相位恰好落在势能份额上）→ 改看**初始速率段**（前 10 秒：油门直接注入功率 $\dot E=V_a(T-D)/m$，升降舵直接改变 $\dot L$ 而几乎不动 E），机制才显现。

**优势与代价（诚实版）**：优势是冲突工况下高度跟踪更好、能量域综合误差小 35%、增益鲁棒性好、天然免疫"爬升掉速"（速度误差直接进入 E，油门自动补偿）；教材还强调它**不依赖气动模型**（只用能量守恒）。代价是单一变量的拉入速度略慢：推力功率有限时 TECS 把能量同时分给高度和速度（两者都顾），经典控制器牺牲高度保速度——纯 Va 指标上经典更快。这是优先级取舍而非缺陷；实际飞控（如 Boeing 737 的真 TECS）选它，正因为能量域才是物理上正确的控制坐标。

## 11. 全部工作总结（2026-10-15 更新）

> 数据完整了，解耦机制得到定量验证。本节总结自 2026-09-23 首次笔记以来按章节完成的全部实现、验收结果与遗留项；与第 1～9 节历史笔记冲突之处，以本节为准。

### 11.1 已完成实现（按章）

| 章 | 文件 | 内容 |
|---|---|---|
| 3 | `models/mav_dynamics.py` | `_f()` 按式 (3.14)–(3.17) 实现：四元数运动学、体轴动力学、$\Gamma$ 系数角动力学（不含重力，重力在第 4 章并入合力）。`chap3_check.py` 8/8 PASS |
| 4 | `models/mav_dynamics_control.py` | `_update_velocity_data`（风变换/Va/α/β）、`_forces_moments`（式 4.9–4.11 非线性 CL/CD + 稳定系旋转 + 侧向 + 力矩）、`_motor_thrust_torque`（slide 35 闭式二次根 + C_T/C_Q）。`chap4_check.py` 27/27 PASS |
| 4 | `models/wind_simulation.py` | Dryden 阵风（slide 40 原式，含 H_v/H_w 分子零点；Table 1 低空轻湍流参数）；v/w 传函分子取 2 系数避免控制标准型直接馈通 |
| 4 | `viewers/draw_mav.py` | chap02 遗留占位导致 chap03 起无法启动，已按 `draw_spacecraft.py` 机制实现机身/主翼/尾翼/螺旋桨网格 |
| 5 | `models/trim.py` | 目标函数 + 约束 + 初值（见 10.6），SLSQP 双趟打磨；解得 Va=25、γ=0 配平 δe=−0.124、δa=+0.0059、δr=−0.001、δt=0.773、α=2.85° |
| 5 | `models/compute_models.py` | `f_euler`（精确欧拉运动学）、数值雅可比 `df_dx/df_du`（ε=0.01）、E1–E4 选择矩阵投影、`compute_tf_model`（a_phi/a_theta/a_V 系数）、`dT_dVa/dT_dδt`、`compute_model` 写 `model_coef.py`（改为 `__file__` 相对路径） |
| 6 | `parameters/control_parameters.py` | 全部增益按讲义公式由 `model_coef.py` 的线性模型计算（wn_roll=8、wn_course=1.6、wn_pitch=12、wn_altitude=1.0、wn_Va=2.0；偏航阻尼器 p_wo=1.5、k_r=0.15） |
| 6 | `controllers/autopilot.py` | 递次回路闭合（slide 27 结构）：χ→φ_c（PI，±30°）→δa（PD+速率阻尼）；r→washout 偏航阻尼；Va→δt（PI+配平前馈）；h→γ_c（PI，±15°）→K_θDC 前馈→θ_c→δe（PD+配平前馈）。修复骨架 `TransferFunction` import 被注释的 `NameError` |
| 6 | `controllers/autopilot_tecs.py` | TECS 纵向：E_err=g·e_h+Va·e_V→油门 PI（E_kp=4e-3、E_ki=1.5e-3+配平前馈），L_err=g·e_h−Va·e_V→俯仰指令 PI（L_kp=1.2e-3，限 ±30°）；高度误差饱和 ±20 m，双积分器抗饱和；增益由 A_lon 增广闭环扫参选定（最差阻尼 0.55）。横侧向与经典版相同 |

### 11.2 验收与验证结果

- `chap3_check.py`：8/8 PASS。另按 3.2/3.3 完成单轴力/力矩响应与 Jxz 滚转-偏航耦合的无头验证（Jxz=0 时完全解耦）。
- `chap4_check.py`：27/27 PASS。定位并证明原 Case 3 的 fx 期望值与 Case 1/2 在数学上不相容（同一 CD 公式无法同时满足，见对话记录）；改为教材完整式后全部通过。多情形力学验证图 `outputs/chap3_demo_cases.png`、`chap3_altitude_check.png`。
- `chap5_check.py`：Case 1 的 A_lon/B_lon/B_lat、11 个传函系数、f_euler、x_euler/x_quat、dT_dVa/dT_dδt 全部 PASS；A_lat、df_dx、df_du 各剩 1～5 个"解析值恰为 0"的噪声单元（参考值 −1.7e-8～2.6e-5 为作者数值管线指纹，1e-4 相对容差下不可复现）。Case 2 的姿态块偏差源于作者参考值混用 slide 38/43 的简化 Va 约定，且其 A_euler 与 A_lat 对同一元素给出 −1.3655 与 −1.2277 两个矛盾值（E3 投影不可能改变元素值），已证明参考文件内部不一致。
- 第 5 章配平保持仿真：0.2 m/s 缓慢下沉（对比第 4 章未配平输入 2.4 m/s），约 30 s 后螺旋模态漂移（见 10.6 的约束集局限）。图 `outputs/chap5_trim_hold.png`。
- 第 6 章闭环：40 m 高度阶跃 10～15 s 收敛（稳定段误差 1.37 m）；空速阶跃稳定段误差 0.12 m/s；90° 航向阶跃约 12 s 完成，±180° 穿越正确回绕；舵面无持续饱和（δe ±23°、δa ±21°、δr ±1.6°）。图 `outputs/chap6_tracking.png`、GUI 数据图 `outputs/ch6_data_plot.png`。
- 第 6 章增益设计方法：用第 5 章 A_lon/A_lat 做闭环特征值扫参定带宽（高度外环 ω=2.4 时最差阻尼比 0.04 导致俯仰通道 2.52 rad/s 发散，退到 1.0 后恢复；偏航阻尼器使荷兰滚阻尼 0.22→0.48/0.59）。三层简化链条（非线性真值→线性化→解耦降阶）的讨论见 10.5。
- 第 6 章 TECS 同风场对比（300 s 方波、固定种子）：能量域综合误差 3.32 vs 5.08 J/kg（小 35%）、冲突段高度误差 6.26 vs 7.27 m；代价是稳定段 Va 误差 0.228 vs 0.014 m/s（能量在高度/速度间分配）。解耦机制开环实验：升降舵 −0.01 rad 使 L +55 J/kg 而 E 仅 +36 J/kg（纯分配阀门），油门 +0.15 使 E +444 J/kg（纯注入阀门）。讨论见 10.7。

### 11.3 工具链修复与约定

- `plotter/plotter.py`：坐标轴关闭 SI 前缀自动缩放（`enableAutoSIPrefix(False)`），数据图不再显示 ×10⁻³；
- `controllers/autopilot.py`：恢复被注释的 `TransferFunction` import（否则实例化即 `NameError`）；
- 所有 `mavsim_chap*.py` 的数据图统一存 `outputs/`（`save_plots=True`），该目录被 `.gitignore` 的 `*.png` 规则覆盖；
- 演示统一用仓库 `.venv`（系统 python3 缺 pyqtgraph）；chap03 演示注意事项见前文。

### 11.4 遗留项

- `autopilot_lqr.py`：教材的另一种替代方案（增广状态 LQR+CARE），未实现；TECS 已完成并与经典法对比（见 10.7），主入口链路默认用 `autopilot.py`，取消 `mavsim_chap6.py` 第 22 行注释即切换 TECS；
- chap05 启动文件中注释掉的模态激励（升降舵脉冲、副翼/方向舵 doublet）为 5.2(c) 的演示活动，逐个打开即可观察 Phugoid/滚转-螺旋/Dutch-roll；
- 第 7 章及以后（传感器、估计器、路径层）仍为原始骨架，按第 9 节建议的章节顺序继续。
