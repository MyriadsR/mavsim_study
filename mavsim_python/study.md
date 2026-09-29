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
