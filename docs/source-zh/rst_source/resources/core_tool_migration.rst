核心工具迁移
============

LIBERO、RoboTwin 和 RoboCasa 现在按照 ``rpent.tools.core`` 中的共享约定（见 :ref:`shared-core-tools`）发布 ``move_to``、``rotate_wrist``、``set_gripper``、``release`` 和 ``view_env_state``。在其中一个机器人上有效的调用，在其他机器人上含义相同。本页列出具体变化，并说明如何转换按旧工具记录的 recipe、Flash plan、对话记录和记忆笔记。

变化内容
--------

**夹爪。** 运动工具的 ``gripper`` 取 ``"hold"``、``"open"`` 或 ``"close"``，默认值为 ``"hold"``。``set_gripper`` 需要必填参数 ``command``（``"open"`` 或 ``"close"``）；RoboTwin 还接受 0（闭合）到 1（全开）之间的开度比例。``release`` 总是完全打开夹爪。

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - 机器人
     - 之前
     - 现在
   * - LIBERO
     - ``gripper`` 为 -1 打开、+1 闭合。``move_to`` 和 ``move_pose`` 默认 -1，因此搬运时省略该参数会让物体掉落；``rotate_wrist`` 和 ``rotate_pitch`` 默认 +1。``set_gripper`` 默认 ``gripper=-1``。
     - ``"hold"`` 发送原生命令 ``0``，即保持上一次夹爪命令。使用 ``set_gripper(command=...)``。
   * - RoboTwin
     - ``gripper`` 和 ``val`` 取 0 到 1，其中 **1 表示打开**；``None`` 保持当前目标。使用 ``set_gripper(val=...)`` 和 ``release(val=1.0)``。
     - ``"open"`` 对应 1，``"close"`` 对应 0，``"hold"`` 保持当前目标。使用 ``set_gripper(command=...)``；``release`` 不再有 ``val``。
   * - RoboCasa
     - ``gripper`` 为 -1 打开、+1 闭合，或 ``"hold"``（运动工具的默认值）。``rotate_pitch`` 和 ``set_gripper`` 默认 +1。
     - 保持方式不变。``rotate_pitch`` 默认 ``"hold"``；使用 ``set_gripper(command=...)``。

**旋转。** 在 LIBERO 和 RoboTwin 上，``rotate_wrist`` 接受 ``delta_yaw`` 或 ``target_yaw``，单位为弧度，绕世界 Z 轴旋转。RoboTwin 之前使用以角度为单位的 ``delta_yaw_deg``。RoboTwin 的 ``move_to`` 也接受 ``target_yaw``，``quat`` 参数仍用于指定完整姿态。RoboCasa 没有 ``rotate_wrist``。

**步数预算。** LIBERO 的 ``release`` 改用 ``steps`` 代替 ``max_steps``。RoboCasa 的 ``view_env_state`` 用 ``step=-1`` 表示最新记录，不再使用 ``null``。

**结果。** 运动和夹爪工具在所有机器人上返回相同的字段：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 含义
   * - ``ok``
     - 是否达到命令目标。对夹爪工具而言，表示命令执行满了步数预算（LIBERO 的 ``release`` 在回合终止时也算达到）。
   * - ``stop_reason``
     - ``reached``、``not_reached``、``plan_failed``、``terminated`` 或 ``truncated``。回合结束优先于其他原因。
   * - ``steps_used``
     - 实际执行的原生环境步数。
   * - ``eef_xyz``
     - 末端执行器最终的世界坐标位置，单位为米。
   * - ``gripper_opening``
     - 最终开度比例，0 为闭合、1 为全开。LIBERO 和 RoboCasa 根据手指关节位置估算。
   * - ``terminated`` / ``truncated``
     - 原生回合标志。
   * - ``dist_m`` / ``yaw_error``
     - 工具有对应目标时，剩余的位置误差（米）和归一化后的 yaw 误差（弧度）。

机器人特有的字段仍然保留，例如 LIBERO 旋转工具的 ``position_hold_ok``，以及 RoboTwin 的 ``native_stop_reason`` 和 ``plan_status``。改名的字段包括：LIBERO 的 ``final_dist_m`` → ``dist_m``、``final_yaw_error_rad`` → ``yaw_error``、``converged`` → ``ok``；RoboTwin 的 ``final_eef_xyz`` → ``eef_xyz``、``gripper_val`` → ``gripper_opening``；RoboCasa 的 ``final_dist`` → ``dist_m``、``eef`` → ``eef_xyz``、``steps`` → ``steps_used``。RoboTwin 的 ``move_to`` 之前只要规划出路径就返回 ``success: true``；现在 ``ok`` 还要求机械臂最终停在目标 ``tol`` 范围内。

``view_env_state`` 在所有机器人上返回 ``step``、``terminated``、``truncated``、``success``、``task_language``、``state``、``artifacts``、``log`` 和 ``images``。``images`` 按顺序给出每张嵌入图像的相机和用途。RoboCasa 的 ``robocasa_terminated`` 改为 ``terminated``。

转换 recipe 和 Flash plan
-------------------------

``rpent.tools.core.migrate_legacy_call`` 转换一次已记录的调用。它把省略的夹爪参数按旧默认值处理，因此只应用于本次变更之前记录的调用。该模块还可以改写 recipe JSONL 文件和 Flash ``*_plan.json`` 文件：

.. code-block:: bash

   python -m rpent.tools.core --robot libero --output-dir migrated/ \
       memory/libero/*_recipe.jsonl

不指定 ``--output-dir`` 时会原地替换文件。经过改写的 Flash plan 会带上 ``"tool_spec": 1``，之后不会再次转换。LIBERO Flash 回放在加载没有 ``tool_spec`` 的 plan 时会自动转换，因此已发布的回放语料无需改动即可继续使用。如果要用本次变更之前记录的 recipe 生成新的 Flash plan，请给 ``python -m robots.libero.flash.generate`` 传入 ``--legacy-tool-args``。

RoboTwin ``move_to`` 中取值不是 0 或 1 的夹爪目标无法表示为运动工具的夹爪模式；转换时 0.5 及以上取 ``"open"``，否则取 ``"close"``。LIBERO 的 ``set_gripper(gripper=0)`` 没有对应写法，转换时会报错。

记忆笔记和对话记录
------------------

Markdown 记忆笔记和 planner 对话记录是自由文本，不会自动改写。已发布记忆语料中的笔记描述的是旧参数，例如“用 ``gripper:+1`` 搬运”。planner 照抄这类调用时，会收到列出合法取值的参数校验错误。请按以下方式更新你维护的笔记：

- 把 ``gripper: +1`` / ``-1`` 改为 ``"close"`` / ``"open"``，或者在搬运时直接省略；
- 把 ``set_gripper(gripper=...)`` / ``set_gripper(val=...)`` 改为 ``set_gripper(command=...)``；
- 把 RoboTwin 的 ``delta_yaw_deg: D`` 改为以弧度为单位的 ``delta_yaw``；
- 更新对上文改名结果字段的引用。

使用旧工具和旧版已发布记忆得到的评测结果，与使用这些工具的运行结果不能直接比较，因为模型可见的 schema 和默认夹爪行为都发生了变化。
