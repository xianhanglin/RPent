Core Tool Migration
===================

LIBERO, RoboTwin, and RoboCasa now publish ``move_to``, ``rotate_wrist``,
``set_gripper``, ``release``, and ``view_env_state`` with the shared
conventions in ``rpent.tools.core`` (see :ref:`shared-core-tools`). A call
that worked on one of these robots now means the same thing on the others.
This page lists what changed and how to translate recipes, Flash plans,
transcripts, and memory notes recorded with the earlier tools.

What Changed
------------

**Gripper.** Motion tools take ``gripper`` as ``"hold"``, ``"open"``, or
``"close"``, and the default is ``"hold"``. ``set_gripper`` takes a required
``command`` (``"open"`` or ``"close"``); RoboTwin also accepts an opening
fraction from 0 (closed) to 1 (open). ``release`` always opens fully.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Robot
     - Before
     - Now
   * - LIBERO
     - ``gripper`` -1 open, +1 close. ``move_to`` and ``move_pose`` defaulted
       to -1, so omitting it while carrying dropped the object;
       ``rotate_wrist`` and ``rotate_pitch`` defaulted to +1.
       ``set_gripper(gripper=-1)`` by default.
     - ``"hold"`` sends the native ``0`` command, which keeps the last gripper
       command. ``set_gripper(command=...)``.
   * - RoboTwin
     - ``gripper`` and ``val`` from 0 to 1 where **1 is open**; ``None`` kept
       the target. ``set_gripper(val=...)``, ``release(val=1.0)``.
     - ``"open"`` is 1, ``"close"`` is 0, ``"hold"`` keeps the target.
       ``set_gripper(command=...)``; ``release`` has no ``val``.
   * - RoboCasa
     - ``gripper`` -1 open, +1 close, or ``"hold"`` (default for moves).
       ``rotate_pitch`` defaulted to +1, ``set_gripper`` to +1.
     - Same hold servo. ``rotate_pitch`` defaults to ``"hold"``;
       ``set_gripper(command=...)``.

**Rotation.** ``rotate_wrist`` takes ``delta_yaw`` or ``target_yaw`` in
radians about world Z on LIBERO and RoboTwin. RoboTwin previously took
``delta_yaw_deg`` in degrees. RoboTwin ``move_to`` also accepts
``target_yaw``; its ``quat`` argument remains for full orientations. RoboCasa
has no ``rotate_wrist``.

**Step budgets.** LIBERO ``release`` takes ``steps`` instead of ``max_steps``.
RoboCasa ``view_env_state`` takes ``step=-1`` for the latest entry instead of
``null``.

**Results.** Motion and gripper tools return the same keys on every robot:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``ok``
     - The commanded goal was reached. For gripper tools, the command ran for
       its full step budget (LIBERO ``release`` also counts the episode
       terminating).
   * - ``stop_reason``
     - ``reached``, ``not_reached``, ``plan_failed``, ``terminated``, or
       ``truncated``. An episode ending takes precedence.
   * - ``steps_used``
     - Native environment steps executed.
   * - ``eef_xyz``
     - Final world-frame end-effector position, meters.
   * - ``gripper_opening``
     - Final opening fraction from 0 (closed) to 1 (open). LIBERO and RoboCasa
       estimate it from finger joint positions.
   * - ``terminated`` / ``truncated``
     - Native episode flags.
   * - ``dist_m`` / ``yaw_error``
     - Remaining position error (m) and wrapped yaw error (rad), when the tool
       has that target.

Robot-specific keys remain alongside them, such as ``position_hold_ok`` on
LIBERO rotations and ``native_stop_reason`` and ``plan_status`` on RoboTwin.
Renamed keys include LIBERO ``final_dist_m`` → ``dist_m``,
``final_yaw_error_rad`` → ``yaw_error``, and ``converged`` → ``ok``;
RoboTwin ``final_eef_xyz`` → ``eef_xyz`` and ``gripper_val`` →
``gripper_opening``; RoboCasa ``final_dist`` → ``dist_m``, ``eef`` →
``eef_xyz``, and ``steps`` → ``steps_used``. RoboTwin ``move_to`` used to
report ``success: true`` once a path was planned; ``ok`` now also requires
the arm to finish within ``tol`` of the target.

``view_env_state`` returns ``step``, ``terminated``, ``truncated``,
``success``, ``task_language``, ``state``, ``artifacts``, ``log``, and
``images`` on every robot. ``images`` names the camera and role of each
embedded image in order. RoboCasa's ``robocasa_terminated`` is now
``terminated``.

Translate Recipes and Flash Plans
---------------------------------

``rpent.tools.core.migrate_legacy_call`` translates one recorded call.
It reads an omitted gripper as the earlier default, so apply it only to calls
recorded before this change. The module also rewrites recipe JSONL files and
Flash ``*_plan.json`` files:

.. code-block:: bash

   python -m rpent.tools.core --robot libero --output-dir migrated/ \
       memory/libero/*_recipe.jsonl

Without ``--output-dir`` the files are replaced in place. Flash plans that the
command rewrites gain ``"tool_spec": 1`` and are not translated again. LIBERO
Flash replay translates plans without ``tool_spec`` when it loads them, so the
published replay corpus keeps working unchanged. To build a new Flash plan from
a recipe recorded before this change, pass ``--legacy-tool-args`` to
``python -m robots.libero.flash.generate``.

A RoboTwin gripper target other than 0 or 1 passed to ``move_to`` cannot be
expressed as a motion gripper mode; the translation rounds it to ``"open"``
at 0.5 and above, otherwise ``"close"``. A LIBERO ``set_gripper(gripper=0)``
has no equivalent and raises an error.

Memory Notes and Transcripts
----------------------------

Markdown memory notes and planner transcripts are free text, so they are not
rewritten automatically. Notes in the published memory corpora describe the
earlier arguments, for example "carry with ``gripper:+1``". A planner that
copies such a call now gets a validation error naming the accepted values.
Update notes you maintain by replacing:

- ``gripper: +1`` / ``-1`` with ``"close"`` / ``"open"``, or omitting it while
  carrying;
- ``set_gripper(gripper=...)`` / ``set_gripper(val=...)`` with
  ``set_gripper(command=...)``;
- RoboTwin ``delta_yaw_deg: D`` with ``delta_yaw`` in radians;
- references to the renamed result keys above.

Evaluation results that used the earlier tools and the earlier published
memory are not directly comparable to runs with these tools, because the
model-facing schemas and default gripper behavior changed.
