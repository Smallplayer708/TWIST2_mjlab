"""mjlab (MuJoCo / mujoco_warp) system-identification port.

Modules
-------
constants             G1 joint names, gains, action scale, KNEES_BENT pose, ranges
space                 bounded parameter space (susp / joint / softbase)
excitation            excitation signals + safety limiting
build_sysid_xml       serialise the mjlab G1 entity (with actuators) to a sysid MJCF
plant                 plain-MuJoCo rollout (position actuators) + gradient-free fit
fit                   staged CLI fitter
record_real_sysid     real excitation/recording via unitree_interface
apply_params          write fitted MJCF + review-only DR patch

See README.md for the workflow and the mujoco_warp batching note.
"""
