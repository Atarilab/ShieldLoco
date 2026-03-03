import argparse
import time
import mujoco
import mujoco.viewer
from mujoco import mjx
import jax
import numpy as np
import os
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_deterministic_ops=true "
    "--xla_gpu_autotune_level=0"
)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"


class MetaData:
    """Full MjLab config for Go2"""

    # ------------- Physics -------------

    _JACOBIAN_MAP = {
    "auto": mujoco.mjtJacobian.mjJAC_AUTO, # type: ignore
    "dense": mujoco.mjtJacobian.mjJAC_DENSE, # type: ignore
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE, # type: ignore
    }
    _CONE_MAP = {
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, # type: ignore
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, # type: ignore
    } 
    _INTEGRATOR_MAP = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER, # type: ignore
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, # type: ignore
    }
    _SOLVER_MAP = {
    "newton": mujoco.mjtSolver.mjSOL_NEWTON, # type: ignore
    "cg": mujoco.mjtSolver.mjSOL_CG, # type: ignore
    "pgs": mujoco.mjtSolver.mjSOL_PGS, # type: ignore
    }


    timestep: float = 0.01
    iterations: int = 8
    ls_iterations: int = 8
    ls_tolerance: float = 1e-5
    tolerance: float = 1e-8
    integrator: str = "implicitfast"
    impratio: float = 1.0
    cone: str = "pyramidal"
    jacobian: str = "auto"
    solver: str = "newton"

    decimation: int = 2

meta_data = MetaData()

# test loading of xml (mjx type)

mj_model = mujoco.MjModel.from_xml_path('robots/go2/scene_mjx.xml')

# Set MJX physics for all meta_data
mj_model.opt.timestep = meta_data.timestep
mj_model.opt.integrator = meta_data._INTEGRATOR_MAP[meta_data.integrator]
mj_model.opt.impratio = meta_data.impratio
mj_model.opt.cone = meta_data._CONE_MAP[meta_data.cone]
mj_model.opt.jacobian = meta_data._JACOBIAN_MAP[meta_data.jacobian]
mj_model.opt.solver = meta_data._SOLVER_MAP[meta_data.solver]
mj_model.opt.iterations = meta_data.iterations
mj_model.opt.tolerance = meta_data.tolerance
mj_model.opt.ls_iterations = meta_data.ls_iterations
mj_model.opt.ls_tolerance = meta_data.ls_tolerance

mj_data = mujoco.MjData(mj_model)

model = mjx.put_model(mj_model, impl="warp")

data = mjx.make_data(
        mj_model,
        impl="warp",
        nconmax=1000,
        njmax=1000,
)

# jit_step = jax.jit(mjx.step)

# # test forward pass
# for _ in range(10):
#     start = time.time()

#     data = jit_step(model, data)

#     print(f"Step took {1000 * (time.time() - start):.2f} ms")


import numpy as np
ngeom = mj_model.ngeom
geom_is_hazard_np = np.zeros(ngeom, dtype=bool)
geom_is_robot_np = np.zeros(ngeom, dtype=bool)

# Make it xml specific remember!
hazard_prefixes = ("obstacle_",)
base_link_id = mj_model.body("base_link").id

for gid in range(ngeom):
    # -- HAZARD CHECK --
    # Handle the case where a geom might not have a name (returns None or empty)
    geom_name = mj_model.geom(gid).name or ""

    if any(geom_name.startswith(p) for p in hazard_prefixes):
        geom_is_hazard_np[gid] = True

    # -- ROBOT CHECK --
    # Find which body this geom belongs to
    body_id = mj_model.geom_bodyid[gid]

    # Check if this body's ultimate parent is the base_link
    if mj_model.body_rootid[body_id] == base_link_id:
        geom_is_robot_np[gid] = True

print("Hazard geoms:", np.where(geom_is_hazard_np)[0])
print("Robot geoms:", np.where(geom_is_robot_np)[0])


def run_viewer_rollout(mj_model, mj_data, model, data, num_steps=500):
    """Run dynamics rollout and visualize in MuJoCo viewer."""
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    
    jit_step = jax.jit(mjx.step)
    
    print(f"Running {num_steps} steps with zero torques...")
    for step in range(num_steps):
        data = jit_step(model, data)
                
        mj_data.qpos[:] = np.asarray(data.qpos)
        mj_data.qvel[:] = np.asarray(data.qvel)
        
        mujoco.mj_forward(mj_model, mj_data)
        
        viewer.sync()
        
        if step % 50 == 0:
            print(f"Step {step}/{num_steps}")
        
        if not viewer.is_running():
            print("Viewer closed, exiting...")
            break
    
    viewer.close()
    print("Rollout complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true", help="Enable MuJoCo viewer visualization")
    args = parser.parse_args()
    
    if args.viewer:
        run_viewer_rollout(mj_model, mj_data, model, data)
    else:
        print("Use --viewer flag to visualize dynamics rollout")