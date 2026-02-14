# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import tempfile
import ctypes
import sys
from typing import Union, Any

# ==============================================================================
# Linux Library Pre-loading Fix
# Resolves: undefined symbol: _ZdlPvm, version Qt_5
# ==============================================================================
if sys.platform == "linux":
    try:
        # Try to find libstdc++.so.6 in the comfy-env environment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Environment is usually in the parent node directory: ComfyUI-UltraShape1/_env_ultrashape1/
        # Current file: .../nodes/ultrashape/postprocessors.py
        env_root = os.path.join(current_dir, "../..") 
        
        env_dir = None
        if os.path.exists(env_root):
             for d in os.listdir(env_root):
                 if d.startswith("_env_") and os.path.isdir(os.path.join(env_root, d)):
                     env_dir = os.path.join(env_root, d)
                     break
        
        if env_dir:
            # We need to pre-load libstdc++ AND all bundled Qt5 libraries
            # because transitive dependencies (like QtXml) can still leak to /usr/lib
            libs_to_load = [os.path.join(env_dir, "lib/libstdc++.so.6")]
            
            # Find all Qt5 libraries in the pymeshlab bundle
            pymeshlab_lib_dir = os.path.join(env_dir, "lib/python3.10/site-packages/pymeshlab/lib")
            if os.path.exists(pymeshlab_lib_dir):
                for f in os.listdir(pymeshlab_lib_dir):
                    if f.startswith("libQt5") and f.endswith(".so.5"):
                        libs_to_load.append(os.path.join(pymeshlab_lib_dir, f))
            
            # RTLD_DEEPBIND (0x8) forces the linker to look in the library's
            # own scope before the global scope. This is essential to
            # resolve the @Qt_5 versioned symbols when other Qt versions
            # are already loaded in the process.
            mode = ctypes.RTLD_GLOBAL
            if hasattr(os, "RTLD_DEEPBIND"):
                mode |= os.RTLD_DEEPBIND
            
            # Explicit priority to satisfy dependency chains: 
            # Core/DBus -> Gui -> Widgets -> OpenGL/Svg
            priority = {
                "libstdc++.so.6": 0,
                "libQt5Core.so.5": 1,
                "libQt5DBus.so.5": 2,
                "libQt5Gui.so.5": 3,
                "libQt5Widgets.so.5": 4,
                "libQt5XcbQpa.so.5": 5,
                "libQt5Xml.so.5": 6
            }
            
            def sort_key(path):
                name = os.path.basename(path)
                return priority.get(name, 99), name

            libs_to_load.sort(key=sort_key) 

            for lib_path in libs_to_load:
                if os.path.exists(lib_path):
                    try:
                        # Pre-load the library into the process
                        ctypes.CDLL(lib_path, mode=mode)
                        if os.path.basename(lib_path) in priority:
                            print(f"[UltraShape] Pre-loaded environment library: {os.path.basename(lib_path)}")
                    except Exception as e:
                        # Only report failures for priority libs
                        if os.path.basename(lib_path) in priority:
                            print(f"[UltraShape] Warning: Failed to pre-load {os.path.basename(lib_path)}: {e}")
    except Exception as e:
        print(f"[UltraShape] Warning: Failed to pre-load environment libraries: {e}")

import numpy as np
import torch
import trimesh

# pymeshlab is optional - used for mesh simplification/post-processing
try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError as e:
    HAS_PYMESHLAB = False
    pymeshlab = None
    print(f"[UltraShape] pymeshlab not found: {e}. Mesh post-processing features will be limited.")
except Exception as e:
    HAS_PYMESHLAB = False
    pymeshlab = None
    print(f"[UltraShape] Error loading pymeshlab: {e}. Mesh post-processing features will be limited.")

from typing import Any
from .models.autoencoders import Latent2MeshOutput
from .utils import synchronize_timer


def load_mesh(path):
    if path.endswith(".glb") or not HAS_PYMESHLAB:
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: "pymeshlab.MeshSet", max_facenum: int = 200000):
    if max_facenum > mesh.current_mesh().face_number():
        return mesh

    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: "pymeshlab.MeshSet"):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: "pymeshlab.MeshSet"):
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        mesh.save_current_mesh(temp_file.name)
        mesh = trimesh.load(temp_file.name)
    if isinstance(mesh, trimesh.Scene):
        combined_mesh = trimesh.Trimesh()
        for geom in mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
        mesh = combined_mesh
    return mesh


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    if not HAS_PYMESHLAB:
        raise RuntimeError("pymeshlab is not available")
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
        if isinstance(mesh, trimesh.scene.Scene):
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
        mesh.export(temp_file.name)
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(temp_file.name)
    return mesh


def export_mesh(input, output):
    if HAS_PYMESHLAB and isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union["pymeshlab.MeshSet", trimesh.Trimesh, Latent2MeshOutput, str]) -> "pymeshlab.MeshSet":
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        if not HAS_PYMESHLAB:
             raise RuntimeError("pymeshlab is not available")
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh


class FaceReducer:
    @synchronize_timer('FaceReducer')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 40000
    ) -> Union[trimesh.Trimesh, Latent2MeshOutput]:
        if not HAS_PYMESHLAB:
            print("[UltraShape] FaceReducer skipped - pymeshlab not available")
            if isinstance(mesh, str):
                return trimesh.load(mesh)
            return mesh
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


class FloaterRemover:
    @synchronize_timer('FloaterRemover')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[trimesh.Trimesh, Latent2MeshOutput]:
        if not HAS_PYMESHLAB:
            print("[UltraShape] FloaterRemover skipped - pymeshlab not available")
            if isinstance(mesh, str):
                return trimesh.load(mesh)
            return mesh
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    @synchronize_timer('DegenerateFaceRemover')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[trimesh.Trimesh, Latent2MeshOutput]:
        if not HAS_PYMESHLAB:
            print("[UltraShape] DegenerateFaceRemover skipped - pymeshlab not available")
            if isinstance(mesh, str):
                return trimesh.load(mesh)
            return mesh
        ms = import_mesh(mesh)

        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as temp_file:
            ms.save_current_mesh(temp_file.name)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file.name)

        mesh = export_mesh(mesh, ms)
        return mesh


def mesh_normalize(mesh):
    """
    Normalize mesh vertices to sphere
    """
    scale_factor = 1.2
    vtx_pos = np.asarray(mesh.vertices)
    max_bb = (vtx_pos - 0).max(0)[0]
    min_bb = (vtx_pos - 0).min(0)[0]

    center = (max_bb + min_bb) / 2

    scale = torch.norm(torch.tensor(vtx_pos - center, dtype=torch.float32), dim=1).max() * 2.0

    vtx_pos = (vtx_pos - center) * (scale_factor / float(scale))
    mesh.vertices = vtx_pos

    return mesh


class MeshSimplifier:
    def __init__(self, executable: str = None):
        if executable is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            executable = os.path.join(CURRENT_DIR, "mesh_simplifier.bin")
        self.executable = executable

    @synchronize_timer('MeshSimplifier')
    def __call__(
        self,
        mesh: Union[trimesh.Trimesh],
    ) -> Union[trimesh.Trimesh]:
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_input:
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as temp_output:
                mesh.export(temp_input.name)
                os.system(f'{self.executable} {temp_input.name} {temp_output.name}')
                ms = trimesh.load(temp_output.name, process=False)
                if isinstance(ms, trimesh.Scene):
                    combined_mesh = trimesh.Trimesh()
                    for geom in ms.geometry.values():
                        combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
                    ms = combined_mesh
                ms = mesh_normalize(ms)
                return ms
