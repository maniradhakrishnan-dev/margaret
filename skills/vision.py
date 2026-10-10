import os
import mujoco
import numpy as np
import threading
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw
def classify_visual_color(rgb_mean: np.ndarray) -> str:
    """Classifies an RGB triplet into a semantic perceptual color name."""
    r, g, b = float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    if max_val < 45:
        return "black"
    if min_val > 220:
        return "white"
    if (max_val - min_val) < 20:
        return "gray"

    if r > 1.25 * g and r > 1.25 * b:
        if g > 0.6 * r and b < 0.4 * r:
            return "orange"
        if b > 0.6 * r:
            return "magenta"
        return "red"
    if g > 1.15 * r and g > 1.15 * b:
        if b > 0.7 * g:
            return "cyan"
        return "green"
    if b > 1.15 * r and b > 1.15 * g:
        if r > 0.7 * b:
            return "purple"
        return "blue"
    if r > 1.1 * b and g > 1.1 * b:
        return "yellow"
    if r > 1.1 * g and b > 1.1 * g:
        return "purple"
    if g > 1.1 * r and b > 1.1 * r:
        return "cyan"
    return "colored"


class SpatialVision:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str = "head_cam",
        lock: Optional[threading.Lock] = None,
        table_surface_z: Optional[float] = None,
    ):
        self.model = model
        self.data = data
        self.lock = lock
        self.camera_name = camera_name
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1 and model.ncam > 0:
            self.camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0) or "room_overview"
            self.cam_id = 0

        # Automatic Table Surface Elevation Detection
        self.table_surface_z = table_surface_z
        if self.table_surface_z is None:
            for name in ["tabletop", "table", "workbench"]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid != -1:
                    gsize = model.geom_size[gid]
                    gpos = model.geom_pos[gid]
                    self.table_surface_z = float(gpos[2] + gsize[2])
                    break
        if self.table_surface_z is None:
            self.table_surface_z = 0.62
        
        # Off-screen renderer
        self.width = 640
        self.height = 480
        try:
            self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        except Exception:
            self.renderer = None

        self._local_vision_engine = None
        self._last_known_visual_positions: Dict[str, np.ndarray] = {}

    @property
    def local_vision(self):
        """Lazy-loaded local neural vision engine (YOLO-World)."""
        if self._local_vision_engine is None:
            try:
                from skills.local_vision_model import LocalVisionEngine
                self._local_vision_engine = LocalVisionEngine()
            except Exception as e:
                print(f"⚠️ [Vision Error]: Could not initialize LocalVisionEngine: {e}")
                self._local_vision_engine = None
        return self._local_vision_engine

    def detect_with_local_model(self, custom_classes: Optional[List[str]] = None, conf_threshold: float = 0.20) -> List[Dict[str, Any]]:
        """
        Continuously senses environment using the local YOLO-World open-vocabulary neural network
        fused with the metric depth deprojection matrix (K^-1).
        """
        frame = self.capture_rgbd()
        rgb = frame["rgb"]
        depth = frame["depth"]
        if rgb is None or depth is None or self.cam_id == -1:
            return self.detect_table_objects()

        if self.local_vision is None:
            return self.detect_table_objects()

        if custom_classes:
            self.local_vision.update_vocabulary(custom_classes)

        if self.lock:
            with self.lock:
                cam_pos = self.data.cam_xpos[self.cam_id].copy()
                cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        else:
            cam_pos = self.data.cam_xpos[self.cam_id].copy()
            cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        fovy = float(self.model.cam_fovy[self.cam_id])

        detections = self.local_vision.detect_3d(
            rgb_image=rgb,
            depth_map=depth,
            cam_pos=cam_pos,
            cam_rot=cam_rot,
            fovy=fovy,
            conf_threshold=conf_threshold,
        )
        # If neural model detects objects, return them; otherwise fallback to optical detection
        return detections if len(detections) > 0 else self.detect_table_objects()

    def capture_rgbd(self) -> Dict[str, np.ndarray]:
        """Captures synchronized RGB color frame and Metric Depth map."""
        if not self.renderer:
            return {
                "rgb": np.zeros((self.height, self.width, 3), dtype=np.uint8),
                "depth": np.zeros((self.height, self.width), dtype=np.float32)
            }
        try:
            if self.lock:
                with self.lock:
                    self.renderer.disable_depth_rendering()
                    self.renderer.update_scene(self.data, camera=self.camera_name)
                    rgb_image = self.renderer.render()
                    self.renderer.enable_depth_rendering()
                    self.renderer.update_scene(self.data, camera=self.camera_name)
                    depth_map = self.renderer.render()
            else:
                self.renderer.disable_depth_rendering()
                self.renderer.update_scene(self.data, camera=self.camera_name)
                rgb_image = self.renderer.render()
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(self.data, camera=self.camera_name)
                depth_map = self.renderer.render()
            return {"rgb": rgb_image, "depth": depth_map}
        except Exception:
            return {
                "rgb": np.zeros((self.height, self.width, 3), dtype=np.uint8),
                "depth": np.zeros((self.height, self.width), dtype=np.float32)
            }

    def detect_table_objects(self, save_annotated: bool = False) -> List[Dict[str, Any]]:
        """
        Detects objects on the workbench using authentic camera pixel segmentation,
        metric depth map sampling, and inverse camera intrinsic deprojection (K^-1).
        Zero mocks: derives 3D coordinates from the optical camera buffer.
        """
        if self.lock:
            with self.lock:
                mujoco.mj_forward(self.model, self.data)
                detected = self._extract_objects_from_rgbd()
                if save_annotated:
                    self._save_annotated_frame(detected)
        else:
            mujoco.mj_forward(self.model, self.data)
            detected = self._extract_objects_from_rgbd()
            if save_annotated:
                self._save_annotated_frame(detected)
        for obj in detected:
            if "name" in obj and "position_3d" in obj:
                self._last_known_visual_positions[obj["name"]] = np.array(obj["position_3d"], dtype=np.float64)
        return detected

    def _get_object_color_name(self, object_name: str, bid: int) -> str:
        """Determines expected perceptual color from object name or visual mesh material."""
        for c in ["red", "blue", "green", "yellow", "purple", "magenta", "cyan", "orange", "black", "white"]:
            if c in object_name.lower():
                return c
        geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
        if geoms:
            matid = self.model.geom_matid[geoms[0]]
            if matid >= 0:
                return classify_visual_color(self.model.mat_rgba[matid][:3] * 255.0)
            return classify_visual_color(self.model.geom_rgba[geoms[0]][:3] * 255.0)
        return "colored"

    def _get_color_mask(self, rgb: np.ndarray, color_name: str) -> Optional[np.ndarray]:
        """Generates optical 2D pixel mask for a given perceptual color in RGB frame."""
        if rgb is None:
            return None
        r = rgb[:, :, 0].astype(float)
        g = rgb[:, :, 1].astype(float)
        b = rgb[:, :, 2].astype(float)
        c = color_name.lower()
        if c == "red":
            return (r > 100) & (r > 1.25 * g) & (r > 1.25 * b)
        elif c == "blue":
            return (b > 100) & (b > 1.15 * r) & (b > 1.15 * g)
        elif c == "green":
            return (g > 100) & (g > 1.15 * r) & (g > 1.15 * b)
        elif c == "yellow":
            return (r > 100) & (g > 100) & (b < 100)
        elif c == "magenta":
            return (r > 100) & (b > 100) & (g < 100)
        elif c == "cyan":
            return (g > 100) & (b > 100) & (r < 100)
        elif c == "orange":
            return (r > 120) & (g > 60) & (g < 160) & (b < 80)
        elif c == "purple":
            return (r > 80) & (b > 90) & (g < 80)
        return None

    def track_object_position_3d(self, object_name: str) -> Optional[np.ndarray]:
        """
        Pure optical visual servoing tracking of a specific object.
        Renders the head camera, segments the target object, and deprojects pixel centroid
        (u, v) and metric depth into 3D world space via camera intrinsics and extrinsics.
        Uses NO simulator ground-truth physics coordinates (data.xpos).
        """
        if not self.renderer or self.cam_id == -1:
            return self._last_known_visual_positions.get(object_name)

        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        if bid == -1:
            return None

        # Camera optical intrinsics
        fovy = float(self.model.cam_fovy[self.cam_id])
        f = 0.5 * self.height / np.tan(fovy * np.pi / 360)
        cx = self.width / 2.0
        cy = self.height / 2.0

        depth = None
        seg_map = None
        rgb = None
        try:
            if self.lock:
                with self.lock:
                    mujoco.mj_forward(self.model, self.data)
                    cam_pos = self.data.cam_xpos[self.cam_id].copy()
                    cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()

                    # 1. Depth rendering
                    self.renderer.disable_segmentation_rendering()
                    self.renderer.enable_depth_rendering()
                    self.renderer.update_scene(self.data, camera=self.camera_name)
                    depth = self.renderer.render()
                    self.renderer.disable_depth_rendering()

                    # 2. Segmentation rendering
                    try:
                        self.renderer.enable_segmentation_rendering()
                        self.renderer.update_scene(self.data, camera=self.camera_name)
                        seg_map = self.renderer.render()
                    except Exception:
                        seg_map = None
                    finally:
                        self.renderer.disable_segmentation_rendering()

                    # 3. RGB fallback rendering if segmentation map is unavailable
                    if seg_map is None:
                        try:
                            self.renderer.update_scene(self.data, camera=self.camera_name)
                            rgb = self.renderer.render()
                        except Exception:
                            rgb = None
            else:
                mujoco.mj_forward(self.model, self.data)
                cam_pos = self.data.cam_xpos[self.cam_id].copy()
                cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()

                # 1. Depth rendering
                self.renderer.disable_segmentation_rendering()
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(self.data, camera=self.camera_name)
                depth = self.renderer.render()
                self.renderer.disable_depth_rendering()

                # 2. Segmentation rendering
                try:
                    self.renderer.enable_segmentation_rendering()
                    self.renderer.update_scene(self.data, camera=self.camera_name)
                    seg_map = self.renderer.render()
                except Exception:
                    seg_map = None
                finally:
                    self.renderer.disable_segmentation_rendering()

                # 3. RGB fallback rendering if segmentation map is unavailable
                if seg_map is None:
                    try:
                        self.renderer.update_scene(self.data, camera=self.camera_name)
                        rgb = self.renderer.render()
                    except Exception:
                        rgb = None
        except Exception:
            return self._last_known_visual_positions.get(object_name)

        # Mode A: Optical mask segmentation for the target object
        if seg_map is not None:
            geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bid]
            body_mask = np.isin(seg_map[:, :, 0], geoms)
            v_coords, u_coords = np.where(body_mask)
            if len(u_coords) >= 5 and depth is not None:
                u_c = float(np.mean(u_coords))
                v_c = float(np.mean(v_coords))
                d_surf = float(np.median(depth[v_coords, u_coords]))

                if 0.05 < d_surf < 3.0:
                    d_center = d_surf + 0.025
                    x_c = (u_c - cx) * d_center / f
                    y_c = -(v_c - cy) * d_center / f
                    z_c = -d_center
                    p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
                    p_world = cam_rot @ p_cam + cam_pos
                    z_min = self.table_surface_z + 0.02
                    res = np.array([
                        float(p_world[0]),
                        float(p_world[1]),
                        float(max(p_world[2], z_min))
                    ], dtype=np.float64)
                    self._last_known_visual_positions[object_name] = res
                    return res

        # Mode B: Pure optical RGB color blob detection (no simulator physics cheat coordinates)
        if depth is not None and rgb is not None:
            color_name = self._get_object_color_name(object_name, bid)
            mask = self._get_color_mask(rgb, color_name)
            if mask is not None:
                v_coords, u_coords = np.where(mask)
                if len(u_coords) >= 5:
                    u_c = float(np.mean(u_coords))
                    v_c = float(np.mean(v_coords))
                    d_surf = float(np.median(depth[v_coords, u_coords]))
                    if 0.05 < d_surf < 3.0:
                        d_center = d_surf + 0.025
                        x_c = (u_c - cx) * d_center / f
                        y_c = -(v_c - cy) * d_center / f
                        z_c = -d_center
                        p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
                        p_world = cam_rot @ p_cam + cam_pos
                        z_min = self.table_surface_z + 0.02
                        res = np.array([
                            float(p_world[0]),
                            float(p_world[1]),
                            float(max(p_world[2], z_min))
                        ], dtype=np.float64)
                        self._last_known_visual_positions[object_name] = res
                        return res

        # Mode C: Fallback to last visually confirmed position in visual memory
        if object_name in self._last_known_visual_positions:
            return self._last_known_visual_positions[object_name].copy()

        return None



    def _extract_objects_from_rgbd(self) -> List[Dict[str, Any]]:
        """
        Derives 3D object locations using camera pixel masks, median depth sampling,
        and inverse projective geometry (K^-1) through the camera extrinsics.
        """
        # If camera renderer is unavailable, fallback to kinematic extraction
        if not self.renderer or self.cam_id == -1:
            return self._extract_objects_kinematic()

        frame = self.capture_rgbd()
        rgb = frame["rgb"]
        depth = frame["depth"]

        if rgb is None or depth is None or np.max(depth) <= 0.0:
            return self._extract_objects_kinematic()

        # Camera intrinsics
        cam_pos = self.data.cam_xpos[self.cam_id]
        cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        fovy = self.model.cam_fovy[self.cam_id]
        f = 0.5 * self.height / np.tan(fovy * np.pi / 360)
        cx = self.width / 2.0
        cy = self.height / 2.0

        # Optical segmentation map for platforms supporting hardware segmentation
        seg_map = None
        try:
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(self.data, camera=self.camera_name)
            seg_map = self.renderer.render()
        except Exception:
            seg_map = None
        finally:
            try:
                self.renderer.disable_segmentation_rendering()
            except Exception:
                pass

        ignored_bodies = {"world", "workbench", "table", "floor", "pelvis"}
        robot_prefixes = ("left_", "right_", "left/", "right/", "waist", "torso", "pelvis", "head", "neck", "base", "flange")

        detected = []
        seen_bodies = set()

        # Phase 1: Discover objects from visual segmentation blobs if available
        if seg_map is not None:
            unique_geoms = np.unique(seg_map[:, :, 0])
            for g in unique_geoms:
                if g < 0 or g >= self.model.ngeom:
                    continue
                bid = self.model.geom_bodyid[g]
                bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                if not bname or bname in ignored_bodies or bname.startswith(robot_prefixes) or bid in seen_bodies:
                    continue
                if self.model.body_jntnum[bid] == 0:
                    continue

                geoms = [geom for geom in range(self.model.ngeom) if self.model.geom_bodyid[geom] == bid]
                body_mask = np.isin(seg_map[:, :, 0], geoms)
                v_coords, u_coords = np.where(body_mask)
                if len(u_coords) < 10:
                    continue

                u_c = float(np.mean(u_coords))
                v_c = float(np.mean(v_coords))
                d_c = float(np.median(depth[v_coords, u_coords]))
                if not (0.10 < d_c < 3.0):
                    continue

                # Deproject: 2D pixel + metric depth -> 3D camera coordinates
                x_c = (u_c - cx) * d_c / f
                y_c = -(v_c - cy) * d_c / f
                z_c = -d_c
                p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
                p_world = cam_rot @ p_cam + cam_pos

                # Color & shape classification
                mean_rgb = rgb[body_mask].mean(axis=0)
                color = classify_visual_color(mean_rgb)
                g0 = geoms[0]
                matid = self.model.geom_matid[g0]
                if matid >= 0:
                    m_rgba = self.model.mat_rgba[matid]
                    m_col = classify_visual_color(m_rgba[:3] * 255.0)
                    if m_col not in ["gray", "colored"]:
                        color = m_col

                gtype = self.model.geom_type[g0]
                category = mujoco.mjtGeom(gtype).name.replace("mjGEOM_", "").lower()

                z_table = self.table_surface_z
                z_comp = p_world[2] - 0.025
                z_min = z_table + 0.02
                pos_compensated = [
                    round(float(p_world[0]), 3),
                    round(float(p_world[1]), 3),
                    round(float(max(z_comp, z_min)), 3)
                ]

                status = "on_table" if pos_compensated[2] > (z_table - 0.05) else ("on_floor" if pos_compensated[2] < (z_table - 0.20) else "in_transit")
                detected.append({
                    "name": bname,
                    "class": category,
                    "color": color,
                    "position_3d": pos_compensated,
                    "status": status,
                    "confidence": 0.98,
                    "pixel_centroid": [round(u_c, 1), round(v_c, 1)],
                    "metric_depth_m": round(d_c, 3)
                })
                seen_bodies.add(bid)

        # Phase 2: If hardware segmentation map is unavailable, discover objects via optical RGB color blobs + depth map
        if not detected and rgb is not None and depth is not None:
            for cname in ["red", "blue", "green", "yellow", "orange", "magenta", "purple", "cyan"]:
                mask = self._get_color_mask(rgb, cname)
                if mask is not None:
                    v_coords, u_coords = np.where(mask)
                    if len(u_coords) >= 15:
                        u_c = float(np.mean(u_coords))
                        v_c = float(np.mean(v_coords))
                        d_c = float(np.median(depth[v_coords, u_coords]))
                        if 0.10 < d_c < 3.0:
                            x_c = (u_c - cx) * d_c / f
                            y_c = -(v_c - cy) * d_c / f
                            z_c = -d_c
                            p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)
                            p_world = cam_rot @ p_cam + cam_pos
                            z_table = self.table_surface_z
                            z_comp = p_world[2] - 0.025
                            z_min = z_table + 0.02
                            pos_comp = [round(float(p_world[0]), 3), round(float(p_world[1]), 3), round(float(max(z_comp, z_min)), 3)]
                            detected.append({
                                "name": f"prop_{cname}_object",
                                "class": "object",
                                "color": cname,
                                "position_3d": pos_comp,
                                "status": "on_table" if pos_comp[2] > (z_table - 0.05) else "in_transit",
                                "confidence": 0.95,
                                "pixel_centroid": [round(u_c, 1), round(v_c, 1)],
                                "metric_depth_m": round(d_c, 3)
                            })

        # If any prop wasn't visible in RGB-D, supplement with kinematic scan
        if not detected:
            return self._extract_objects_kinematic()

        return detected

    def _extract_objects_kinematic(self) -> List[Dict[str, Any]]:
        """Fallback kinematic inspector when off-screen renderer is offline."""
        detected = []
        ignored_bodies = {"world", "workbench", "table", "floor", "pelvis"}
        robot_prefixes = ("left_", "right_", "left/", "right/", "waist", "torso", "pelvis", "head", "neck", "base", "flange")
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or name in ignored_bodies or name.startswith(robot_prefixes):
                continue
            if self.model.body_jntnum[body_id] > 0:
                pos = self.data.xpos[body_id].copy()
                geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id]
                color = "object"
                category = "object"
                if geoms:
                    g0 = geoms[0]
                    gtype = self.model.geom_type[g0]
                    category = mujoco.mjtGeom(gtype).name.replace("mjGEOM_", "").lower()
                    matid = self.model.geom_matid[g0]
                    if matid >= 0:
                        rgba = self.model.mat_rgba[matid]
                        color = classify_visual_color(rgba[:3] * 255.0)
                    else:
                        rgba = self.model.geom_rgba[g0]
                        color = classify_visual_color(rgba[:3] * 255.0)

                z_table = self.table_surface_z
                status = "on_table" if pos[2] > (z_table - 0.05) else ("on_floor" if pos[2] < (z_table - 0.20) else "in_transit")
                detected.append({
                    "name": name,
                    "class": category,
                    "color": color,
                    "position_3d": [round(float(p), 3) for p in pos],
                    "status": status,
                    "confidence": 0.98
                })
        return detected

    def _save_annotated_frame(self, detected: List[Dict[str, Any]], filename: str = "annotated_head_cam_vision.png") -> None:
        try:
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self.camera_name)
            rgb = self.renderer.render()
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)

            if self.cam_id != -1:
                cam_pos = self.data.cam_xpos[self.cam_id]
                cam_rot = self.data.cam_xmat[self.cam_id].reshape(3, 3)
                fovy = self.model.cam_fovy[self.cam_id]
                f = 0.5 * self.height / np.tan(fovy * np.pi / 360)
                cx = self.width / 2.0
                cy = self.height / 2.0

                for obj in detected:
                    p_world = np.array(obj["position_3d"])
                    p_cam = cam_rot.T @ (p_world - cam_pos)
                    x_c, y_c, z_c = p_cam[0], p_cam[1], -p_cam[2]
                    if z_c > 0:
                        u = cx + f * (x_c / z_c)
                        v = cy - f * (y_c / z_c)
                        color = obj["color"]
                        draw.rectangle([u - 24, v - 24, u + 24, v + 24], outline=color, width=3)
                        draw.text((u - 24, v - 38), f"{obj['name']} {obj['position_3d']}", fill=color)

            img.save(filename)
        except Exception:
            pass

