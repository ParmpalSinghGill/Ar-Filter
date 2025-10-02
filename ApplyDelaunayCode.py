"""
Requirements (install once):
pip install opencv-python mediapipe numpy pandas

Usage:
python apply_filter.py --image path/to/face.jpg --filter path/to/filter.png --csv path/to/points.csv --out out.png
The CSV must have 3 columns (header optional): index,x,y
- index: MediaPipe FaceMesh landmark index (0..468 for the base mesh; MediaPipe can give up to 478 with iris)
- x,y: pixel coordinates on the FILTER image (PNG) for that index
"""

import argparse
import sys

import cv2,csv
import numpy as np
import pandas as pd
import mediapipe as mp


# ------------- Utils -------------
def read_csv_annotations(csv_path):
    with open(csv_path) as f:
        pts={int(row[0]):(float(row[1]), float(row[2])) for row in list(csv.reader(f))}

    return pts


def detect_facemesh_points(image_bgr, max_faces=1):
    """Return a list of dicts (one per face): idx -> (x,y) in image pixel coords."""
    mp_face_mesh = mp.solutions.face_mesh
    h, w = image_bgr.shape[:2]
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    faces = []
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,  # includes iris points
        max_num_faces=max_faces,
        min_detection_confidence=0.5,
    ) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return []

        for face_landmarks in res.multi_face_landmarks:
            pts = {}
            for i, lm in enumerate(face_landmarks.landmark):
                x = lm.x * w
                y = lm.y * h
                pts[i] = (x, y)
            faces.append(pts)
    return faces


def build_delaunay_indices(points_dict, bounds):
    """
    Build Delaunay triangulation over given points (2D) using OpenCV Subdiv2D.
    Returns list of triangles as tuples of landmark indices (i,j,k).
    `points_dict` : {idx: (x,y)} in the same coordinate system as `bounds`.
    `bounds`      : (xmin, ymin, xmax, ymax) rectangle for Subdiv2D.
    """
    (xmin, ymin, xmax, ymax) = bounds
    subdiv = cv2.Subdiv2D((int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)))

    # We maintain a list of indices and a parallel array of points
    idxs = list(points_dict.keys())
    pts = np.array([points_dict[i] for i in idxs], dtype=np.float32)

    # Insert points
    for p in pts:
        subdiv.insert((float(p[0]), float(p[1])))

    # Query triangles as coordinates
    tri_list = subdiv.getTriangleList()
    tri_list = np.array(tri_list, dtype=np.float32).reshape(-1, 3, 2)

    # Map triangle vertex coords back to nearest input point index
    # Build a simple nearest lookup (no scipy)
    def nearest_idx(pt):
        d = np.sum((pts - pt) ** 2, axis=1)
        j = int(np.argmin(d))
        return idxs[j]

    # Filter triangles fully inside bounds and convert to index triplets
    triangles = []
    for tri in tri_list:
        if np.all(
            (tri[:, 0] >= xmin)
            & (tri[:, 0] < xmax)
            & (tri[:, 1] >= ymin)
            & (tri[:, 1] < ymax)
        ):
            i0 = nearest_idx(tri[0])
            i1 = nearest_idx(tri[1])
            i2 = nearest_idx(tri[2])

            # Avoid degenerate or duplicate-index triangles
            if len({i0, i1, i2}) == 3:
                triangles.append((i0, i1, i2))

    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for t in triangles:
        key = tuple(sorted(t))
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def warp_triangle(src_img, dst_img, tri_src, tri_dst, alpha_src=None):
    """
    Piecewise-affine warp for a single triangle.
    src_img: source image (Hsrc, Wsrc, 3)
    dst_img: destination image (Hdst, Wdst, 3), modified in place
    tri_src: np.float32 shape (3,2) triangle in src image coords
    tri_dst: np.float32 shape (3,2) triangle in dst image coords
    alpha_src: optional alpha map (Hsrc, Wsrc), same size as src_img; used for blending
    """
    tri_src = np.array(tri_src, dtype=np.float32)
    tri_dst = np.array(tri_dst, dtype=np.float32)

    # Bounding rects
    r_src = cv2.boundingRect(tri_src)
    r_dst = cv2.boundingRect(tri_dst)

    # Offsets
    tri_src_rect = tri_src - np.array([r_src[0], r_src[1]], dtype=np.float32)
    tri_dst_rect = tri_dst - np.array([r_dst[0], r_dst[1]], dtype=np.float32)

    # Extract source patch (RGB and alpha if provided)
    src_patch = src_img[r_src[1] : r_src[1] + r_src[3], r_src[0] : r_src[0] + r_src[2]]
    if alpha_src is not None:
        alpha_patch = alpha_src[
            r_src[1] : r_src[1] + r_src[3], r_src[0] : r_src[0] + r_src[2]
        ]
    else:
        alpha_patch = None

    # Create masks for triangular area
    mask_dst = np.zeros((r_dst[3], r_dst[2]), dtype=np.float32)
    cv2.fillConvexPoly(mask_dst, np.int32(tri_dst_rect), 1.0)

    # Affine transform
    M = cv2.getAffineTransform(tri_src_rect, tri_dst_rect)
    warped = cv2.warpAffine(
        src_patch,
        M,
        (r_dst[2], r_dst[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if alpha_patch is not None:
        warped_alpha = cv2.warpAffine(
            alpha_patch,
            M,
            (r_dst[2], r_dst[3]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # Keep only the triangle area
        warped_alpha = warped_alpha * mask_dst
        alpha = np.expand_dims(warped_alpha, axis=2) / 255.0  # (H,W,1) 0..1
    else:
        alpha = np.expand_dims(mask_dst, axis=2)  # 0..1

    # Composite into dst
    x, y, w, h = r_dst
    roi = dst_img[y : y + h, x : x + w]
    # Ensure types
    warped = warped.astype(np.float32)
    roi_f = roi.astype(np.float32)
    out = roi_f * (1.0 - alpha) + warped * alpha
    dst_img[y : y + h, x : x + w] = np.clip(out, 0, 255).astype(np.uint8)

# ---------------- FaceMesh helpers for video ----------------
class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_track_conf=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_track_conf,
        )

    def detect(self, frame_bgr):
        """Return list of dicts (one per face): idx -> (x,y) in image pixel coords."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        faces = []
        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks:
                pts = {i: (lm.x * w, lm.y * h) for i, lm in enumerate(fl.landmark)}
                faces.append(pts)
        return faces

    def close(self):
        self.mesh.close()


# ---------------- Core application logic ----------------
def load_filter_and_topology(filter_path, csv_path):
    filt_bgra = cv2.imread(filter_path, cv2.IMREAD_UNCHANGED)
    assert filt_bgra is not None, f"Could not read filter PNG: {filter_path}"
    assert filt_bgra.shape[2] in (3, 4), "Filter must be RGB or RGBA"

    if filt_bgra.shape[2] == 3:
        alpha = np.full(filt_bgra.shape[:2], 255, dtype=np.uint8)
        filt_bgr = filt_bgra
    else:
        b, g, r, a = cv2.split(filt_bgra)
        filt_bgr = cv2.merge([b, g, r])
        alpha = a

    filt_pts = read_csv_annotations(csv_path)  # {idx: (x,y) in filter coords}

    fh, fw = filt_bgr.shape[:2]
    triangles = build_delaunay_indices(filt_pts, bounds=(0, 0, float(fw), float(fh)))
    if len(triangles) == 0:
        raise RuntimeError("Delaunay triangulation produced no triangles from CSV.")

    return filt_bgr, alpha, filt_pts, triangles


def render_filter_on_face(frame_bgr, face_pts, filt_bgr, alpha, filt_pts, triangles):
    # Use only indices present in both sets
    common = sorted(set(filt_pts.keys()).intersection(face_pts.keys()))
    if len(common) < 3:
        return frame_bgr  # nothing to draw

    # Build lightweight views of points
    filt_pts_c = {i: filt_pts[i] for i in common}
    face_pts_c = {i: face_pts[i] for i in common}

    out = frame_bgr.copy()
    for (i, j, k) in triangles:
        # Skip triangles that use any index not present in current frame
        if (i not in face_pts_c) or (j not in face_pts_c) or (k not in face_pts_c):
            continue
        tri_src = np.float32([filt_pts_c[i], filt_pts_c[j], filt_pts_c[k]])
        tri_dst = np.float32([face_pts_c[i], face_pts_c[j], face_pts_c[k]])
        warp_triangle(filt_bgr, out, tri_src, tri_dst, alpha_src=alpha)
    return out


# ---------------- CLIs ----------------
def apply_filter_to_image(image_path, filter_path, csv_path, out_path):
    face_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert face_bgr is not None, f"Could not read image: {image_path}"

    filt_bgr, alpha, filt_pts, triangles = load_filter_and_topology(filter_path, csv_path)

    detector = FaceMeshDetector(max_faces=1)
    faces = detector.detect(face_bgr)
    detector.close()
    assert len(faces) > 0, "No face detected."

    out = render_filter_on_face(face_bgr, faces[0], filt_bgr, alpha, filt_pts, triangles)
    cv2.imwrite(out_path, out)
    print(f"Saved: {out_path}")


def run_webcam(stream, filter_path, csv_path, mirror=False, max_faces=1, target_fps=30):
    filt_bgr, alpha, filt_pts, triangles = load_filter_and_topology(filter_path, csv_path)

    cap = cv2.VideoCapture(stream)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {stream}")

    # For smoother output, try to hint FPS
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    detector = FaceMeshDetector(max_faces=max_faces)

    win_name = "Face Filter (press 'q' to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            faces = detector.detect(frame)

            # Composite all faces onto a working copy
            out = frame
            for face_pts in faces:
                out = render_filter_on_face(out, face_pts, filt_bgr, alpha, filt_pts, triangles)

            cv2.imshow(win_name, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


# ---------------- Main ----------------
if __name__ == "__main__":
    # sys.argv.extend("--image pexels-pixabay-415829.jpg --filter filters/glasses.png --csv filters/glasses_annotations_2.csv --out FacewithFilter.jpg".split())
    sys.argv.extend("--image pexels-pixabay-415829.jpg --filter filters/Squid-Game-Front-Man-Mask.png --csv filters/Squid-Game-Front-Man-Mask_annotation.csv --out FacewithFilter.jpg".split())
    # sys.argv.extend("--webcam 0 --filter filters/glasses.png --csv filters/glasses_annotations_2.csv".split())
    # sys.argv.extend("--webcam 0 --filter filters/Squid-Game-Front-Man-Mask.png --csv filters/Squid-Game-Front-Man-Mask_annotation.csv".split())

    parser = argparse.ArgumentParser(description="Apply PNG filter via FaceMesh (image or webcam).")
    parser.add_argument("--image", help="Path to input face image (mutually exclusive with --webcam)")
    parser.add_argument("--webcam", help="Camera index (e.g., 0) or video file path")
    parser.add_argument("--filter", required=True, help="Path to PNG filter")
    parser.add_argument("--csv", required=True, help="CSV with index,x,y for filter")
    parser.add_argument("--out", default="out.png", help="Output path for image mode")
    parser.add_argument("--mirror", action="store_true", help="Mirror the preview (selfie view)")
    parser.add_argument("--max_faces", type=int, default=1, help="Detect up to N faces in webcam mode")
    parser.add_argument("--fps", type=int, default=30, help="Target webcam FPS (hint)")
    args = parser.parse_args()

    if (args.image is None) == (args.webcam is None):
        raise SystemExit("Specify exactly one of --image or --webcam.")

    if args.image:
        apply_filter_to_image(args.image, args.filter, args.csv, args.out)
    else:
        # Try to parse webcam index if an integer, else treat as path/URL
        src = args.webcam
        try:
            src = int(src)
        except ValueError:
            pass
        run_webcam(src, args.filter, args.csv, mirror=args.mirror,
                   max_faces=args.max_faces, target_fps=args.fps)

