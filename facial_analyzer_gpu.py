"""
Facial Uniqueness Analyzer - GPU Accelerated
Celebrates what makes each face distinctly itself.
Uses InsightFace with ONNX Runtime GPU for fast inference on NVIDIA GPUs.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import os

# Set ONNX Runtime to use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Undertone(Enum):
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"


class ContrastLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ProportionalSignature:
    eye_spacing_ratio: float
    nose_length_ratio: float
    lip_fullness_index: float
    forehead_proportion: float
    jaw_width_ratio: float
    face_height_width_ratio: float
    eye_height_width_ratio: float
    philtrum_ratio: float


@dataclass
class SymmetryProfile:
    overall_score: float
    dominant_side: str
    eye_asymmetry: Dict[str, float]
    brow_asymmetry: Dict[str, float]
    mouth_asymmetry: Dict[str, float]
    nose_deviation: float


@dataclass
class FeatureProminence:
    eyes: float
    eyebrows: float
    nose: float
    lips: float
    cheekbones: float
    jawline: float
    primary_feature: str


@dataclass
class GeometricArchetype:
    angular_soft_score: float
    long_compact_score: float
    classification: str


@dataclass
class ColorCharacter:
    undertone: Undertone
    contrast_level: ContrastLevel
    value_range: float
    skin_luminance: float


@dataclass
class ImageQualityWarning:
    """Warnings about image quality that may affect analysis accuracy."""
    code: str  # e.g., "POSE_ANGLE", "LIGHTING", "RESOLUTION"
    message: str
    severity: str  # "warning" or "unreliable"
    affected_metrics: List[str]  # Which metrics are affected


@dataclass
class FacialFingerprint:
    proportions: ProportionalSignature
    symmetry: SymmetryProfile
    prominence: FeatureProminence
    archetype: GeometricArchetype
    color: ColorCharacter
    headline: str
    detailed_report: str
    quality_warnings: List[ImageQualityWarning] = None  # None means no issues
    is_reliable: bool = True  # False if major issues detected


class FacialAnalyzer:
    """
    GPU-accelerated facial feature analyzer using InsightFace.

    InsightFace provides 106 2D landmarks + 68 3D landmarks with GPU acceleration.
    Much faster than MediaPipe on NVIDIA GPUs.
    """

    # InsightFace 2D landmark indices (106 points)
    # Reference: https://github.com/deepinsight/insightface/blob/master/alignment/coordinate_reg/

    # Face contour (indices 0-32)
    FACE_CONTOUR = list(range(0, 33))

    # Left eyebrow (33-37)
    LEFT_EYEBROW = [33, 34, 35, 36, 37]

    # Right eyebrow (42-46)
    RIGHT_EYEBROW = [42, 43, 44, 45, 46]

    # Nose bridge and tip (47-51)
    NOSE_BRIDGE = [47, 48, 49, 50, 51]

    # Nose bottom (52-54)
    NOSE_BOTTOM = [52, 53, 54]

    # Left eye (60-67)
    LEFT_EYE = [60, 61, 62, 63, 64, 65, 66, 67]

    # Right eye (68-75)
    RIGHT_EYE = [68, 69, 70, 71, 72, 73, 74, 75]

    # Outer lip (76-87)
    OUTER_LIP = list(range(76, 88))

    # Inner lip (88-95)
    INNER_LIP = list(range(88, 96))

    # Key points
    NOSE_TIP = 54
    CHIN = 16  # Bottom of face contour
    FOREHEAD_TOP = 0  # Will estimate from brow
    # Face contour: 0=top, goes clockwise. Widest points are around 1-3 (left) and 29-31 (right)
    LEFT_CHEEK = 1  # Near temple/cheekbone area
    RIGHT_CHEEK = 31  # Near temple/cheekbone area
    JAW_LEFT = 6  # Lower jaw
    JAW_RIGHT = 26  # Lower jaw
    # For accurate face width, we'll compute from contour extremes

    def __init__(self, gpu_id: int = 0):
        """Initialize with GPU acceleration."""
        try:
            from insightface.app import FaceAnalysis

            # Initialize InsightFace with GPU
            self.app = FaceAnalysis(
                name='buffalo_l',  # High accuracy model
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=gpu_id, det_size=(640, 640))
            self.use_insightface = True
            print(f"InsightFace initialized with GPU {gpu_id}")

        except ImportError:
            print("InsightFace not available, falling back to MediaPipe")
            self._init_mediapipe()
            self.use_insightface = False

    def _init_mediapipe(self):
        """Fallback to MediaPipe if InsightFace unavailable."""
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        # MediaPipe landmark mappings
        self.mp_landmarks = {
            'LEFT_EYE': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'RIGHT_EYE': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'LEFT_EYEBROW': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            'RIGHT_EYEBROW': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
            'NOSE_TIP': 4,
            'NOSE_BRIDGE': [168, 6, 197, 195, 5],
            'UPPER_LIP': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
            'LOWER_LIP': [146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61],
            'CHIN': 152,
            'FOREHEAD': 10,
            'LEFT_CHEEK': 234,
            'RIGHT_CHEEK': 454,
            'JAW_LEFT': 172,
            'JAW_RIGHT': 397,
            'FACE_OVAL': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }

    def _validate_image_quality(self, image: np.ndarray, face, points: np.ndarray,
                                  kps: np.ndarray = None, bbox: np.ndarray = None) -> List[ImageQualityWarning]:
        """
        Validate image quality and detect conditions that may produce unreliable results.
        Returns a list of warnings, empty if image quality is acceptable.
        """
        warnings = []
        h, w = image.shape[:2]

        # 1. Check face size relative to image (too small = low detail)
        if bbox is not None:
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area_ratio = (face_width * face_height) / (w * h)

            if face_area_ratio < 0.03:  # Face is less than 3% of image
                warnings.append(ImageQualityWarning(
                    code="FACE_TOO_SMALL",
                    message="Face is too small in the image. Use a closer crop for accurate analysis.",
                    severity="unreliable",
                    affected_metrics=["all"]
                ))
            elif face_area_ratio < 0.08:  # Face is less than 8% of image
                warnings.append(ImageQualityWarning(
                    code="FACE_SMALL",
                    message="Face is relatively small. Results may be less precise.",
                    severity="warning",
                    affected_metrics=["eye_spacing", "lip_fullness", "symmetry"]
                ))

        # 2. Check pose angle using eye positions
        if kps is not None:
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]

            # Eye level difference indicates head tilt
            eye_y_diff = abs(left_eye[1] - right_eye[1])
            eye_x_dist = abs(right_eye[0] - left_eye[0])

            if eye_x_dist > 0:
                tilt_ratio = eye_y_diff / eye_x_dist

                if tilt_ratio > 0.25:  # Significant head tilt
                    warnings.append(ImageQualityWarning(
                        code="HEAD_TILT",
                        message="Head is tilted significantly. Symmetry measurements may be affected.",
                        severity="warning",
                        affected_metrics=["symmetry", "eye_asymmetry", "brow_asymmetry"]
                    ))

            # Check for profile/3/4 view using nose position relative to eyes
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_offset = abs(nose[0] - eye_center_x) / eye_x_dist if eye_x_dist > 0 else 0

            if nose_offset > 0.3:  # Nose significantly off-center
                warnings.append(ImageQualityWarning(
                    code="NOT_FRONTAL",
                    message="Face is not frontal (profile or 3/4 view detected). Proportions will be inaccurate.",
                    severity="unreliable",
                    affected_metrics=["eye_spacing", "jaw_width", "face_shape", "symmetry"]
                ))
            elif nose_offset > 0.15:
                warnings.append(ImageQualityWarning(
                    code="SLIGHT_ANGLE",
                    message="Face is slightly angled. Some measurements may be affected.",
                    severity="warning",
                    affected_metrics=["eye_spacing", "jaw_width"]
                ))

        # 3. Check lighting consistency using skin samples
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Sample skin from both cheeks
        left_cheek = points[self.LEFT_CHEEK]
        right_cheek = points[self.RIGHT_CHEEK]

        def sample_region_brightness(center, radius=15):
            x, y = int(center[0]), int(center[1])
            x_min, x_max = max(0, x - radius), min(w, x + radius)
            y_min, y_max = max(0, y - radius), min(h, y + radius)
            region = gray[y_min:y_max, x_min:x_max]
            return np.mean(region) if region.size > 0 else 128

        left_brightness = sample_region_brightness(left_cheek)
        right_brightness = sample_region_brightness(right_cheek)

        # Large difference indicates uneven lighting
        brightness_diff = abs(left_brightness - right_brightness)
        avg_brightness = (left_brightness + right_brightness) / 2

        if brightness_diff > 40:  # Very uneven lighting
            warnings.append(ImageQualityWarning(
                code="UNEVEN_LIGHTING",
                message="Lighting is very uneven across the face. Color analysis and symmetry may be unreliable.",
                severity="unreliable",
                affected_metrics=["contrast", "undertone", "symmetry", "prominence"]
            ))
        elif brightness_diff > 25:
            warnings.append(ImageQualityWarning(
                code="LIGHTING_IMBALANCE",
                message="Some lighting imbalance detected. Color analysis may be affected.",
                severity="warning",
                affected_metrics=["contrast", "undertone"]
            ))

        # 4. Check for extreme brightness/darkness
        if avg_brightness < 50:
            warnings.append(ImageQualityWarning(
                code="TOO_DARK",
                message="Image is very dark. Color and contrast analysis will be unreliable.",
                severity="unreliable",
                affected_metrics=["contrast", "undertone", "skin_luminance"]
            ))
        elif avg_brightness > 220:
            warnings.append(ImageQualityWarning(
                code="OVEREXPOSED",
                message="Image appears overexposed. Color analysis will be unreliable.",
                severity="unreliable",
                affected_metrics=["contrast", "undertone", "skin_luminance"]
            ))

        # 5. Check image resolution
        if w < 200 or h < 200:
            warnings.append(ImageQualityWarning(
                code="LOW_RESOLUTION",
                message="Image resolution is too low for accurate analysis.",
                severity="unreliable",
                affected_metrics=["all"]
            ))

        return warnings

    def analyze(self, image: np.ndarray) -> Optional[FacialFingerprint]:
        """Analyze a face image and return its uniqueness fingerprint."""
        if self.use_insightface:
            return self._analyze_insightface(image)
        else:
            return self._analyze_mediapipe(image)

    def _analyze_insightface(self, image: np.ndarray) -> Optional[FacialFingerprint]:
        """Analyze using InsightFace (GPU accelerated)."""
        # InsightFace expects BGR
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        faces = self.app.get(image)

        if not faces:
            return None

        face = faces[0]  # Take the first/largest face

        # Get landmarks (106 2D points)
        if face.landmark_2d_106 is None:
            return None

        points = face.landmark_2d_106
        h, w = image.shape[:2]

        # Get 5-point keypoints for accurate eye positions
        # kps format: [left_eye, right_eye, nose, left_mouth, right_mouth]
        kps = face.kps if hasattr(face, 'kps') and face.kps is not None else None

        # Get bounding box for face width reference
        bbox = face.bbox if hasattr(face, 'bbox') else None

        # Validate image quality FIRST
        quality_warnings = self._validate_image_quality(image, face, points, kps=kps, bbox=bbox)

        # Determine if results should be marked unreliable
        is_reliable = not any(w.severity == "unreliable" for w in quality_warnings)

        # Analyze each dimension
        proportions = self._analyze_proportions_insight(points, w, h, kps=kps, bbox=bbox)
        symmetry = self._analyze_symmetry_insight(points, kps=kps)
        prominence = self._analyze_prominence(points, image)
        archetype = self._analyze_archetype_insight(points, w, h, bbox=bbox)
        color = self._analyze_color(image, points)

        headline = self._generate_headline(proportions, symmetry, prominence, archetype, color)
        detailed_report = self._generate_detailed_report(proportions, symmetry, prominence, archetype, color)

        # Add quality warnings to detailed report if any
        if quality_warnings:
            warning_text = "\n\nIMAGE QUALITY NOTES"
            for qw in quality_warnings:
                severity_marker = "[!]" if qw.severity == "unreliable" else "[i]"
                warning_text += "\n  {} {}".format(severity_marker, qw.message)
            detailed_report += warning_text

        return FacialFingerprint(
            proportions=proportions,
            symmetry=symmetry,
            prominence=prominence,
            archetype=archetype,
            color=color,
            headline=headline,
            detailed_report=detailed_report,
            quality_warnings=quality_warnings if quality_warnings else None,
            is_reliable=is_reliable
        )

    def _analyze_mediapipe(self, image: np.ndarray) -> Optional[FacialFingerprint]:
        """Analyze using MediaPipe (CPU fallback)."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        points = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])

        proportions = self._analyze_proportions_mp(points, w, h)
        symmetry = self._analyze_symmetry_mp(points)
        prominence = self._analyze_prominence_mp(points, image)
        archetype = self._analyze_archetype_mp(points, w, h)
        color = self._analyze_color_mp(image, points)

        headline = self._generate_headline(proportions, symmetry, prominence, archetype, color)
        detailed_report = self._generate_detailed_report(proportions, symmetry, prominence, archetype, color)

        return FacialFingerprint(
            proportions=proportions,
            symmetry=symmetry,
            prominence=prominence,
            archetype=archetype,
            color=color,
            headline=headline,
            detailed_report=detailed_report
        )

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

    def _analyze_proportions_insight(self, points: np.ndarray, w: int, h: int,
                                       kps: np.ndarray = None, bbox: np.ndarray = None) -> ProportionalSignature:
        """Analyze facial proportions using InsightFace landmarks."""
        # Face dimensions from contour
        face_contour = points[self.FACE_CONTOUR]

        # Get face width - use bbox if available (more reliable), else use contour
        if bbox is not None:
            face_width = bbox[2] - bbox[0]  # bbox format: [x1, y1, x2, y2]
            face_height = bbox[3] - bbox[1]
        else:
            face_width = np.max(face_contour[:, 0]) - np.min(face_contour[:, 0])
            face_height = np.max(face_contour[:, 1]) - np.min(face_contour[:, 1])

        # Estimate forehead from eyebrows
        left_brow_top = points[self.LEFT_EYEBROW].min(axis=0)
        right_brow_top = points[self.RIGHT_EYEBROW].min(axis=0)
        brow_y = min(left_brow_top[1], right_brow_top[1])

        # Eye spacing - USE 5-POINT KEYPOINTS (kps) for accurate eye positions
        # kps format: [left_eye, right_eye, nose, left_mouth, right_mouth]
        if kps is not None:
            left_eye_center = kps[0]
            right_eye_center = kps[1]
        else:
            # Fallback to 106-point landmarks (less accurate for eyes)
            left_eye = points[self.LEFT_EYE]
            right_eye = points[self.RIGHT_EYE]
            left_eye_center = left_eye.mean(axis=0)
            right_eye_center = right_eye.mean(axis=0)

        inter_pupil_distance = self._distance(left_eye_center, right_eye_center)

        # Eye spacing ratio: inter-pupillary distance / face width
        # Typical range: 0.38-0.48, with 0.43 being average
        # Wide-set > 0.46, Close-set < 0.40
        eye_spacing_ratio = inter_pupil_distance / face_width if face_width > 0 else 0.43

        # Nose length
        nose_top = points[self.NOSE_BRIDGE[0]]
        nose_bottom = points[self.NOSE_TIP]
        nose_length = self._distance(nose_top, nose_bottom)
        nose_length_ratio = nose_length / face_height if face_height > 0 else 0.3

        # Lip fullness
        outer_lip = points[self.OUTER_LIP]
        upper_lip_height = abs(outer_lip[3][1] - outer_lip[0][1])
        lower_lip_height = abs(outer_lip[9][1] - outer_lip[6][1])
        lip_fullness = lower_lip_height / upper_lip_height if upper_lip_height > 0 else 1.0

        # Forehead proportion (estimated)
        brow_center = (points[self.LEFT_EYEBROW].mean(axis=0) + points[self.RIGHT_EYEBROW].mean(axis=0)) / 2
        forehead_height = brow_center[1] - (brow_y - 40)  # Estimate
        forehead_proportion = abs(forehead_height) / face_height if face_height > 0 else 0.3

        # Jaw width ratio
        jaw_left = points[self.JAW_LEFT]
        jaw_right = points[self.JAW_RIGHT]
        jaw_width = self._distance(jaw_left, jaw_right)
        jaw_width_ratio = jaw_width / face_width if face_width > 0 else 0.8

        # Face height/width ratio
        face_hw_ratio = face_height / face_width if face_width > 0 else 1.4

        # Eye dimensions
        left_eye = points[self.LEFT_EYE]
        eye_width = self._distance(left_eye[0], left_eye[4])
        eye_height = self._distance(left_eye[2], left_eye[6])
        eye_hw_ratio = eye_height / eye_width if eye_width > 0 else 0.4

        # Philtrum ratio
        nose_tip = points[self.NOSE_TIP]
        lip_top = points[self.OUTER_LIP[0]]
        philtrum_length = self._distance(nose_tip, lip_top)
        philtrum_ratio = philtrum_length / face_height if face_height > 0 else 0.1

        return ProportionalSignature(
            eye_spacing_ratio=min(1.0, max(0.1, eye_spacing_ratio)),
            nose_length_ratio=min(1.0, max(0.1, nose_length_ratio)),
            lip_fullness_index=min(3.0, max(0.3, lip_fullness)),
            forehead_proportion=min(0.5, max(0.1, forehead_proportion)),
            jaw_width_ratio=min(1.2, max(0.5, jaw_width_ratio)),
            face_height_width_ratio=min(2.0, max(0.8, face_hw_ratio)),
            eye_height_width_ratio=min(0.8, max(0.1, eye_hw_ratio)),
            philtrum_ratio=min(0.3, max(0.01, philtrum_ratio))
        )

    def _analyze_symmetry_insight(self, points: np.ndarray, kps: np.ndarray = None) -> SymmetryProfile:
        """Analyze facial symmetry using InsightFace landmarks."""
        # Calculate face center from contour
        face_contour = points[self.FACE_CONTOUR]
        center_x = (np.max(face_contour[:, 0]) + np.min(face_contour[:, 0])) / 2
        face_width = np.max(face_contour[:, 0]) - np.min(face_contour[:, 0])

        # Eye asymmetry - use kps for accurate eye positions
        if kps is not None:
            left_eye_center = kps[0]
            right_eye_center = kps[1]
            # For height/width, we still need landmarks, but use simplified metrics
            left_eye_height = 10  # placeholder
            right_eye_height = 10
            left_eye_width = 20
            right_eye_width = 20
        else:
            left_eye = points[self.LEFT_EYE]
            right_eye = points[self.RIGHT_EYE]
            left_eye_center = left_eye.mean(axis=0)
            right_eye_center = right_eye.mean(axis=0)
            left_eye_height = np.max(left_eye[:, 1]) - np.min(left_eye[:, 1])
            right_eye_height = np.max(right_eye[:, 1]) - np.min(right_eye[:, 1])
            left_eye_width = np.max(left_eye[:, 0]) - np.min(left_eye[:, 0])
            right_eye_width = np.max(right_eye[:, 0]) - np.min(right_eye[:, 0])

        # Calculate vertical difference using eye centers
        vertical_diff = abs(left_eye_center[1] - right_eye_center[1]) / max(left_eye_center[1], 1)

        eye_asymmetry = {
            "height_diff": abs(left_eye_height - right_eye_height) / max(left_eye_height, right_eye_height, 1),
            "width_diff": abs(left_eye_width - right_eye_width) / max(left_eye_width, right_eye_width, 1),
            "vertical_diff": vertical_diff,
            "larger_side": "left" if left_eye_height * left_eye_width > right_eye_height * right_eye_width else "right"
        }

        # Brow asymmetry
        left_brow = points[self.LEFT_EYEBROW]
        right_brow = points[self.RIGHT_EYEBROW]
        left_brow_height = left_brow.mean(axis=0)[1]
        right_brow_height = right_brow.mean(axis=0)[1]

        brow_asymmetry = {
            "height_diff": abs(left_brow_height - right_brow_height) / max(abs(left_brow_height), abs(right_brow_height), 1),
            "higher_side": "left" if left_brow_height < right_brow_height else "right"
        }

        # Mouth asymmetry
        outer_lip = points[self.OUTER_LIP]
        mouth_left = outer_lip[0]
        mouth_right = outer_lip[6]

        mouth_asymmetry = {
            "corner_diff": abs(mouth_left[1] - mouth_right[1]) / max(mouth_left[1], mouth_right[1], 1),
            "higher_corner": "left" if mouth_left[1] < mouth_right[1] else "right"
        }

        # Nose deviation (face_width already calculated from contour at top of function)
        nose_tip = points[self.NOSE_TIP]
        nose_deviation = (nose_tip[0] - center_x) / face_width if face_width > 0 else 0

        # Overall symmetry score
        asymmetry_factors = [
            eye_asymmetry["height_diff"],
            eye_asymmetry["width_diff"],
            brow_asymmetry["height_diff"],
            mouth_asymmetry["corner_diff"],
            abs(nose_deviation)
        ]
        overall_score = 1 - np.mean(asymmetry_factors)

        # Dominant side
        left_indicators = sum([
            1 if eye_asymmetry["larger_side"] == "left" else 0,
            1 if brow_asymmetry["higher_side"] == "left" else 0,
            1 if mouth_asymmetry["higher_corner"] == "left" else 0
        ])

        if left_indicators >= 2:
            dominant_side = "left"
        elif left_indicators <= 1:
            dominant_side = "right"
        else:
            dominant_side = "balanced"

        return SymmetryProfile(
            overall_score=max(0, min(1, overall_score)),
            dominant_side=dominant_side,
            eye_asymmetry=eye_asymmetry,
            brow_asymmetry=brow_asymmetry,
            mouth_asymmetry=mouth_asymmetry,
            nose_deviation=nose_deviation
        )

    def _analyze_prominence(self, points: np.ndarray, image: np.ndarray) -> FeatureProminence:
        """Analyze which features are most prominent."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        def get_region_contrast(indices: List[int]) -> float:
            pts = points[indices]
            x_min, y_min = int(max(0, pts[:, 0].min() - 5)), int(max(0, pts[:, 1].min() - 5))
            x_max, y_max = int(min(w, pts[:, 0].max() + 5)), int(min(h, pts[:, 1].max() + 5))

            if x_max <= x_min or y_max <= y_min:
                return 0.5

            region = gray[y_min:y_max, x_min:x_max]
            if region.size == 0:
                return 0.5

            return min(1.0, np.std(region) / 50)

        eyes_prominence = (get_region_contrast(self.LEFT_EYE) + get_region_contrast(self.RIGHT_EYE)) / 2
        brows_prominence = (get_region_contrast(self.LEFT_EYEBROW) + get_region_contrast(self.RIGHT_EYEBROW)) / 2
        nose_prominence = get_region_contrast(self.NOSE_BRIDGE + self.NOSE_BOTTOM)
        lips_prominence = get_region_contrast(self.OUTER_LIP)
        cheekbone_prominence = get_region_contrast([self.LEFT_CHEEK, self.RIGHT_CHEEK])
        jaw_prominence = get_region_contrast([self.JAW_LEFT, self.JAW_RIGHT, self.CHIN])

        features = {
            "eyes": eyes_prominence,
            "eyebrows": brows_prominence,
            "nose": nose_prominence,
            "lips": lips_prominence,
            "cheekbones": cheekbone_prominence,
            "jawline": jaw_prominence
        }
        primary = max(features, key=features.get)

        return FeatureProminence(
            eyes=eyes_prominence,
            eyebrows=brows_prominence,
            nose=nose_prominence,
            lips=lips_prominence,
            cheekbones=cheekbone_prominence,
            jawline=jaw_prominence,
            primary_feature=primary
        )

    def _analyze_archetype_insight(self, points: np.ndarray, w: int, h: int,
                                     bbox: np.ndarray = None) -> GeometricArchetype:
        """Determine geometric face archetype."""
        # Use bbox for face dimensions (more accurate than contour which is just jawline)
        if bbox is not None:
            face_width = bbox[2] - bbox[0]  # x2 - x1
            face_height = bbox[3] - bbox[1]  # y2 - y1
        else:
            # Fallback to contour (less accurate)
            face_contour = points[self.FACE_CONTOUR]
            face_width = np.max(face_contour[:, 0]) - np.min(face_contour[:, 0])
            face_height = np.max(face_contour[:, 1]) - np.min(face_contour[:, 1])

        # Long vs compact based on height/width ratio
        # Average face ratio is ~1.3-1.4
        # Long > 1.5, Compact < 1.2
        hw_ratio = face_height / face_width if face_width > 0 else 1.35

        # Normalize: 0 = average (1.35), positive = long, negative = compact
        long_compact_score = (hw_ratio - 1.35) / 0.15
        long_compact_score = max(-1, min(1, long_compact_score))

        # Angular vs soft - analyze jaw shape
        # InsightFace contour points are NOT in sequential order around the face
        # We need to find widths at different Y levels dynamically
        contour = points[self.FACE_CONTOUR]

        # Sort points by Y to find face regions
        sorted_by_y = sorted(enumerate(contour), key=lambda x: x[1][1])
        n = len(sorted_by_y)

        # Top third (cheekbone/temple area)
        top_points = [p[1] for p in sorted_by_y[:n//3]]
        top_width = max(p[0] for p in top_points) - min(p[0] for p in top_points)

        # Bottom third (jaw area)
        bot_points = [p[1] for p in sorted_by_y[2*n//3:]]
        bot_width = max(p[0] for p in bot_points) - min(p[0] for p in bot_points)

        # Jaw squareness ratio: how wide is jaw compared to cheekbones
        # Square jaw: ratio > 0.70, Tapered/V-shape: ratio < 0.55
        jaw_ratio = bot_width / top_width if top_width > 0 else 0.65

        # Find chin/jaw angle - use points well above chin for meaningful angle
        chin_idx = sorted_by_y[-1][0]  # Point with max Y
        chin = contour[chin_idx]
        chin_y = chin[1]

        # Find all points and separate by left/right of face center
        center_x = np.mean(contour[:, 0])
        left_contour = [(idx, pt) for idx, pt in enumerate(contour) if pt[0] < center_x]
        right_contour = [(idx, pt) for idx, pt in enumerate(contour) if pt[0] >= center_x]

        # For chin angle, find points that are 20% up from chin (on the jaw line)
        # This gives us the jaw angle rather than measuring at chin level
        y_range = np.max(contour[:, 1]) - np.min(contour[:, 1])
        target_y = chin_y - y_range * 0.20  # 20% up from chin

        # Find closest point to target_y on each side
        if left_contour and right_contour:
            chin_left_idx, chin_left = min(left_contour, key=lambda x: abs(x[1][1] - target_y))
            chin_right_idx, chin_right = min(right_contour, key=lambda x: abs(x[1][1] - target_y))

            # Calculate chin/jaw angle
            v1 = chin_right - chin
            v2 = chin_left - chin
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = max(-1, min(1, cos_angle))
                chin_angle = np.arccos(cos_angle)
            else:
                chin_angle = 2.0
        else:
            chin_angle = 2.0

        # Score calculation
        # Angular = NEGATIVE score, Soft = POSITIVE score
        # Jaw ratio: >0.70 = angular (square jaw), <0.55 = soft (tapered/V-shape)
        # INVERT the score: higher ratio = more angular = more negative
        jaw_score = -(jaw_ratio - 0.625) / 0.075
        jaw_score = max(-1, min(1, jaw_score))

        # Chin angle: <1.8 rad (103 deg) = angular (sharp), >2.2 rad (126 deg) = soft (round)
        # Lower angle = sharper chin = more angular = more negative
        chin_score = (chin_angle - 2.0) / 0.4
        chin_score = max(-1, min(1, chin_score))

        # Combine scores (jaw shape is primary indicator)
        angular_soft_score = (jaw_score * 0.7 + chin_score * 0.3)
        angular_soft_score = max(-1, min(1, angular_soft_score))

        # Classification
        if long_compact_score > 0.3:
            length_desc = "Long"
        elif long_compact_score < -0.3:
            length_desc = "Compact"
        else:
            length_desc = "Balanced"

        if angular_soft_score < -0.2:
            shape_desc = "angular"
        elif angular_soft_score > 0.2:
            shape_desc = "soft"
        else:
            shape_desc = "balanced"

        return GeometricArchetype(
            angular_soft_score=angular_soft_score,
            long_compact_score=long_compact_score,
            classification=f"{length_desc}-{shape_desc}"
        )

    def _analyze_color(self, image: np.ndarray, points: np.ndarray) -> ColorCharacter:
        """Analyze color characteristics."""
        h, w = image.shape[:2]

        left_cheek = points[self.LEFT_CHEEK]
        right_cheek = points[self.RIGHT_CHEEK]

        def sample_skin(center_point, radius=20):
            x, y = int(center_point[0]), int(center_point[1])
            x_min, x_max = max(0, x - radius), min(w, x + radius)
            y_min, y_max = max(0, y - radius), min(h, y + radius)
            return image[y_min:y_max, x_min:x_max]

        left_skin = sample_skin(left_cheek)
        right_skin = sample_skin(right_cheek)

        if left_skin.size == 0 or right_skin.size == 0:
            return ColorCharacter(
                undertone=Undertone.NEUTRAL,
                contrast_level=ContrastLevel.MEDIUM,
                value_range=0.5,
                skin_luminance=0.5
            )

        skin_sample = np.vstack([left_skin.reshape(-1, 3), right_skin.reshape(-1, 3)])
        skin_lab = cv2.cvtColor(skin_sample.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        skin_lab = skin_lab.reshape(-1, 3)

        mean_l, mean_a, mean_b = skin_lab.mean(axis=0)

        if mean_b > 135:
            undertone = Undertone.WARM
        elif mean_a > 135:
            undertone = Undertone.COOL
        else:
            undertone = Undertone.NEUTRAL

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Improved contrast detection using multiple signals
        # Contrast is about the visual difference between skin, hair, eyes, and brows

        # 1. Sample eyebrow region (usually dark features)
        brow_points = np.vstack([points[self.LEFT_EYEBROW], points[self.RIGHT_EYEBROW]])
        brow_x_min, brow_y_min = int(max(0, brow_points[:, 0].min())), int(max(0, brow_points[:, 1].min()))
        brow_x_max, brow_y_max = int(min(w, brow_points[:, 0].max())), int(min(h, brow_points[:, 1].max()))
        brow_region = gray[brow_y_min:brow_y_max, brow_x_min:brow_x_max]

        # 2. Sample hair region (above eyebrows)
        hair_y_min = max(0, brow_y_min - 40)
        hair_y_max = brow_y_min
        hair_x_center = (brow_x_min + brow_x_max) // 2
        hair_x_min = max(0, hair_x_center - 40)
        hair_x_max = min(w, hair_x_center + 40)
        hair_region = gray[hair_y_min:hair_y_max, hair_x_min:hair_x_max]

        # 3. Sample eye region (iris/pupil area - should be dark)
        left_eye_pts = points[self.LEFT_EYE]
        right_eye_pts = points[self.RIGHT_EYE]

        def sample_eye_center(eye_pts, size=8):
            center = eye_pts.mean(axis=0)
            x, y = int(center[0]), int(center[1])
            x_min, x_max = max(0, x - size), min(w, x + size)
            y_min, y_max = max(0, y - size), min(h, y + size)
            return gray[y_min:y_max, x_min:x_max]

        left_eye_region = sample_eye_center(left_eye_pts)
        right_eye_region = sample_eye_center(right_eye_pts)

        # Get feature darkness (use darkest of brows, hair, eyes)
        feature_values = []
        if brow_region.size > 0:
            # Use lower percentile to get actual dark parts (not skin between brow hairs)
            feature_values.append(np.percentile(brow_region, 15))
        if hair_region.size > 0:
            feature_values.append(np.percentile(hair_region, 15))
        if left_eye_region.size > 0:
            feature_values.append(np.percentile(left_eye_region, 10))
        if right_eye_region.size > 0:
            feature_values.append(np.percentile(right_eye_region, 10))

        feature_darkness = min(feature_values) if feature_values else 128

        # Skin brightness - sample from cheek LAB L channel (already have mean_l)
        # Also get skin in grayscale for consistency
        skin_gray = np.mean([
            np.mean(left_skin.mean(axis=(0,1))),
            np.mean(right_skin.mean(axis=(0,1)))
        ]) if left_skin.size > 0 and right_skin.size > 0 else 128

        # Calculate contrast using RELATIVE difference
        # This works better across skin tones
        # For light skin (200): feature_darkness=30 gives (200-30)/200 = 0.85 HIGH
        # For dark skin (80): feature_darkness=20 gives (80-20)/80 = 0.75 HIGH
        # For medium skin (140): feature_darkness=100 gives (140-100)/140 = 0.29 MEDIUM

        if skin_gray > 20:  # Avoid division by very small numbers
            relative_contrast = (skin_gray - feature_darkness) / skin_gray
        else:
            relative_contrast = 0.3

        # Also consider absolute contrast for edge cases
        absolute_contrast = abs(skin_gray - feature_darkness) / 255

        # Combine: primarily use relative, but boost with absolute for very high absolute differences
        contrast_score = relative_contrast * 0.7 + absolute_contrast * 0.3

        # Thresholds calibrated for known celebrities:
        # HIGH: Robert Pattinson (pale+dark hair), Lupita Nyong'o (dark skin+very dark features)
        # LOW: Those with similar skin/hair tones (blonde hair+fair skin, etc.)
        if contrast_score > 0.45:
            contrast_level = ContrastLevel.HIGH
        elif contrast_score > 0.25:
            contrast_level = ContrastLevel.MEDIUM
        else:
            contrast_level = ContrastLevel.LOW

        value_range = (np.max(gray) - np.min(gray)) / 255
        skin_luminance = mean_l / 255

        return ColorCharacter(
            undertone=undertone,
            contrast_level=contrast_level,
            value_range=value_range,
            skin_luminance=skin_luminance
        )

    # MediaPipe fallback methods
    def _analyze_proportions_mp(self, points: np.ndarray, w: int, h: int) -> ProportionalSignature:
        """MediaPipe version of proportion analysis."""
        lm = self.mp_landmarks

        face_top = points[lm['FOREHEAD']]
        face_bottom = points[lm['CHIN']]
        face_left = points[lm['LEFT_CHEEK']]
        face_right = points[lm['RIGHT_CHEEK']]

        face_height = self._distance(face_top, face_bottom)
        face_width = self._distance(face_left, face_right)

        left_eye_center = points[lm['LEFT_EYE']].mean(axis=0)
        right_eye_center = points[lm['RIGHT_EYE']].mean(axis=0)
        eye_distance = self._distance(left_eye_center, right_eye_center)
        eye_spacing_ratio = eye_distance / face_width if face_width > 0 else 0.3

        nose_top = points[lm['NOSE_BRIDGE'][0]]
        nose_bottom = points[lm['NOSE_TIP']]
        nose_length = self._distance(nose_top, nose_bottom)
        nose_length_ratio = nose_length / face_height if face_height > 0 else 0.3

        upper_lip = points[lm['UPPER_LIP']]
        lower_lip = points[lm['LOWER_LIP']]
        upper_lip_height = np.max(upper_lip[:, 1]) - np.min(upper_lip[:, 1])
        lower_lip_height = np.max(lower_lip[:, 1]) - np.min(lower_lip[:, 1])
        lip_fullness = lower_lip_height / upper_lip_height if upper_lip_height > 0 else 1.0

        brow_center = (points[lm['LEFT_EYEBROW']].mean(axis=0) + points[lm['RIGHT_EYEBROW']].mean(axis=0)) / 2
        forehead_height = self._distance(face_top, brow_center)
        forehead_proportion = forehead_height / face_height if face_height > 0 else 0.3

        jaw_width = self._distance(points[lm['JAW_LEFT']], points[lm['JAW_RIGHT']])
        jaw_width_ratio = jaw_width / face_width if face_width > 0 else 0.8

        face_hw_ratio = face_height / face_width if face_width > 0 else 1.4

        left_eye = points[lm['LEFT_EYE']]
        eye_width = self._distance(left_eye[0], left_eye[8])
        eye_height = self._distance(left_eye[4], left_eye[12]) if len(left_eye) > 12 else eye_width * 0.4
        eye_hw_ratio = eye_height / eye_width if eye_width > 0 else 0.4

        philtrum_length = self._distance(points[lm['NOSE_TIP']], points[13])
        philtrum_ratio = philtrum_length / face_height if face_height > 0 else 0.1

        return ProportionalSignature(
            eye_spacing_ratio=eye_spacing_ratio,
            nose_length_ratio=nose_length_ratio,
            lip_fullness_index=lip_fullness,
            forehead_proportion=forehead_proportion,
            jaw_width_ratio=jaw_width_ratio,
            face_height_width_ratio=face_hw_ratio,
            eye_height_width_ratio=eye_hw_ratio,
            philtrum_ratio=philtrum_ratio
        )

    def _analyze_symmetry_mp(self, points: np.ndarray) -> SymmetryProfile:
        """MediaPipe version of symmetry analysis."""
        lm = self.mp_landmarks
        center_x = (points[lm['LEFT_CHEEK']][0] + points[lm['RIGHT_CHEEK']][0]) / 2

        left_eye = points[lm['LEFT_EYE']]
        right_eye = points[lm['RIGHT_EYE']]

        left_eye_height = np.max(left_eye[:, 1]) - np.min(left_eye[:, 1])
        right_eye_height = np.max(right_eye[:, 1]) - np.min(right_eye[:, 1])
        left_eye_width = np.max(left_eye[:, 0]) - np.min(left_eye[:, 0])
        right_eye_width = np.max(right_eye[:, 0]) - np.min(right_eye[:, 0])

        eye_asymmetry = {
            "height_diff": abs(left_eye_height - right_eye_height) / max(left_eye_height, right_eye_height, 1),
            "width_diff": abs(left_eye_width - right_eye_width) / max(left_eye_width, right_eye_width, 1),
            "vertical_diff": 0,
            "larger_side": "left" if left_eye_height * left_eye_width > right_eye_height * right_eye_width else "right"
        }

        left_brow = points[lm['LEFT_EYEBROW']]
        right_brow = points[lm['RIGHT_EYEBROW']]

        brow_asymmetry = {
            "height_diff": abs(left_brow.mean(axis=0)[1] - right_brow.mean(axis=0)[1]) / max(left_brow.mean(axis=0)[1], 1),
            "higher_side": "left" if left_brow.mean(axis=0)[1] < right_brow.mean(axis=0)[1] else "right"
        }

        mouth_asymmetry = {
            "corner_diff": 0.05,
            "higher_corner": "balanced"
        }

        nose_tip = points[lm['NOSE_TIP']]
        face_width = points[lm['RIGHT_CHEEK']][0] - points[lm['LEFT_CHEEK']][0]
        nose_deviation = (nose_tip[0] - center_x) / face_width if face_width > 0 else 0

        asymmetry_factors = [
            eye_asymmetry["height_diff"],
            eye_asymmetry["width_diff"],
            brow_asymmetry["height_diff"],
            mouth_asymmetry["corner_diff"],
            abs(nose_deviation)
        ]
        overall_score = 1 - np.mean(asymmetry_factors)

        return SymmetryProfile(
            overall_score=max(0, min(1, overall_score)),
            dominant_side="balanced",
            eye_asymmetry=eye_asymmetry,
            brow_asymmetry=brow_asymmetry,
            mouth_asymmetry=mouth_asymmetry,
            nose_deviation=nose_deviation
        )

    def _analyze_prominence_mp(self, points: np.ndarray, image: np.ndarray) -> FeatureProminence:
        """MediaPipe version of prominence analysis."""
        return FeatureProminence(
            eyes=0.7,
            eyebrows=0.5,
            nose=0.6,
            lips=0.5,
            cheekbones=0.4,
            jawline=0.4,
            primary_feature="eyes"
        )

    def _analyze_archetype_mp(self, points: np.ndarray, w: int, h: int) -> GeometricArchetype:
        """MediaPipe version of archetype analysis."""
        lm = self.mp_landmarks

        face_top = points[lm['FOREHEAD']]
        face_bottom = points[lm['CHIN']]
        face_left = points[lm['LEFT_CHEEK']]
        face_right = points[lm['RIGHT_CHEEK']]

        face_height = self._distance(face_top, face_bottom)
        face_width = self._distance(face_left, face_right)

        hw_ratio = face_height / face_width if face_width > 0 else 1.4
        long_compact_score = (hw_ratio - 1.4) / 0.2
        long_compact_score = max(-1, min(1, long_compact_score))

        angular_soft_score = 0.0

        if long_compact_score > 0.3:
            length_desc = "Long"
        elif long_compact_score < -0.3:
            length_desc = "Compact"
        else:
            length_desc = "Balanced"

        return GeometricArchetype(
            angular_soft_score=angular_soft_score,
            long_compact_score=long_compact_score,
            classification=f"{length_desc}-balanced"
        )

    def _analyze_color_mp(self, image: np.ndarray, points: np.ndarray) -> ColorCharacter:
        """MediaPipe version of color analysis."""
        lm = self.mp_landmarks
        h, w = image.shape[:2]

        left_cheek = points[lm['LEFT_CHEEK']]
        right_cheek = points[lm['RIGHT_CHEEK']]

        def sample_skin(center_point, radius=20):
            x, y = int(center_point[0]), int(center_point[1])
            x_min, x_max = max(0, x - radius), min(w, x + radius)
            y_min, y_max = max(0, y - radius), min(h, y + radius)
            return image[y_min:y_max, x_min:x_max]

        left_skin = sample_skin(left_cheek)
        right_skin = sample_skin(right_cheek)

        if left_skin.size == 0 or right_skin.size == 0:
            return ColorCharacter(
                undertone=Undertone.NEUTRAL,
                contrast_level=ContrastLevel.MEDIUM,
                value_range=0.5,
                skin_luminance=0.5
            )

        skin_sample = np.vstack([left_skin.reshape(-1, 3), right_skin.reshape(-1, 3)])
        skin_lab = cv2.cvtColor(skin_sample.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
        mean_l, mean_a, mean_b = skin_lab.reshape(-1, 3).mean(axis=0)

        if mean_b > 135:
            undertone = Undertone.WARM
        elif mean_a > 135:
            undertone = Undertone.COOL
        else:
            undertone = Undertone.NEUTRAL

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value_range = (np.max(gray) - np.min(gray)) / 255

        return ColorCharacter(
            undertone=undertone,
            contrast_level=ContrastLevel.MEDIUM,
            value_range=value_range,
            skin_luminance=mean_l / 255
        )

    def _generate_headline(self, proportions: ProportionalSignature,
                          symmetry: SymmetryProfile,
                          prominence: FeatureProminence,
                          archetype: GeometricArchetype,
                          color: ColorCharacter) -> str:
        """Generate a one-line headline summary."""
        tone_desc = {
            Undertone.WARM: "warm-toned",
            Undertone.COOL: "cool-toned",
            Undertone.NEUTRAL: "balanced-tone"
        }[color.undertone]

        feature_desc = {
            "eyes": "eye-forward",
            "eyebrows": "brow-defined",
            "nose": "nose-prominent",
            "lips": "lip-forward",
            "cheekbones": "cheekbone-led",
            "jawline": "jaw-defined"
        }[prominence.primary_feature]

        if color.contrast_level == ContrastLevel.HIGH and symmetry.overall_score > 0.8:
            energy = "striking"
        elif color.contrast_level == ContrastLevel.LOW and archetype.angular_soft_score > 0:
            energy = "gentle"
        elif archetype.angular_soft_score < -0.2:
            energy = "bold"
        else:
            energy = "expressive"

        return f"{energy.capitalize()}, {tone_desc} face with {feature_desc} composition"

    def _generate_detailed_report(self, proportions: ProportionalSignature,
                                  symmetry: SymmetryProfile,
                                  prominence: FeatureProminence,
                                  archetype: GeometricArchetype,
                                  color: ColorCharacter) -> str:
        """Generate the full uniqueness report."""
        lines = []

        lines.append("PROPORTIONAL SIGNATURE")

        # Eye spacing: IPD/face_width ratio. Average ~0.43, Wide >0.46, Close <0.40
        if proportions.eye_spacing_ratio > 0.46:
            lines.append("  - Wide-set eyes create an open, approachable quality")
        elif proportions.eye_spacing_ratio < 0.40:
            lines.append("  - Close-set eyes add intensity and focus")
        else:
            lines.append("  - Classically spaced eyes create balanced harmony")

        if proportions.forehead_proportion > 0.35:
            lines.append("  - High forehead suggests an intellectual, open quality")
        elif proportions.forehead_proportion < 0.25:
            lines.append("  - Compact forehead creates grounded presence")
        else:
            lines.append("  - Balanced forehead-to-face ratio")

        if proportions.lip_fullness_index > 1.3:
            lines.append("  - Fuller lower lip adds expressiveness to your smile")
        elif proportions.lip_fullness_index < 0.8:
            lines.append("  - Defined upper lip creates elegant balance")
        else:
            lines.append("  - Harmoniously proportioned lips")

        if proportions.nose_length_ratio > 0.35:
            lines.append("  - Prominent nose adds character and strength to your profile")
        elif proportions.nose_length_ratio < 0.25:
            lines.append("  - Delicate nose creates subtle refinement")

        if proportions.jaw_width_ratio > 0.85:
            lines.append("  - Strong jaw creates a grounded, confident presence")
        elif proportions.jaw_width_ratio < 0.7:
            lines.append("  - Tapered jaw adds elegance to your profile")

        lines.append("")
        lines.append("SYMMETRY CHARACTER")

        if symmetry.overall_score > 0.85:
            lines.append("  - Remarkably balanced features create calm harmony")
        elif symmetry.overall_score > 0.7:
            lines.append("  - Natural asymmetry adds character and interest")
        else:
            lines.append("  - Distinctive asymmetry creates memorable, dynamic presence")

        if symmetry.dominant_side != "balanced":
            lines.append(f"  - {symmetry.dominant_side.capitalize()}-dominant pattern adds natural expressiveness")

        if symmetry.eye_asymmetry["height_diff"] > 0.1:
            larger = symmetry.eye_asymmetry["larger_side"]
            lines.append(f"  - {larger.capitalize()} eye slightly larger - adds intrigue")

        if symmetry.brow_asymmetry["height_diff"] > 0.05:
            higher = symmetry.brow_asymmetry["higher_side"]
            lines.append(f"  - {higher.capitalize()} brow sits slightly higher - dynamic expressions")

        lines.append("")
        lines.append("WHAT DRAWS THE EYE")

        descriptions = {
            "eyes": "Your eyes are the anchor - they're the natural focal point",
            "eyebrows": "Strong brow definition frames and emphasizes your gaze",
            "nose": "Your nose commands attention - a distinctive centerpiece",
            "lips": "Your lips are the expressive center of your face",
            "cheekbones": "Strong bone structure leads - cheekbones create architecture",
            "jawline": "Defined jawline anchors your face with strength"
        }
        lines.append(f"  - {descriptions[prominence.primary_feature]}")

        scores = {
            "eyes": prominence.eyes,
            "eyebrows": prominence.eyebrows,
            "nose": prominence.nose,
            "lips": prominence.lips,
            "cheekbones": prominence.cheekbones,
            "jawline": prominence.jawline
        }
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_features) > 1 and sorted_features[1][0] != prominence.primary_feature:
            secondary = sorted_features[1][0]
            secondary_desc = {
                "eyes": "Eyes provide expressive support",
                "eyebrows": "Brows frame and define",
                "nose": "Nose adds central character",
                "lips": "Lips add warmth and expression",
                "cheekbones": "Cheekbones add sculptural depth",
                "jawline": "Jawline provides structural foundation"
            }
            lines.append(f"  - {secondary_desc[secondary]}")

        lines.append("")
        lines.append("GEOMETRIC ARCHETYPE")
        lines.append(f"  - {archetype.classification} face shape")

        archetype_descriptions = {
            "Long-angular": "Elongated with defined angles - often described as striking or editorial",
            "Long-soft": "Elongated with gentle contours - elegant and graceful",
            "Long-balanced": "Elongated with harmonious transitions - refined presence",
            "Compact-angular": "Strong, defined features in a balanced frame - powerful presence",
            "Compact-soft": "Rounded with gentle features - approachable and warm",
            "Compact-balanced": "Well-proportioned with subtle definition - classic appeal",
            "Balanced-angular": "Harmonious proportions with defined edges - distinctive",
            "Balanced-soft": "Harmonious proportions with gentle curves - universally appealing",
            "Balanced-balanced": "Classic proportions throughout - timeless quality"
        }

        desc = archetype_descriptions.get(archetype.classification, "Unique combination of proportions")
        lines.append(f"  - {desc}")

        lines.append("")
        lines.append("COLOR CHARACTER")

        undertone_desc = {
            Undertone.WARM: "Warm undertones - golden and peachy hues dominate",
            Undertone.COOL: "Cool undertones - pink and blue hues dominate",
            Undertone.NEUTRAL: "Neutral undertones - balanced and versatile"
        }
        lines.append(f"  - {undertone_desc[color.undertone]}")

        contrast_desc = {
            ContrastLevel.HIGH: "High contrast - dark features against lighter skin create natural drama",
            ContrastLevel.MEDIUM: "Medium contrast - approachable, balanced definition",
            ContrastLevel.LOW: "Soft contrast - gentle transitions create subtle harmony"
        }
        lines.append(f"  - {contrast_desc[color.contrast_level]}")

        return "\n".join(lines)

    def visualize_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Draw landmarks on the image for visualization."""
        annotated = image.copy()

        if self.use_insightface:
            faces = self.app.get(image)
            if not faces:
                return None

            face = faces[0]
            if face.landmark_2d_106 is not None:
                for point in face.landmark_2d_106:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

                # Draw connections for key features
                # Eyes
                for eye_indices in [self.LEFT_EYE, self.RIGHT_EYE]:
                    pts = face.landmark_2d_106[eye_indices].astype(np.int32)
                    cv2.polylines(annotated, [pts], True, (255, 255, 0), 1)

                # Eyebrows
                for brow_indices in [self.LEFT_EYEBROW, self.RIGHT_EYEBROW]:
                    pts = face.landmark_2d_106[brow_indices].astype(np.int32)
                    cv2.polylines(annotated, [pts], False, (255, 0, 255), 1)

                # Lips
                outer_lip_pts = face.landmark_2d_106[self.OUTER_LIP].astype(np.int32)
                cv2.polylines(annotated, [outer_lip_pts], True, (0, 0, 255), 1)

                # Face contour
                contour_pts = face.landmark_2d_106[self.FACE_CONTOUR].astype(np.int32)
                cv2.polylines(annotated, [contour_pts], False, (0, 255, 255), 1)

                # Nose
                nose_pts = face.landmark_2d_106[self.NOSE_BRIDGE + self.NOSE_BOTTOM].astype(np.int32)
                cv2.polylines(annotated, [nose_pts], False, (255, 128, 0), 1)
        else:
            # MediaPipe fallback
            import mediapipe as mp
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return None

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=results.multi_face_landmarks[0],
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        return annotated
