import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from svgpath2mpl import parse_path
from matplotlib.transforms import Affine2D
 


class CustomMarkers:
    def __init__(self):

        # ── Compound path: all robot shapes as subpaths (M...Z M...Z)
        # Each subpath is a closed shape. The coordinate space matches the SVG viewBox.
        # Winding: outer shapes use clockwise, holes (if any) counter-clockwise.
        # Here all shapes are solid fills so all are clockwise.
        
        ROBOT_PATH_D = """
        M 273 108 H 407 Q 415 108 415 116 V 352 Q 415 360 407 360
            H 273 Q 265 360 265 352 V 116 Q 265 108 273 108 Z
        
        M 233 122 H 253 Q 262 122 262 131 V 185 Q 262 194 253 194
            H 233 Q 224 194 224 185 V 131 Q 224 122 233 122 Z
        
        M 427 122 H 447 Q 456 122 456 131 V 185 Q 456 194 447 194
            H 427 Q 418 194 418 185 V 131 Q 418 122 427 122 Z
        
        M 233 266 H 253 Q 262 266 262 275 V 329 Q 262 338 253 338
            H 233 Q 224 338 224 329 V 275 Q 224 266 233 266 Z
        
        M 427 266 H 447 Q 456 266 456 275 V 329 Q 456 338 447 338
            H 427 Q 418 338 418 329 V 275 Q 418 266 427 266 Z
        """
        
        # ── Parse into a matplotlib Path
        robot_path = parse_path(ROBOT_PATH_D)
        
        # ── Normalize: center at origin and scale to fit within [-0.5, 0.5]
        # SVG coords: x in [224, 456], y in [108, 360]
        cx, cy = (224 + 456) / 2, (108 + 360) / 2   # 340, 234
        span = max(456 - 224, 360 - 108)              # 252 (height wins)
        
        # Build transform: translate to origin, flip Y (SVG y-down → mpl y-up), scale
        transform = (
            Affine2D()
            .translate(-cx, -cy)
            .scale(1 / span, -1 / span)   # flip Y axis
        )
        self.rr100_marker = robot_path.transformed(transform)


if __name__ == "__main__":

    mark = CustomMarkers()