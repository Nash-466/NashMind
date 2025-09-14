from __future__ import annotations
# genesis_language.py

import numpy as np
from enum import Enum
from collections import Counter, deque

# المرحلة الأولى: بناء "لغة المادة" - طبقة الإدراك الحسي

# 1.1. تعريف الكيانات الأولية
class Pixel:
    """يمثل بكسل فردي في الشبكة مع موقعه ولونه وبيانات وصفية اختيارية."""
    def __init__(self, x: int, y: int, color: int, metadata: dict = None):
        self.x = x
        self.y = y
        self.color = color
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Pixel(x={self.x}, y={self.y}, color={self.color})"

    def __eq__(self, other):
        if not isinstance(other, Pixel):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.color == other.color

    def __hash__(self):
        return hash((self.x, self.y, self.color)) # Added for set operations

class Grid:
    """يمثل شبكة ثنائية الأبعاد من البكسلات، مع أبعادها وخصائص فيزيائية."""
    def __init__(self, data: np.ndarray, gravity_direction: str = 'down', topology: str = 'bounded'):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Grid data must be a 2D numpy array.")
        self.data = data.astype(int) # Ensure integer colors
        self.dimensions = data.shape  # (height, width)
        self.gravity_direction = gravity_direction
        self.topology = topology

    def __repr__(self):
        return f"Grid(dimensions={self.dimensions}, gravity='{self.gravity_direction}', topology='{self.topology}')"

    def get_pixel(self, x: int, y: int) -> Pixel | None:
        if 0 <= y < self.dimensions[0] and 0 <= x < self.dimensions[1]:
            return Pixel(x, y, self.data[y, x])
        return None

    def set_pixel(self, x: int, y: int, color: int):
        if 0 <= y < self.dimensions[0] and 0 <= x < self.dimensions[1]:
            self.data[y, x] = color
        else:
            # Handle toroidal topology if needed, otherwise raise error
            if self.topology == 'toroidal':
                wrapped_x = x % self.dimensions[1]
                wrapped_y = y % self.dimensions[0]
                self.data[wrapped_y, wrapped_x] = color
            else:
                raise IndexError(f"Pixel coordinates ({x}, {y}) out of bounds for grid of size {self.dimensions}")

    def copy(self):
        return Grid(self.data.copy(), self.gravity_direction, self.topology)

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        return np.array_equal(self.data, other.data) and \
               self.gravity_direction == other.gravity_direction and \
               self.topology == other.topology

    def __hash__(self):
        return hash((self.data.tobytes(), self.gravity_direction, self.topology))

# 1.2. تعريف الكيانات المُجمَّعة
class GridObject:
    """يمثل كائنًا مجمّعًا من البكسلات في الشبكة، ويحسب خصائصه الهندسية والتناظرية."""
    def __init__(self, pixel_set: set[Pixel], grid: Grid):
        if not pixel_set:
            raise ValueError("GridObject must be initialized with a non-empty set of pixels.")
        self.pixel_set = frozenset(pixel_set) # Make it immutable
        self.grid = grid # Reference to the grid it belongs to
        self._calculate_properties()

    def _calculate_properties(self):
        # الهوية
        colors = [p.color for p in self.pixel_set]
        self.color = Counter(colors).most_common(1)[0][0] if colors else None # Dominant color
        self.size = len(self.pixel_set)

        # الهندسة
        xs = [p.x for p in self.pixel_set]
        ys = [p.y for p in self.pixel_set]
        self.bounding_box = (min(xs), min(ys), max(xs), max(ys)) # (x_min, y_min, x_max, y_max)
        self.center_of_mass = (sum(xs) / self.size, sum(ys) / self.size) if self.size > 0 else (0,0)
        self.shape_signature = self._calculate_shape_signature() # تمثيل رياضي للشكل
        self.hole_count = self._calculate_hole_count() # عدد الثقوب
        self.perimeter = self._calculate_perimeter() # المحيط

        # التناظر
        self.symmetry_axes = self._calculate_symmetry_axes() # محاور التناظر

    def _calculate_shape_signature(self):
        # يمكن أن يكون هذا تمثيلاً أكثر تعقيدًا، مثل ميزات Hu Moments أو سلسلة من الحواف
        # للتبسيط، نستخدم هنا تمثيلاً بدائيًا يعتمد على الأبعاد النسبية
        x_min, y_min, x_max, y_max = self.bounding_box
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        return f"bbox_ratio_{width/height:.2f}" if height > 0 else "bbox_ratio_inf"

    def _calculate_hole_count(self):
        # يتطلب خوارزمية تعبئة الفيضان أو تحليل المكونات المتصلة
        # للتبسيط، نعتبر الثقوب هي المساحات الفارغة داخل الصندوق المحيط بالكائن
        x_min, y_min, x_max, y_max = self.bounding_box
        bbox_pixels = set()
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                bbox_pixels.add((x, y))
        # Pixels within bounding box that are not part of the object
        internal_empty_pixels = {(p.x, p.y) for p in self.pixel_set}.symmetric_difference(bbox_pixels)
        # This is a very rough estimate. A proper implementation would use connected components.
        return len(internal_empty_pixels) # Placeholder

    def _calculate_perimeter(self):
        # يتطلب حساب عدد البكسلات الحدودية
        perimeter_count = 0
        for p in self.pixel_set:
            # Check 4-connectivity neighbors
            neighbors = [(p.x + 1, p.y), (p.x - 1, p.y), (p.x, p.y + 1), (p.x, p.y - 1)]
            is_border = False
            for nx, ny in neighbors:
                if (nx, ny) not in [(px.x, px.y) for px in self.pixel_set]:
                    is_border = True
                    break
            if is_border:
                perimeter_count += 1
        return perimeter_count

    def _calculate_symmetry_axes(self):
        # يتطلب تحليل هندسي
        axes = []
        x_min, y_min, x_max, y_max = self.bounding_box
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Check horizontal symmetry
        is_h_symmetric = True
        for p in self.pixel_set:
            reflected_y = y_min + (y_max - p.y)
            if (p.x, reflected_y) not in [(px.x, px.y) for px in self.pixel_set]:
                is_h_symmetric = False
                break
        if is_h_symmetric: axes.append('horizontal')

        # Check vertical symmetry
        is_v_symmetric = True
        for p in self.pixel_set:
            reflected_x = x_min + (x_max - p.x)
            if (reflected_x, p.y) not in [(px.x, px.y) for px in self.pixel_set]:
                is_v_symmetric = False
                break
        if is_v_symmetric: axes.append('vertical')

        # Rotational symmetry (90, 180, 270) is more complex and depends on center of mass
        # Placeholder for now
        return axes

    def __repr__(self):
        return f"GridObject(color={self.color}, size={self.size}, bbox={self.bounding_box})"

    def __eq__(self, other):
        if not isinstance(other, GridObject):
            return NotImplemented
        return self.pixel_set == other.pixel_set and self.grid == other.grid

    def __hash__(self):
        return hash((self.pixel_set, id(self.grid))) # Use id for grid as it's a reference

class ObjectSet:
    """حاوية لمجموعة من كائنات GridObject، تدعم عمليات المجموعات."""
    def __init__(self, objects: set[GridObject]):
        self.objects = frozenset(objects)

    def union(self, other: 'ObjectSet') -> 'ObjectSet':
        return ObjectSet(self.objects.union(other.objects))

    def intersection(self, other: 'ObjectSet') -> 'ObjectSet':
        return ObjectSet(self.objects.intersection(other.objects))

    def difference(self, other: 'ObjectSet') -> 'ObjectSet':
        return ObjectSet(self.objects.difference(other.objects))

    def __repr__(self):
        return f"ObjectSet(count={len(self.objects)})"

    def __iter__(self):
        return iter(self.objects)

    def __len__(self):
        return len(self.objects)

    def __contains__(self, item):
        return item in self.objects

# 1.3. تعريف محللات العلاقات
class RelationshipAnalyzer:
    """مكتبة من الدوال الثابتة لتحليل العلاقات بين كائنات GridObject."""
    @staticmethod
    def is_touching(obj1: GridObject, obj2: GridObject) -> bool:
        """يتحقق مما إذا كان الكائنان متجاورين (4-اتصال)."""
        for p1 in obj1.pixel_set:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_pixel = Pixel(p1.x + dx, p1.y + dy, 0) # Color doesn't matter for adjacency check
                if any(p2.x == neighbor_pixel.x and p2.y == neighbor_pixel.y for p2 in obj2.pixel_set):
                    return True
        return False

    @staticmethod
    def is_inside(obj1: GridObject, obj2: GridObject) -> bool:
        """يتحقق مما إذا كان obj1 بالكامل داخل obj2."""
        for p1 in obj1.pixel_set:
            if (p1.x, p1.y) not in [(p2.x, p2.y) for p2 in obj2.pixel_set]:
                return False
        return True

    @staticmethod
    def get_alignment(obj1: GridObject, obj2: GridObject) -> tuple[bool, bool]:
        """يتحقق مما إذا كان الكائنان محاذيين أفقيًا أو رأسيًا (بناءً على مراكز الكتلة)."""
        cx1, cy1 = obj1.center_of_mass
        cx2, cy2 = obj2.center_of_mass
        # Allow for small floating point differences
        h_aligned = abs(cy1 - cy2) < 1e-6
        v_aligned = abs(cx1 - cx2) < 1e-6
        return (h_aligned, v_aligned)

    @staticmethod
    def get_distance(obj1: GridObject, obj2: GridObject) -> float:
        """يحسب المسافة الإقليدية بين مراكز كتلة الكائنين."""
        cx1, cy1 = obj1.center_of_mass
        cx2, cy2 = obj2.center_of_mass
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    @staticmethod
    def share_property(obj1: GridObject, obj2: GridObject, prop_name: str) -> bool:
        """يتحقق مما إذا كان الكائنان يتشاركان نفس قيمة خاصية معينة."""
        return getattr(obj1, prop_name, None) == getattr(obj2, prop_name, None)

    @staticmethod
    def is_scaled_version_of(obj1: GridObject, obj2: GridObject) -> bool:
        """يتحقق مما إذا كان obj1 نسخة مصغرة/مكبرة من obj2. (تنفيذ معقد، هنا مجرد مثال)."""
        # This is a complex geometric check. For a robust solution, one would compare shape signatures
        # or perform image registration techniques.
        # For now, a very basic check based on bounding box aspect ratio and pixel count ratio.
        x1_min, y1_min, x1_max, y1_max = obj1.bounding_box
        x2_min, y2_min, x2_max, y2_max = obj2.bounding_box
        width1, height1 = x1_max - x1_min + 1, y1_max - y1_min + 1
        width2, height2 = x2_max - x2_min + 1, y2_max - y2_min + 1

        if width2 == 0 or height2 == 0: return False

        aspect_ratio1 = width1 / height1
        aspect_ratio2 = width2 / height2

        # Check if aspect ratios are roughly similar and sizes are different
        if abs(aspect_ratio1 - aspect_ratio2) < 0.1 and obj1.size != obj2.size:
            return True
        return False

    @staticmethod
    def is_rotated_version_of(obj1: GridObject, obj2: GridObject) -> bool:
        """يتحقق مما إذا كان obj1 نسخة مدورة من obj2. (تنفيذ معقد، هنا مجرد مثال)."""
        # This is a complex geometric check. Requires comparing shape after rotation.
        # For now, a very basic check: same size and similar shape signature (if rotation invariant).
        if obj1.size == obj2.size and obj1.shape_signature == obj2.shape_signature:
            return True # This is a very weak check
        return False

# Helper function to extract objects from a grid (simplified)
def extract_objects_from_grid(grid: Grid) -> list[GridObject]:
    """يستخرج كائنات GridObject من شبكة معينة باستخدام خوارزمية المكونات المتصلة."""
    objects = []
    visited = np.zeros(grid.dimensions, dtype=bool)

    for r in range(grid.dimensions[0]):
        for c in range(grid.dimensions[1]):
            if grid.data[r, c] != 0 and not visited[r, c]:
                # Found a new object (or part of one)
                current_object_pixels = set()
                q = deque([(c, r)])
                visited[r, c] = True
                object_color = grid.data[r, c]

                while q:
                    cx, cy = q.popleft()
                    current_object_pixels.add(Pixel(cx, cy, object_color))

                    # Check 4-connectivity neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= ny < grid.dimensions[0] and 0 <= nx < grid.dimensions[1] and \
                           not visited[ny, nx] and grid.data[ny, nx] == object_color:
                            visited[ny, nx] = True
                            q.append((nx, ny))
                if current_object_pixels:
                    objects.append(GridObject(current_object_pixels, grid))
    return objects


# المرحلة الثانية: بناء "لغة الفعل" - طبقة الفيزياء والتلاعب

class ActionLanguage:
    """مجموعة من الأفعال القابلة للتركيب لتعديل حالة Grid."""
    @staticmethod
    def _create_empty_grid_like(original_grid: Grid) -> Grid:
        return Grid(np.zeros_like(original_grid.data), original_grid.gravity_direction, original_grid.topology)

    @staticmethod
    def _apply_pixels_to_grid(target_grid: Grid, pixels: set[Pixel]):
        for pixel in pixels:
            try:
                target_grid.set_pixel(pixel.x, pixel.y, pixel.color)
            except IndexError: # Handle out of bounds for bounded topology
                pass

    @staticmethod
    def move(grid: Grid, obj_set: ObjectSet, dx: int, dy: int) -> Grid:
        """ينقل مجموعة من الكائنات بمسافة (dx, dy) في شبكة جديدة."""
        new_grid = ActionLanguage._create_empty_grid_like(grid)
        # Copy all objects not in obj_set first
        all_objects = extract_objects_from_grid(grid)
        for obj in all_objects:
            if obj not in obj_set:
                ActionLanguage._apply_pixels_to_grid(new_grid, obj.pixel_set)

        # Apply moved objects
        moved_pixels = set()
        for obj in obj_set:
            for pixel in obj.pixel_set:
                moved_pixels.add(Pixel(pixel.x + dx, pixel.y + dy, pixel.color))
        ActionLanguage._apply_pixels_to_grid(new_grid, moved_pixels)
        return new_grid

    @staticmethod
    def rotate(grid: Grid, obj_set: ObjectSet, center: tuple[int, int], angle: int) -> Grid:
        """يدور مجموعة من الكائنات حول نقطة مركزية بزاوية معينة (90, 180, 270)."""
        new_grid = ActionLanguage._create_empty_grid_like(grid)
        all_objects = extract_objects_from_grid(grid)
        for obj in all_objects:
            if obj not in obj_set:
                ActionLanguage._apply_pixels_to_grid(new_grid, obj.pixel_set)

        rotated_pixels = set()
        for obj in obj_set:
            for pixel in obj.pixel_set:
                px, py = pixel.x - center[0], pixel.y - center[1]
                if angle == 90:
                    new_px, new_py = -py, px
                elif angle == 180:
                    new_px, new_py = -px, -py
                elif angle == 270:
                    new_px, new_py = py, -px
                else:
                    raise ValueError("Only 90, 180, 270 degree rotations supported for now.")
                rotated_pixels.add(Pixel(new_px + center[0], new_py + center[1], pixel.color))
        ActionLanguage._apply_pixels_to_grid(new_grid, rotated_pixels)
        return new_grid

    @staticmethod
    def flip(grid: Grid, obj_set: ObjectSet, axis: str) -> Grid:
        """يعكس مجموعة من الكائنات أفقيًا أو رأسيًا."""
        new_grid = ActionLanguage._create_empty_grid_like(grid)
        all_objects = extract_objects_from_grid(grid)
        for obj in all_objects:
            if obj not in obj_set:
                ActionLanguage._apply_pixels_to_grid(new_grid, obj.pixel_set)

        flipped_pixels = set()
        for obj in obj_set:
            for pixel in obj.pixel_set:
                if axis == 'horizontal': # Flip across vertical axis (x-coordinate changes)
                    new_x = grid.dimensions[1] - 1 - pixel.x
                    new_y = pixel.y
                elif axis == 'vertical': # Flip across horizontal axis (y-coordinate changes)
                    new_x = pixel.x
                    new_y = grid.dimensions[0] - 1 - pixel.y
                else:
                    raise ValueError("Axis must be 'horizontal' or 'vertical'.")
                flipped_pixels.add(Pixel(new_x, new_y, pixel.color))
        ActionLanguage._apply_pixels_to_grid(new_grid, flipped_pixels)
        return new_grid

    @staticmethod
    def scale(grid: Grid, obj_set: ObjectSet, factor: float) -> Grid:
        """يغير حجم مجموعة من الكائنات بعامل معين. (تنفيذ معقد، هنا مجرد مثال)."""
        # This is a non-rigid transformation, more complex. Placeholder.
        # A proper implementation would involve resampling pixels.
        return grid.copy() # Return original for now

    @staticmethod
    def stretch(grid: Grid, obj_set: ObjectSet, axis: str, factor: float) -> Grid:
        """يمدد مجموعة من الكائنات على طول محور معين. (تنفيذ معقد، هنا مجرد مثال)."""
        # Non-rigid transformation. Placeholder.
        return grid.copy() # Return original for now

    @staticmethod
    def recolor(grid: Grid, obj_set: ObjectSet, color_map_function) -> Grid:
        """يعيد تلوين مجموعة من الكائنات باستخدام دالة تعيين الألوان."""
        new_grid = ActionLanguage._create_empty_grid_like(grid)
        all_objects = extract_objects_from_grid(grid)
        for obj in all_objects:
            if obj not in obj_set:
                ActionLanguage._apply_pixels_to_grid(new_grid, obj.pixel_set)

        recolored_pixels = set()
        for obj in obj_set:
            for pixel in obj.pixel_set:
                new_color = color_map_function(pixel)
                recolored_pixels.add(Pixel(pixel.x, pixel.y, new_color))
        ActionLanguage._apply_pixels_to_grid(new_grid, recolored_pixels)
        return new_grid

    @staticmethod
    def copy_objects(grid: Grid, obj_set: ObjectSet, offset_vector: tuple[int, int]) -> Grid:
        """ينسخ مجموعة من الكائنات إلى موقع جديد بإزاحة معينة."""
        new_grid = grid.copy() # Start with current grid, then add copies
        dx, dy = offset_vector
        copied_pixels = set()
        for obj in obj_set:
            for pixel in obj.pixel_set:
                copied_pixels.add(Pixel(pixel.x + dx, pixel.y + dy, pixel.color))
        ActionLanguage._apply_pixels_to_grid(new_grid, copied_pixels)
        return new_grid

    @staticmethod
    def create_primitive(grid: Grid, shape_name: str, color: int, position: tuple[int, int], size: int = 1) -> Grid:
        """ينشئ شكلًا بدائيًا (مثل مربع أو دائرة) في الشبكة."""
        new_grid = grid.copy()
        x, y = position
        if shape_name == 'square':
            for i in range(size):
                for j in range(size):
                    new_grid.set_pixel(x + i, y + j, color)
        elif shape_name == 'circle':
            # Simplified circle for now
            radius = size // 2
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i*i + j*j <= radius*radius:
                        new_grid.set_pixel(x + i, y + j, color)
        else:
            raise ValueError(f"Unknown primitive shape: {shape_name}")
        return new_grid

    @staticmethod
    def destroy(grid: Grid, obj_set: ObjectSet) -> Grid:
        """يدمر مجموعة من الكائنات من الشبكة (يضع لون الخلفية 0)."""
        new_grid = grid.copy()
        for obj in obj_set:
            for pixel in obj.pixel_set:
                new_grid.set_pixel(pixel.x, pixel.y, 0) # Set to background color (0)
        return new_grid

    @staticmethod
    def apply_to_all(grid: Grid, obj_set: ObjectSet, action_function, *args, **kwargs) -> Grid:
        """يطبق دالة فعل على كل كائن في مجموعة الكائنات."""
        current_grid = grid.copy()
        # Re-extract objects after each action to ensure consistency if action_function modifies the grid
        # This is a simplified approach; a more robust one would involve passing the object itself
        # and having the action_function return a modified object or a new grid.
        # For now, we'll assume action_function takes grid, ObjectSet, and other args.
        for obj in obj_set:
            # Need to re-extract the specific object from the current_grid to ensure it's up-to-date
            # This is a placeholder for a more sophisticated object tracking mechanism.
            # For now, we'll just pass the original object and hope the action function is robust.
            current_grid = action_function(current_grid, ObjectSet({obj}), *args, **kwargs)
        return current_grid

    @staticmethod
    def group_by(objects: list[GridObject], property_name: str) -> dict[any, ObjectSet]:
        """يجمع الكائنات بناءً على قيمة خاصية مشتركة."""
        grouped_objects = {}
        for obj in objects:
            prop_value = getattr(obj, property_name, None)
            if prop_value not in grouped_objects:
                grouped_objects[prop_value] = set()
            grouped_objects[prop_value].add(obj)
        return {k: ObjectSet(v) for k, v in grouped_objects.items()}

# المرحلة الثالثة: بناء "لغة القصد" - طبقة التوجيه الميتافيزيقي

class Principle(Enum):
    """تعداد للمبادئ العليا التي تصف نوع الحل."""
    SYMMETRY_COMPLETION = "SymmetryCompletion"
    PATTERN_CONTINUATION = "PatternContinuation"
    OUTLIER_REMOVAL = "OutlierRemoval"
    HOMOGENIZATION = "Homogenization"

class PrincipleBase:
    """كلاس أساسي للمبادئ"""
    def get_candidate_actions(self, grid: Grid, initial_objects: list[GridObject]):
        """يجب أن تُرجع قائمة بالأفعال المرشحة لتحقيق هذا المبدأ."""
        raise NotImplementedError("Subclasses must implement get_candidate_actions")

class SymmetryCompletion(PrincipleBase):
    """مبدأ إكمال التناظر."""
    def __init__(self):
        super().__init__()

    def get_candidate_actions(self, grid: Grid, initial_objects: list[GridObject]):
        actions = []
        # If there's one object, suggest flipping it to create symmetry
        if len(initial_objects) == 1:
            obj = initial_objects[0]
            # Calculate center of the grid for reflection if no specific center is given
            grid_center_x = grid.dimensions[1] // 2
            grid_center_y = grid.dimensions[0] // 2

            # Suggest flipping across grid centerlines
            actions.append((ActionLanguage.flip, grid, ObjectSet({obj}), 'horizontal'))
            actions.append((ActionLanguage.flip, grid, ObjectSet({obj}), 'vertical'))

            # Suggest rotating around its own center of mass to complete rotational symmetry
            # This is more about self-symmetry, not completing a pattern with another object
            # For completing symmetry with another object, it would involve copying and then transforming

        # More complex logic needed to find symmetry axes and complete patterns between multiple objects
        return actions

class PatternContinuation(PrincipleBase):
    """مبدأ استمرارية النمط."""
    def __init__(self):
        super().__init__()

    def analyze_pattern(self, grid: Grid, initial_objects: list[GridObject]):
        """يحلل النمط الموجود بين الكائنات ويحدد الإزاحة أو التحويل."""
        if len(initial_objects) < 2: return None

        # Sort objects by some criteria, e.g., x-coordinate, then y-coordinate
        sorted_objects = sorted(initial_objects, key=lambda obj: (obj.center_of_mass[0], obj.center_of_mass[1]))

        # Simple linear pattern detection based on the first two objects
        obj1 = sorted_objects[0]
        obj2 = sorted_objects[1]

        dx = int(obj2.center_of_mass[0] - obj1.center_of_mass[0])
        dy = int(obj2.center_of_mass[1] - obj1.center_of_mass[1])

        # Check if this pattern holds for subsequent objects (simplified)
        is_consistent = True
        for i in range(len(sorted_objects) - 1):
            current_obj = sorted_objects[i]
            next_obj = sorted_objects[i+1]
            if not (abs((next_obj.center_of_mass[0] - current_obj.center_of_mass[0]) - dx) < 1e-6 and
                    abs((next_obj.center_of_mass[1] - current_obj.center_of_mass[1]) - dy) < 1e-6):
                is_consistent = False
                break

        if is_consistent and (dx != 0 or dy != 0):
            return {'type': 'linear_offset', 'dx': dx, 'dy': dy, 'last_object': sorted_objects[-1]}
        return None

    def get_candidate_actions(self, grid: Grid, initial_objects: list[GridObject]):
        actions = []
        pattern_info = self.analyze_pattern(grid, initial_objects)
        if pattern_info and pattern_info['type'] == 'linear_offset':
            last_obj = pattern_info['last_object']
            dx, dy = pattern_info['dx'], pattern_info['dy']
            # Suggest copying the last object with the detected offset
            actions.append((ActionLanguage.copy_objects, grid, ObjectSet({last_obj}), (dx, dy)))
        return actions

class OutlierRemoval(PrincipleBase):
    """مبدأ إزالة الشواذ."""
    def __init__(self):
        super().__init__()

    def find_outlier(self, grid: Grid, initial_objects: list[GridObject]):
        """يجد الكائن الشاذ بناءً على خصائصه (الحجم، اللون، الشكل)."""
        if not initial_objects: return None

        # Analyze properties: size, color, shape_signature
        sizes = [obj.size for obj in initial_objects]
        colors = [obj.color for obj in initial_objects]
        shape_signatures = [obj.shape_signature for obj in initial_objects]

        # Simple outlier detection: find the object whose size/color/shape is least common
        outlier_scores = {obj: 0 for obj in initial_objects}

        if len(sizes) > 1:
            size_counts = Counter(sizes)
            for obj in initial_objects:
                outlier_scores[obj] += (len(initial_objects) - size_counts[obj.size])

        if len(colors) > 1:
            color_counts = Counter(colors)
            for obj in initial_objects:
                outlier_scores[obj] += (len(initial_objects) - color_counts[obj.color])

        if len(shape_signatures) > 1:
            shape_counts = Counter(shape_signatures)
            for obj in initial_objects:
                outlier_scores[obj] += (len(initial_objects) - shape_counts[obj.shape_signature])

        if not outlier_scores: return None

        # The object with the highest outlier score is the most likely outlier
        outlier = max(outlier_scores, key=outlier_scores.get)
        # Only consider it an outlier if its score is significantly higher than others
        scores_values = list(outlier_scores.values())
        if len(scores_values) > 1 and outlier_scores[outlier] > np.mean(scores_values) + np.std(scores_values):
             return outlier
        return None

    def get_candidate_actions(self, grid: Grid, initial_objects: list[GridObject]):
        actions = []
        outlier = self.find_outlier(grid, initial_objects)
        if outlier:
            actions.append((ActionLanguage.destroy, grid, ObjectSet({outlier})))
        return actions

class Homogenization(PrincipleBase):
    """مبدأ التجانس (جعل الأشياء متشابهة)."""
    def __init__(self):
        super().__init__()

    def get_candidate_actions(self, grid: Grid, initial_objects: list[GridObject]):
        actions = []
        if not initial_objects: return actions

        # Suggest recoloring all objects to the most common color
        colors = [obj.color for obj in initial_objects]
        if colors:
            most_common_color = Counter(colors).most_common(1)[0][0]
            # A simple color map function that always returns the most common color
            color_map_func = lambda pixel: most_common_color
            actions.append((ActionLanguage.recolor, grid, ObjectSet(initial_objects), color_map_func))

        # Suggest resizing/stretching all objects to the most common size/shape (placeholder for actual scale/stretch)
        # This would require more sophisticated actions than currently implemented in ActionLanguage
        return actions

class IntentClassifier:
    """مصنف النوايا الذي يحاول استنتاج المبدأ الأكثر احتمالاً من تحويل الشبكة."""
    def __init__(self):
        # This would be a trained ML model. For now, a simple rule-based system.
        self.principles = {
            Principle.SYMMETRY_COMPLETION: SymmetryCompletion(),
            Principle.PATTERN_CONTINUATION: PatternContinuation(),
            Principle.OUTLIER_REMOVAL: OutlierRemoval(),
            Principle.HOMOGENIZATION: Homogenization()
        }

    def classify(self, before_grid: Grid, after_grid: Grid) -> Principle | None:
        """يصنف التحويل من شبكة 'قبل' إلى 'بعد' إلى مبدأ واحد."""
        # This is a placeholder for a machine learning model
        # In a real scenario, this would analyze the transformation from before_grid to after_grid
        # and infer the most likely principle.

        # For demonstration, we'll use simple heuristics based on changes.
        before_objects = extract_objects_from_grid(before_grid)
        after_objects = extract_objects_from_grid(after_grid)

        # Heuristic 1: If number of objects decreased, possibly OutlierRemoval or Destroy
        if len(after_objects) < len(before_objects):
            # Try to find an outlier in before_objects that is missing in after_objects
            outlier_principle = self.principles[Principle.OUTLIER_REMOVAL]
            outlier = outlier_principle.find_outlier(before_grid, before_objects)
            if outlier and outlier not in ObjectSet(after_objects):
                return Principle.OUTLIER_REMOVAL

        # Heuristic 2: If number of objects increased, possibly PatternContinuation or Copy
        if len(after_objects) > len(before_objects):
            # Check if new objects form a pattern with existing ones
            pattern_principle = self.principles[Principle.PATTERN_CONTINUATION]
            pattern_info = pattern_principle.analyze_pattern(before_grid, before_objects)
            if pattern_info:
                # This is a very weak check, needs to verify if after_grid matches the predicted pattern
                return Principle.PATTERN_CONTINUATION

        # Heuristic 3: If colors changed but shapes/positions are similar, possibly Homogenization or Recolor
        if len(before_objects) == len(after_objects):
            # Check if colors are more uniform in after_grid
            before_colors = [obj.color for obj in before_objects]
            after_colors = [obj.color for obj in after_objects]
            if len(set(before_colors)) > len(set(after_colors)):
                return Principle.HOMOGENIZATION

            # Check for symmetry changes
            before_symmetry_axes = [obj.symmetry_axes for obj in before_objects]
            after_symmetry_axes = [obj.symmetry_axes for obj in after_objects]
            # This is hard to compare directly. A better check would be to apply symmetry actions
            # and see if it matches after_grid.

        # Default or fallback if no clear pattern
        return Principle.SYMMETRY_COMPLETION # As a default example


# Docstrings and examples would be added here for each class/method

