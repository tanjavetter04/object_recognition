import cv2
import numpy as np
from collections import Counter

# define ranges for color masks
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 100])
# two ranges for red due to hue properties in hsv
LOWER_RED1 = np.array([0, 60, 80])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 60, 80])
UPPER_RED2 = np.array([180, 255, 255])
LOWER_GREEN = np.array([78, 70, 45])
UPPER_GREEN = np.array([104, 255, 255])
LOWER_BLUE = np.array([103, 85, 0])
UPPER_BLUE = np.array([152, 255, 255])
LOWER_YELLOW = np.array([20, 80, 50])
UPPER_YELLOW = np.array([35, 255, 255])

COLOR_MASKS = {
    "Red": (LOWER_RED1, UPPER_RED1, LOWER_RED2, UPPER_RED2),
    "Green": (LOWER_GREEN, UPPER_GREEN),
    "Blue": (LOWER_BLUE, UPPER_BLUE),
    "Yellow": (LOWER_YELLOW, UPPER_YELLOW),
}

PROCESSING_INTERVAL = 30 
ZOOM_FACTOR = 0.7
MORPH_KERNEL_SIZE = 5
APPROX_EPSILON_FACTOR = 0.06 
MIN_CONTOUR_AREA = 100 
CIRCLE_FILL_THRESHOLD = 0.8
TRIANGLE_FILL_THRESHOLD = 0.65
SMALL_CUBE_AREA_THRESHOLD = 20000

def initialize_camera(camera_index=1):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"Error: Camera not found at index {camera_index}")
        return None
    return capture

def bgr_to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def write_text(frame, text, x, y, color=(0, 0, 0), font_scale=0.5, thickness=1):
    cv2.putText(
        frame,
        f"{text}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def morph_operation(frame, operation, kernel_size=MORPH_KERNEL_SIZE, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(frame, operation, kernel, iterations=iterations)

def find_contours(frame, method=cv2.RETR_EXTERNAL, return_hierarchy=False):
    contours, hierarchy = cv2.findContours(frame, method, cv2.CHAIN_APPROX_SIMPLE)
    if return_hierarchy:
        # convert to 2d format instead of 3d format with size 1 in one dimension
        if hierarchy.ndim > 2:
            hierarchy = hierarchy[0]
        return contours, hierarchy
    else:
        return contours

def zoom_frame(frame, scale_factor=ZOOM_FACTOR):
    # zoom to center of the frame
    frame_height, frame_width = frame.shape[:2]
    roi_width = int(frame_width * scale_factor)
    roi_height = int(frame_height * scale_factor)
    start_x = (frame_width - roi_width) // 2
    start_y = (frame_height - roi_height) // 2
    end_x = start_x + roi_width
    end_y = start_y + roi_height
    return frame[start_y:end_y, start_x:end_x].copy()

def find_reference_area(hsv_frame):
    # find the inner corners of the black border in which the objects are placed
    mask_black = cv2.inRange(hsv_frame, LOWER_BLACK, UPPER_BLACK)
    mask_black_closed = morph_operation(mask_black, cv2.MORPH_CLOSE, iterations=2)
    mask_black_opened = morph_operation(mask_black_closed, cv2.MORPH_OPEN, iterations=1)

    black_contours, hierarchy = find_contours(mask_black_opened, cv2.RETR_TREE, return_hierarchy=True)

    ref_x, ref_y, ref_w, ref_h = 0, 0, hsv_frame.shape[1], hsv_frame.shape[0]
    inner_contour_found = False

    if len(black_contours) > 1:
        largest_area = -1
        outer_contour_index = -1
        # find the largest top-level contour 
        for i, contour in enumerate(black_contours):
            if hierarchy[i][3] == -1:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    largest_area = area
                    outer_contour_index = i

        # find the largest child of the outer contour 
        if outer_contour_index != -1:
            largest_child_area = -1
            inner_contour_index = -1
            for i, h in enumerate(hierarchy):
                if h[3] == outer_contour_index: 
                    area = cv2.contourArea(black_contours[i])
                    if area > largest_child_area:
                        largest_child_area = area
                        inner_contour_index = i

            if inner_contour_index != -1:
                inner_contour = black_contours[inner_contour_index]
                ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(inner_contour)
                if ref_w > 0 and ref_h > 0:
                    inner_contour_found = True

    return ref_x, ref_y, ref_w, ref_h, inner_contour_found

def find_contours_in_color_mask(hsv_frame, color_name, color_ranges):
    # special for red because of the hsv color scheme
    if color_name == "Red":
        mask1 = cv2.inRange(hsv_frame, color_ranges[0], color_ranges[1])
        mask2 = cv2.inRange(hsv_frame, color_ranges[2], color_ranges[3])
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        color_mask = cv2.inRange(hsv_frame, color_ranges[0], color_ranges[1])

    opened_mask = morph_operation(color_mask, cv2.MORPH_OPEN, kernel_size=7 ,iterations=3)
    return find_contours(opened_mask)

def calculate_shape_properties(contour):
    properties = {"area": 0, "center_x": -1, "center_y": -1, "approx": None,
                  "vertex_count": 0, "circle_fill_ratio": 0,
                  "triangle_fill_ratio": 0}

    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return None
    properties["area"] = area

    # calculate center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        properties["center_x"] = int(M["m10"] / M["m00"])
        properties["center_y"] = int(M["m01"] / M["m00"])
    else:
        return None

    # approximate contour
    epsilon = APPROX_EPSILON_FACTOR * cv2.arcLength(contour, True)
    properties["approx"] = cv2.approxPolyDP(contour, epsilon, True)
    properties["vertex_count"] = len(properties["approx"])

    # calc circle fill ratio
    _, radius = cv2.minEnclosingCircle(contour)
    if radius > 0:
        circle_area = np.pi * (radius**2)
        if circle_area > 0:
            properties["circle_fill_ratio"] = area / circle_area

    # calc triangle fill ratio
    triangle_area, _ = cv2.minEnclosingTriangle(contour)
    if triangle_area > 0:
        properties["triangle_fill_ratio"] = area / triangle_area

    return properties

def determine_shape(properties):
    if properties["circle_fill_ratio"] > CIRCLE_FILL_THRESHOLD:
        return "Sphere"
    elif properties["triangle_fill_ratio"] > TRIANGLE_FILL_THRESHOLD:
        return "Pyramid"
    elif properties["vertex_count"] == 4:
        if properties["area"] < SMALL_CUBE_AREA_THRESHOLD:
            return "Cube"
        else:
            return "Cuboid"
    return "Unknown"

def calculate_normalized_coords(center_x, center_y, roi_width, roi_height, ref_x, ref_y, ref_w, ref_h, use_reference):
    if use_reference and ref_w > 0 and ref_h > 0:
        relative_center_x = center_x - ref_x
        relative_center_y = center_y - ref_y
        norm_center_x = relative_center_x / ref_w
        norm_center_y = relative_center_y / ref_h
        is_relative = True
    else:
        norm_center_x = center_x / roi_width
        norm_center_y = center_y / roi_height
        is_relative = False

    return norm_center_x, norm_center_y, is_relative

def analyze_contour(contour, roi_dims, ref_area):
    roi_width, roi_height = roi_dims
    ref_x, ref_y, ref_w, ref_h, inner_contour_found = ref_area

    properties = calculate_shape_properties(contour)
    if properties is None:
        return None

    shape = determine_shape(properties)

    norm_center_x, norm_center_y, is_relative = calculate_normalized_coords(
        properties["center_x"], properties["center_y"],
        roi_width, roi_height,
        ref_x, ref_y, ref_w, ref_h,
        inner_contour_found
    )

    return {
        "shape": shape,
        "center_x": properties["center_x"],     # pixel coordinates
        "center_y": properties["center_y"],     # pixel coordinates
        "norm_center_x": norm_center_x,         # normalized coordinates [0, 1]
        "norm_center_y": norm_center_y,         # normalized coordinates [0, 1]
        "is_relative": is_relative,             # reference area: true, frame: false
    }

def draw_object_info(frame, detected_object):
    center_x, center_y = detected_object["center_x"], detected_object["center_y"]
    norm_center_x, norm_center_y = detected_object["norm_center_x"], detected_object["norm_center_y"]
    color = detected_object["color"]
    shape = detected_object["shape"]
    is_relative = detected_object["is_relative"]

    # center marker
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1) 

    label_color = f"{color}"
    label_shape = f"{shape}"
    coord_suffix = "" if is_relative else "*"
    label_coords = f"({norm_center_x:.2f}, {norm_center_y:.2f}){coord_suffix}"

    text_y_offset = -15
    write_text(frame, label_color, center_x + 10, center_y + text_y_offset)
    text_y_offset += 15
    write_text(frame, label_shape, center_x + 10, center_y + text_y_offset)
    if norm_center_x >= 0 and norm_center_y >= 0:
        text_y_offset += 15
        write_text(frame, label_coords, center_x + 10, center_y + text_y_offset)

def draw_statistics(frame, stats):
    start_x = 10
    start_y = 20
    line_height = 18 

    current_y = start_y

    # reference area info
    ref_status = "Detected" if stats.get('ref_found', False) else "Not Found"
    ref_dims = f"({stats.get('ref_w', 0)}x{stats.get('ref_h', 0)}px)" if stats.get('ref_found', False) else ""
    text = f"Ref Area: {ref_status} {ref_dims}"
    write_text(frame, text, start_x, current_y)
    current_y += line_height + 5 

    text = f"Total Objects: {stats.get('total_objects', 0)}"
    write_text(frame, text, start_x, current_y)
    current_y += line_height

    color_counts = stats.get('color_counts', {})
    if color_counts:
        text = " Colors:"
        write_text(frame, text, start_x, current_y)
        current_y += line_height
        for color, count in color_counts.items():
            text = f"  - {color}: {count}"
            write_text(frame, text, start_x, current_y)
            current_y += line_height

    shape_counts = stats.get('shape_counts', {})
    if shape_counts:
        text = " Shapes:"
        write_text(frame, text, start_x, current_y)
        current_y += line_height
        for shape, count in shape_counts.items():
            text = f"  - {shape}: {count}"
            write_text(frame, text, start_x, current_y)
            current_y += line_height

def main():
    capture = initialize_camera(1)
    if capture is None:
        return

    frame_counter = 0

    while True:
        frame_available, frame = capture.read()
        if not frame_available:
            print("End of video stream or error reading frame.")
            break

        frame_counter += 1
        if frame_counter % PROCESSING_INTERVAL != 0:
            continue

        # 1. preprocess frame
        roi_frame = zoom_frame(frame)
        roi_height, roi_width = roi_frame.shape[:2]
        hsv_roi = bgr_to_hsv(roi_frame)

        # 2. find reference area
        ref_x, ref_y, ref_w, ref_h, inner_contour_found = find_reference_area(hsv_roi)
        ref_area_details = (ref_x, ref_y, ref_w, ref_h, inner_contour_found)
        if inner_contour_found:
             cv2.rectangle(roi_frame, (ref_x, ref_y), (ref_x + ref_w, ref_y + ref_h), (0, 255, 0), 2)

        # 3. detect colored objects
        detected_objects = []
        for color_name, color_ranges in COLOR_MASKS.items():
            color_contours = find_contours_in_color_mask(hsv_roi, color_name, color_ranges)

            for contour in color_contours:
                analysis_result = analyze_contour(contour, (roi_width, roi_height), ref_area_details)
                if analysis_result:
                    analysis_result["color"] = color_name
                    detected_objects.append(analysis_result)

        # 4. prepare statistics data
        color_counts = Counter(obj['color'] for obj in detected_objects)
        shape_counts = Counter(obj['shape'] for obj in detected_objects)
        stats_data = {
            "ref_found": inner_contour_found,
            "ref_x": ref_x, "ref_y": ref_y, "ref_w": ref_w, "ref_h": ref_h,
            "total_objects": len(detected_objects),
            "color_counts": dict(color_counts), # convert counter to dict for display func
            "shape_counts": dict(shape_counts)  # convert counter to dict for display func
        }

        # 5. output and display results
        if not detected_objects:
            print("No objects detected.")
        else:
            for obj in detected_objects:
                draw_object_info(roi_frame, obj)
                coord_tag = "" if obj["is_relative"] else "* (Fallback)"
                print(f"Color: {obj['color']}, Shape: {obj['shape']}, ")
                if obj['norm_center_x'] >= 0 and obj['norm_center_y'] >= 0:
                    print(f"Norm Coords: ({obj['norm_center_x']:.2f}, {obj['norm_center_y']:.2f}){coord_tag}")

        print("--------------------")

        draw_statistics(roi_frame, stats_data)

        cv2.imshow("Detected Objects", roi_frame)

        # 6. handle user input
        key = cv2.waitKey(10) 
        if key >= 0:
            break

    capture.release()
    cv2.destroyAllWindows()

main()