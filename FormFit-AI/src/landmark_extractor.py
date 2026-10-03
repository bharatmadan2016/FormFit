import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle at point b
    using three points: a, b, c.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def get_elbow_angle(shoulder, elbow, wrist):
    return calculate_angle((shoulder.x, shoulder.y), (elbow.x, elbow.y), (wrist.x, wrist.y))

def get_shoulder_angle(hip, shoulder, elbow):
    return calculate_angle((hip.x, hip.y), (shoulder.x, shoulder.y), (elbow.x, elbow.y))

def get_elbow_horizontal_offset(shoulder, elbow):
    return elbow.x - shoulder.x

def get_knee_angle(hip, knee, ankle):
    return calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))

def get_hip_angle(shoulder, hip, knee):
    return calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))

def get_torso_angle(shoulder, hip):
    # Angle of torso relative to vertical
    vertical_point = (hip.x, 0)
    return calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), vertical_point)

def get_body_angle(shoulder, hip, ankle):
    # For pushups
    return calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (ankle.x, ankle.y))

def get_hip_alignment(shoulder, hip, ankle):
    # Perpendicular distance from hip to the line connecting shoulder and ankle
    p1 = np.array([shoulder.x, shoulder.y])
    p2 = np.array([ankle.x, ankle.y])
    p3 = np.array([hip.x, hip.y])
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    norm_v1 = np.linalg.norm(v1)
    if norm_v1 == 0:
        return 0.0
        
    # Manual 2D cross product (v1_x * v2_y - v1_y * v2_x)
    cross_product_2d = v1[0] * v2[1] - v1[1] * v2[0]
    
    d = cross_product_2d / norm_v1
    return d

if __name__ == "__main__":
    pass