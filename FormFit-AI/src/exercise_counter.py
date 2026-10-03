from collections import deque

class ExerciseCounter:
    def __init__(self, up_threshold, down_threshold, window_size=5):
        self.state = "DOWN" if up_threshold < down_threshold else "UP"
        self.reps = 0
        self.angle_history = deque(maxlen=window_size)
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

    def update(self, angle):
        if angle is None:
            return self.reps, self.state, 0.0

        self.angle_history.append(angle)
        smoothed_angle = sum(self.angle_history) / len(self.angle_history)

        if self.up_threshold < self.down_threshold:
            # e.g. Bicep Curl (up is small angle)
            if smoothed_angle <= self.up_threshold:
                self.state = "UP"
            elif smoothed_angle >= self.down_threshold:
                if self.state == "UP":
                    self.reps += 1
                self.state = "DOWN"
        else:
            # e.g. Squat (down is small angle, up is large angle)
            if smoothed_angle <= self.down_threshold:
                self.state = "DOWN"
            elif smoothed_angle >= self.up_threshold:
                if self.state == "DOWN":
                    self.reps += 1
                self.state = "UP"

        return self.reps, self.state, smoothed_angle
