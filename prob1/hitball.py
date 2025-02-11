import cv2
import numpy as np
import random
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


class DBSCAN:
    def __init__(self, data, eps, minPts, norm) -> None:
        self.data = data
        self.eps = eps
        self.minPts = minPts
        self.norm = norm
        self.tree = KDTree(self.data[:, :2])

    def find_neighbors(self, p):
        indices = self.tree.query_ball_point(p[:2], self.eps)
        return indices

    def algorithm(self):
        C = 0
        labels = {}
        visited = -np.ones(len(self.data))

        for idx, point in enumerate(self.data):
            if visited[idx] == 1:
                continue
            visited[idx] = 1

            neighbors = self.find_neighbors(point)

            if len(neighbors) < self.minPts:
                labels.setdefault(-1, []).append(idx)
            else:
                C += 1
                labels.setdefault(C, []).append(idx)
                neighbors = set(neighbors)
                neighbors.remove(idx)
                for q in neighbors:
                    if visited[q] == -1:
                        visited[q] = 1
                        q_neighbors = self.find_neighbors(self.data[q])

                        if len(q_neighbors) >= self.minPts:
                            neighbors = list(set(neighbors).union(q_neighbors))
                    if q not in labels[C]:
                        labels[C].append(q)
        centroids = {}
        for cluster_id, indices in labels.items():
            if cluster_id > 0:
                cluster_data = self.data[indices]
                centroids[cluster_id] = self.find_centroid(cluster_data)
        return labels, centroids

    def find_centroid(self, cluster):
        if len(cluster) == 0:
            return None
        center_x = np.mean(cluster[:, 0])
        center_y = np.mean(cluster[:, 1])
        return np.array([center_x, center_y])


def get_contours_from_dbscan(labels, data, img):
    contours = {}
    for cluster_id, indices in labels.items():
        if cluster_id == -1:
            continue

        cluster_points = data[indices]
        if len(cluster_points) >= 3:
            contours[cluster_id] = cluster_points[:, :2]
    else:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def F(t, s):
    return np.dot(np.array([[0, 1], [0, -9.8 / s[1]]]), s)


def rk4(F, t_span, s0, t_eval):
    t0, tf = t_span
    dt = (tf - t0) / (len(t_eval) - 1)
    t = np.linspace(t0, tf, len(t_eval))
    s = np.zeros((len(t_eval), len(s0)))
    s[0] = s0

    for i in range(1, len(t_eval)):
        k1 = F(t[i - 1], s[i - 1])
        k2 = F(t[i - 1] + 0.5 * dt, s[i - 1] + 0.5 * dt * k1)
        k3 = F(t[i - 1] + 0.5 * dt, s[i - 1] + 0.5 * dt * k2)
        k4 = F(t[i - 1] + dt, s[i - 1] + dt * k3)

        s[i] = s[i - 1] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t, s.T


def euler(F, t_span, s0, t_eval):
    t0, tf = t_span
    dt = (tf - t0) / (len(t_eval) - 1)
    t = np.linspace(t0, tf, len(t_eval))
    s = np.zeros((len(t_eval), len(s0)))
    s[0] = s0

    for i in range(1, len(t_eval)):
        s[i] = s[i - 1] + dt * F(t[i - 1], s[i - 1])

    return t, s.T


def obj_func(v0):
    # sol_t, sol_y = euler(F, [start_x, target_x], [y0, v0], t_eval)
    sol_t, sol_y = rk4(F, [start_x, target_x], [y0, v0], t_eval)

    y = sol_y[0]
    return y[-1] - target_y


def obj_func_derivative(v0, delta=1e-5):
    return (obj_func(v0 + delta) - obj_func(v0 - delta)) / (2 * delta)


def newton_method(func, func_derivative, x0, tol=1e-6, max_iter=10):
    x = x0
    for _ in range(max_iter):
        f_x = func(x)
        f_prime_x = func_derivative(x)
        if abs(f_x) < tol:
            return x
        if f_prime_x == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        x -= f_x / f_prime_x
    raise RuntimeError(
        "Newton's method did not converge within the maximum number of iterations."
    )


def sobel_magnitude(I):
    h_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = convolve(I, h_x, mode="same")
    grad_y = convolve(I, h_y, mode="same")
    return np.sqrt(grad_x**2 + grad_y**2)


def generate_random_color():
    return mcolors.to_hex(np.random.rand(4))


image_path = "./pics/5goodballs.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (1, 1), 2)


sb = sobel_magnitude(blurred)
_, binary = cv2.threshold(sb.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

binary[0, :] = 0
binary[-1, :] = 0
binary[:, 0] = 0
binary[:, -1] = 0

plt.imshow(binary)
plt.show()

y, x = np.nonzero(sb)
val = binary[y, x]
data = np.column_stack((x, y, val))
dbscan = DBSCAN(data, eps=5, minPts=85, norm=2)
labels, centroids = dbscan.algorithm()
contours = get_contours_from_dbscan(labels, data, binary)
wmin, wmax = 30, 300
hmin, hmax = 15, 500


ball_centers = []
filtered_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if wmin < w < wmax and hmin < h < hmax:
        filtered_contours.append(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            ball_centers.append((cx, cy))


def is_point_inside_contours(point, contours):
    for contour in contours:
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            return True
    return False


def generate_random_point(image, contours):
    while True:
        random_point = (
            random.randint(0, image.shape[1]),
            random.randint(0, image.shape[0]),
        )
        if not is_point_inside_contours(random_point, contours):
            return random_point


random_point = generate_random_point(image, filtered_contours)

plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

for contour in filtered_contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

for center in ball_centers:
    cv2.circle(image, center, 5, (255, 0, 0), -1)

cv2.circle(image, random_point, 5, (0, 0, 255), -1)

plt.imshow(image)
plt.show()

start_x, start_y = random_point

print(f"Number of Balls: {len(ball_centers)}")


fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(start_x, start_y, color="blue", label="Start Point")

all_trajectory_points = []

for center in ball_centers:
    target_x, target_y = center
    interval = [start_x, target_x]

    print(f"target coordinates: {target_x}, {target_y}")

    y0 = start_y

    v0_guess = 10

    t_eval = np.linspace(start_x, target_x, 30)

    v0 = newton_method(obj_func, obj_func_derivative, v0_guess)

    # sol_t, sol_y = euler(F, [start_x, target_x], [y0, v0], t_eval)
    sol_t, sol_y = rk4(F, [start_x, target_x], [y0, v0], t_eval)

    trajectory_points = [(int(x), int(sol_y[0][i])) for i, x in enumerate(sol_t)]

    print(f"last trajectory point: {sol_t[-1]}, {sol_y[0][-1]}")

    all_trajectory_points.append(trajectory_points)


lines = []
trajectory_colors = [generate_random_color() for _ in range(len(all_trajectory_points))]


for contour in filtered_contours:
    contour_points = np.array([point[0] for point in contour], dtype=np.int32)
    ax.plot(
        contour_points[:, 0],
        contour_points[:, 1],
        color="green",
        linestyle="--",
        alpha=0.5,
    )


for center in ball_centers:
    ax.scatter(center[0], center[1], color="red")


all_x_vals = [point[0] for trajectory in all_trajectory_points for point in trajectory]
all_y_vals = [point[1] for trajectory in all_trajectory_points for point in trajectory]

x_min, x_max = min(all_x_vals), max(all_x_vals)
y_min, y_max = min(all_y_vals), max(all_y_vals)

ax.set_xlim(x_min - 10, x_max + 10)
ax.set_ylim(y_min - 10, y_max + 10)


(tracing_line,) = ax.plot(
    [], [], "-", color="blue", linewidth=2, label="Trajectory Line"
)
(moving_ball,) = ax.plot([], [], "o", color="red", markersize=8, label="Moving Ball")

permanent_trajectories = []


def update(frame):
    global current_trajectory, current_frame

    if current_trajectory < len(all_trajectory_points):
        trajectory_points = all_trajectory_points[current_trajectory]

        if current_frame < len(trajectory_points):
            x_vals = [point[0] for point in trajectory_points[: current_frame + 1]]
            y_vals = [point[1] for point in trajectory_points[: current_frame + 1]]

            tracing_line.set_data(x_vals, y_vals)

            moving_ball.set_data(
                [trajectory_points[current_frame][0]],
                [trajectory_points[current_frame][1]],
            )
            current_frame += 1
        else:
            x_vals = [point[0] for point in trajectory_points]
            y_vals = [point[1] for point in trajectory_points]
            (permanent_trajectory,) = ax.plot(
                x_vals, y_vals, "-", color=trajectory_colors[current_trajectory]
            )
            permanent_trajectories.append(permanent_trajectory)

            current_trajectory += 1
            current_frame = 0

    return [tracing_line, moving_ball] + permanent_trajectories


current_trajectory = 0
current_frame = 0

ani = FuncAnimation(fig, update, frames=500, interval=100, blit=True, repeat=False)


plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.title("Projectile Trajectories with Contours")
plt.show()
