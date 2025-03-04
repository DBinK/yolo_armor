import pygame
import math

# 初始化 Pygame
pygame.init()

# 设置窗口尺寸
width, height = 1280, 720
screen = pygame.display.set_mode((width, height))

# 设置颜色
black = (0, 0, 0)
red = (255, 0, 0)
white = (255, 255, 0)
gray = (200, 200, 200)  # 灰色
blue = (0, 0, 255)
green = (0, 255, 0)  # 绿色

# 字体设置
font = pygame.font.Font(None, 36)

# 目标点的初始位置
target_pos = [width // 2, height // 2]
# 追踪点的初始位置
chaser_pos = [100, 100]

# PID控制参数
Kp = 0.1  # 比例系数
Ki = 0.01  # 积分系数
Kd = 0.1  # 微分系数

# PID控制变量
integral = [0, 0]  # 积分部分
previous_error = [0, 0]  # 上一帧的误差

# 轨迹列表
chaser_trail = []
mouse_trail = []

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:  # 按下 R 键复位
                chaser_pos = [100, 100]  # 重置追踪点位置
                integral = [0, 0]  # 重置积分
                previous_error = [0, 0]  # 重置上一个误差
                chaser_trail.clear()  # 清除追踪点轨迹
                mouse_trail.clear()  # 清除鼠标轨迹

    # 获取鼠标位置
    target_pos = pygame.mouse.get_pos()

    # 计算误差
    error = [target_pos[0] - chaser_pos[0], target_pos[1] - chaser_pos[1]]

    # 计算积分和微分
    integral[0] += error[0]
    integral[1] += error[1]
    derivative = [error[0] - previous_error[0], error[1] - previous_error[1]]

    # PID控制输出
    output = [Kp * error[0] + Ki * integral[0] + Kd * derivative[0],
              Kp * error[1] + Ki * integral[1] + Kd * derivative[1]]

    # 更新追踪点位置
    chaser_pos[0] += output[0]
    chaser_pos[1] += output[1] 

    # 将当前追踪点位置添加到追踪轨迹列表
    chaser_trail.append((int(chaser_pos[0]), int(chaser_pos[1])))

    # 将当前鼠标位置添加到鼠标轨迹列表
    mouse_trail.append((target_pos[0], target_pos[1]))

    # 绘制
    screen.fill(black)

    # 绘制追踪点的轨迹
    if len(chaser_trail) > 1:
        pygame.draw.lines(screen, white, False, chaser_trail, 1)

    # 绘制鼠标的轨迹
    if len(mouse_trail) > 1:
        pygame.draw.lines(screen, gray, False, mouse_trail, 1)  # 使用灰色绘制鼠标轨迹

    # 绘制追踪点
    pygame.draw.circle(screen, red, (int(chaser_pos[0]), int(chaser_pos[1])), 10)

    # 计算速度的大小
    speed = math.hypot(output[0], output[1])

    # 绘制速度方向向量箭头
    if speed > 0:  # 只有在有速度时才绘制箭头
        arrow_length = speed * 5.0  # 箭头长度与速度成正比
        direction = [output[0] / speed, output[1] / speed]  # 归一化方向向量
        arrow_end = (chaser_pos[0] + direction[0] * arrow_length, 
                     chaser_pos[1] + direction[1] * arrow_length)
        pygame.draw.line(screen, green, chaser_pos, arrow_end, 2)  # 绘制箭头为绿色

        # 绘制箭头的头部
        arrow_head_length = 10  # 箭头头部的长度
        angle = math.atan2(direction[1], direction[0])  # 计算箭头的角度
        arrow_head1 = (arrow_end[0] - arrow_head_length * math.cos(angle + math.pi / 6),
                       arrow_end[1] - arrow_head_length * math.sin(angle + math.pi / 6))
        arrow_head2 = (arrow_end[0] - arrow_head_length * math.cos(angle - math.pi / 6),
                       arrow_end[1] - arrow_head_length * math.sin(angle - math.pi / 6))

        pygame.draw.polygon(screen, green, [arrow_end, arrow_head1, arrow_head2])  # 绘制箭头头部

    # 绘制鼠标周围的绿色矩形框
    rect_width = 120
    rect_height = 90
    rect_x = target_pos[0] - rect_width // 2  # 矩形框的左上角 x 坐标
    rect_y = target_pos[1] - rect_height // 2  # 矩形框的左上角 y 坐标
    pygame.draw.rect(screen, green, (rect_x, rect_y, rect_width, rect_height), 2)  # 绘制矩形框

    # 绘制 UI
    pygame.draw.rect(screen, blue, (10, 10, 300, 150))  # UI 背景
    pygame.draw.rect(screen, white, (10, 10, 300, 150), 2)  # UI 边框

    # 显示参数信息
    accel_text = font.render(f"加速度: {output[0]:.2f}, {output[1]:.2f}", True, white)
    reset_text = font.render("按 R 键复位", True, white)  # 复位提示

    screen.blit(accel_text, (20, 20))
    screen.blit(reset_text, (20, 100))  # 显示复位提示

    # 更新上一帧误差
    previous_error = error.copy()

    # 更新屏幕
    pygame.display.flip()
    pygame.time.delay(10)  # 控制帧率

pygame.quit()