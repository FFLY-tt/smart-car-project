import sys
import random

# 固定随机种子，确保每次生成的城市外观一致
random.seed(42)

def generate_gazebo_world():
    # 1. 定义 14x14 网格路网
    grid = [[0 for _ in range(14)] for _ in range(14)]
    for x in range(14):
        grid[0][x] = 1; grid[7][x] = 1; grid[13][x] = 1
    for y in range(14):
        grid[y][0] = 1
        if y <= 7: grid[y][4] = 1
        grid[y][8] = 1; grid[y][13] = 1

    def is_road(x, y):
        return 0 <= x < 14 and 0 <= y < 14 and grid[y][x] == 1

    # SDF 文件头
    sdf = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="grid_city_base">
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>10000</real_time_update_rate>
    </physics>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
    </plugin>
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
    
    <model name="road_network">
      <static>true</static>
      <link name="link">
"""
    unit = 2.4
    element_id = 0
    
    # --- A. 生成路面、围墙、标线 ---
    for y in range(14):
        for x in range(14):
            if not is_road(x, y): continue
            cx, cy = x * unit, y * unit
            sdf += f"""        <collision name="road_col_{element_id}"><pose>{cx} {cy} 0 0 0 0</pose><geometry><box><size>{unit} {unit} 0.01</size></box></geometry></collision>
        <visual name="road_vis_{element_id}"><pose>{cx} {cy} 0 0 0 0</pose><geometry><box><size>{unit} {unit} 0.01</size></box></geometry><material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material></visual>\n"""
            
            # 边界护栏与实线 (护栏高0.5m)
            if not is_road(x, y + 1):
                sdf += f"""        <collision name="wall_t_{element_id}"><pose>{cx} {cy + 1.2} 0.25 0 0 0</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry></collision>
        <visual name="wall_tv_{element_id}"><pose>{cx} {cy + 1.2} 0.25 0 0 0</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry><material><ambient>0.6 0.6 0.6 1</ambient></material></visual>
        <visual name="line_t_{element_id}"><pose>{cx} {cy + 1.18} 0.006 0 0 0</pose><geometry><box><size>{unit} 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>\n"""
            if not is_road(x, y - 1):
                sdf += f"""        <collision name="wall_b_{element_id}"><pose>{cx} {cy - 1.2} 0.25 0 0 0</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry></collision>
        <visual name="wall_bv_{element_id}"><pose>{cx} {cy - 1.2} 0.25 0 0 0</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry><material><ambient>0.6 0.6 0.6 1</ambient></material></visual>
        <visual name="line_b_{element_id}"><pose>{cx} {cy - 1.18} 0.006 0 0 0</pose><geometry><box><size>{unit} 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>\n"""
            if not is_road(x + 1, y):
                sdf += f"""        <collision name="wall_r_{element_id}"><pose>{cx + 1.2} {cy} 0.25 0 0 1.570796</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry></collision>
        <visual name="wall_rv_{element_id}"><pose>{cx + 1.2} {cy} 0.25 0 0 1.570796</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry><material><ambient>0.6 0.6 0.6 1</ambient></material></visual>
        <visual name="line_r_{element_id}"><pose>{cx + 1.18} {cy} 0.006 0 0 1.570796</pose><geometry><box><size>{unit} 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>\n"""
            if not is_road(x - 1, y):
                sdf += f"""        <collision name="wall_l_{element_id}"><pose>{cx - 1.2} {cy} 0.25 0 0 1.570796</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry></collision>
        <visual name="wall_lv_{element_id}"><pose>{cx - 1.2} {cy} 0.25 0 0 1.570796</pose><geometry><box><size>{unit} 0.1 0.5</size></box></geometry><material><ambient>0.6 0.6 0.6 1</ambient></material></visual>
        <visual name="line_l_{element_id}"><pose>{cx - 1.18} {cy} 0.006 0 0 1.570796</pose><geometry><box><size>{unit} 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>\n"""

            if is_road(x-1, y) and is_road(x+1, y) and not is_road(x, y-1) and not is_road(x, y+1):
                sdf += f"""        <visual name="dash_h1_{element_id}"><pose>{cx - 0.6} {cy} 0.006 0 0 0</pose><geometry><box><size>0.4 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>
        <visual name="dash_h2_{element_id}"><pose>{cx + 0.6} {cy} 0.006 0 0 0</pose><geometry><box><size>0.4 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>\n"""
            elif not is_road(x-1, y) and not is_road(x+1, y) and is_road(x, y-1) and is_road(x, y+1):
                sdf += f"""        <visual name="dash_v1_{element_id}"><pose>{cx} {cy - 0.6} 0.006 0 0 1.570796</pose><geometry><box><size>0.4 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>
        <visual name="dash_v2_{element_id}"><pose>{cx} {cy + 0.6} 0.006 0 0 1.570796</pose><geometry><box><size>0.4 0.02 0.01</size></box></geometry><material><ambient>1 1 1 1</ambient></material></visual>\n"""
            element_id += 1

    # --- B. 生成建筑物和广场 ---
    blocks = [
        {"id": 1, "x": [1.2, 8.4], "y": [1.2, 15.6], "type": "dense"},
        {"id": 2, "x": [10.8, 18.0], "y": [1.2, 15.6], "type": "dense"},
        {"id": 3, "x": [20.4, 30.0], "y": [1.2, 15.6], "type": "dense"},
        {"id": 4, "x": [1.2, 18.0], "y": [18.0, 30.0], "type": "plaza"},
        {"id": 5, "x": [20.4, 30.0], "y": [18.0, 30.0], "type": "dense"}
    ]
    padding_wall, gap_bldg, b_idx = 0.8, 0.6, 0

    for b in blocks:
        ux_min, ux_max = b["x"][0] + padding_wall, b["x"][1] - padding_wall
        uy_min, uy_max = b["y"][0] + padding_wall, b["y"][1] - padding_wall

        if b["type"] == "plaza":
            plaza_start_x = ux_min + (ux_max - ux_min) * 0.4
            p_w, p_h = ux_max - plaza_start_x, uy_max - uy_min
            p_cx, p_cy = plaza_start_x + p_w / 2, uy_min + p_h / 2
            sdf += f"""
        <visual name="plaza_floor_{b['id']}"><pose>{p_cx} {p_cy} 0.015 0 0 0</pose><geometry><box><size>{p_w} {p_h} 0.01</size></box></geometry><material><ambient>0.7 0.65 0.55 1</ambient></material></visual>
        <collision name="monument_col"><pose>{p_cx} {p_cy} 1.5 0 0 0</pose><geometry><cylinder><radius>0.6</radius><length>3.0</length></cylinder></geometry></collision>
        <visual name="monument_vis"><pose>{p_cx} {p_cy} 1.5 0 0 0</pose><geometry><cylinder><radius>0.6</radius><length>3.0</length></cylinder></geometry><material><ambient>0.2 0.2 0.2 1</ambient></material></visual>\n"""
            ux_max = plaza_start_x

        cell_size = 3.5
        cols, rows = max(1, int((ux_max - ux_min) / cell_size)), max(1, int((uy_max - uy_min) / cell_size))
        act_cw, act_ch = (ux_max - ux_min) / cols, (uy_max - uy_min) / rows
        
        for r in range(rows):
            for c in range(cols):
                cx, cy = ux_min + c * act_cw + act_cw / 2, uy_min + r * act_ch + act_ch / 2
                bw, bh, bhz = act_cw - gap_bldg - random.uniform(0, 0.5), act_ch - gap_bldg - random.uniform(0, 0.5), random.uniform(2.0, 8.0)
                gray = random.uniform(0.4, 0.8)
                sdf += f"""        <collision name="bldg_col_{b_idx}"><pose>{cx:.3f} {cy:.3f} {bhz/2:.3f} 0 0 0</pose><geometry><box><size>{bw:.3f} {bh:.3f} {bhz:.3f}</size></box></geometry></collision>
        <visual name="bldg_vis_{b_idx}"><pose>{cx:.3f} {cy:.3f} {bhz/2:.3f} 0 0 0</pose><geometry><box><size>{bw:.3f} {bh:.3f} {bhz:.3f}</size></box></geometry><material><ambient>{gray:.2f} {gray:.2f} {gray+0.05:.2f} 1</ambient></material></visual>\n"""
                b_idx += 1
    # 封口 Link (原道路环境完毕)
    sdf += """      </link>
    </model>\n"""

    # --- C. 起点和终点区域 ---
    s_cx, s_cy = 4 * 2.4, 3 * 2.4 - 0.6
    sdf += f"""    <model name="start_zone"><static>true</static><pose>{s_cx} {s_cy} 0.015 0 0 0</pose><link name="link"><visual name="vis"><geometry><box><size>2.4 1.2 0.01</size></box></geometry><material><ambient>0.5 0.8 0.2 1</ambient></material></visual></link></model>\n"""
    e_cx, e_cy = 13 * 2.4, 11 * 2.4 + 0.6
    sdf += f"""    <model name="end_zone"><static>true</static><pose>{e_cx} {e_cy} 0.015 0 0 0</pose><link name="link"><visual name="vis"><geometry><box><size>2.4 1.2 0.01</size></box></geometry><material><ambient>1.0 0.6 0.0 1</ambient></material></visual></link></model>\n"""

    # --- D. 动态与静态任务元素 (回归基础几何体) ---
    
    # 1. P1 (行人圆柱体) -> (x=4, y=5) 纵向道路最左侧，蓝色
    p1_cx, p1_cy = 4 * 2.4 - 1.0, 5 * 2.4
    sdf += f"""
    <model name="P1_pedestrian">
      <pose>{p1_cx} {p1_cy} 0.0875 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><cylinder><radius>0.025</radius><length>0.175</length></cylinder></geometry></collision>
        <visual name="vis"><geometry><cylinder><radius>0.025</radius><length>0.175</length></cylinder></geometry><material><ambient>0 0.5 1 1</ambient></material></visual>
      </link>
    </model>\n"""

    # 2. P2 (行人圆柱体) -> (x=10, y=7) 横向道路下侧，蓝色
    p2_cx, p2_cy = 10 * 2.4, 7 * 2.4 - 1.0
    sdf += f"""
    <model name="P2_pedestrian">
      <pose>{p2_cx} {p2_cy} 0.0875 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><cylinder><radius>0.025</radius><length>0.175</length></cylinder></geometry></collision>
        <visual name="vis"><geometry><cylinder><radius>0.025</radius><length>0.175</length></cylinder></geometry><material><ambient>0 0.5 1 1</ambient></material></visual>
      </link>
    </model>\n"""

    # 3. O1 (路桩圆柱体) -> (x=6, y=7) 横向道路中心，橙色
    o1_cx, o1_cy = 6 * 2.4, 7 * 2.4
    sdf += f"""
    <model name="O1_barrel">
      <static>true</static><pose>{o1_cx} {o1_cy} 0.07 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><cylinder><radius>0.04</radius><length>0.14</length></cylinder></geometry></collision>
        <visual name="vis"><geometry><cylinder><radius>0.04</radius><length>0.14</length></cylinder></geometry><material><ambient>1 0.5 0 1</ambient></material></visual>
      </link>
    </model>\n"""

    # 4. O2 (路桩圆柱体) -> (x=13, y=9) 纵向道路中心，橙色
    o2_cx, o2_cy = 13 * 2.4, 9 * 2.4
    sdf += f"""
    <model name="O2_barrel">
      <static>true</static><pose>{o2_cx} {o2_cy} 0.07 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><cylinder><radius>0.04</radius><length>0.14</length></cylinder></geometry></collision>
        <visual name="vis"><geometry><cylinder><radius>0.04</radius><length>0.14</length></cylinder></geometry><material><ambient>1 0.5 0 1</ambient></material></visual>
      </link>
    </model>\n"""

    sdf += """  </world>\n</sdf>"""
    print(sdf)

if __name__ == "__main__":
    generate_gazebo_world()