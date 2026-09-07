#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气囊基因线路可视化系统

该应用使用Streamlit实现气囊基因线路的可视化，包括：
1. 线路结构图
2. 参数变化vs输出曲线
3. 用户交互参数修改
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import hashlib
import os
import json
from gene_circuit_model import GasVesicleCircuit
from modular_ai_design import (
    ModularElementLibrary,
    train_predictor_from_library,
    predict_design_expression,
    simulate_design_in_circuit,
)

# 设置Matplotlib支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置页面标题和布局
st.set_page_config(
    page_title="基因线路设计与仿真结果可视化系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加CSS样式实现表格内容垂直居中并优化整体样式
st.markdown("""
<style>
/* 全局样式 */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* 表格内容垂直居中 */
.stDataFrame {
    display: flex;
    justify-content: center;
    margin: 1rem 0;
}

.stDataFrame table {
    margin: 0 auto;
    border-collapse: collapse;
}

.stDataFrame th,
.stDataFrame td {
    vertical-align: middle !important;
    text-align: center !important;
    padding: 0.5rem !important;
}

/* 原生表格垂直居中 */
.stTable {
    display: flex;
    justify-content: center;
    margin: 1rem 0;
}

.stTable table {
    margin: 0 auto;
    border-collapse: collapse;
    width: 100%;
    max-width: 800px;
}

.stTable th,
.stTable td {
    vertical-align: middle !important;
    text-align: center !important;
    padding: 0.5rem !important;
}

/* 列容器中的表格居中 */
[data-testid="column"] .stTable,
[data-testid="column"] .stDataFrame {
    justify-content: center;
}

/* 卡片样式 */
.stCard {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    padding: 1rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* 标题样式 */
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50;
    margin-bottom: 1rem;
}

/* 按钮样式 */
.stButton > button {
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

/* 滑块样式 */
.stSlider > div {
    margin: 0.5rem 0;
}

/* 选择框样式 */
.stSelectbox > div {
    margin: 0.5rem 0;
}

/* 标签页样式 */
.stTabs {
    margin-bottom: 1.5rem;
}

/* 侧边栏样式 */
.stSidebar {
    background-color: #f8f9fa;
    padding: 1rem;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .stDataFrame, .stTable {
        overflow-x: auto;
    }
    
    .stColumns {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)

# 标题
st.title("🧬 气囊基因线路设计与仿真结果可视化系统")

# 初始化基因线路模型
circuit = GasVesicleCircuit()

# 数据缓存，用于存储已读取的数据，避免重复读取和处理
# 键为文件路径，值为数据内容
DATA_CACHE = {
    'metadata': {},      # 元数据缓存
    'sequence': {},      # 序列数据缓存
    'experimental': {},  # 实验数据缓存
    'uploaded': {}       # 用户上传数据缓存
}

# 缓存访问时间，用于LRU策略
CACHE_ACCESS = {
    'metadata': {},
    'sequence': {},
    'experimental': {},
    'uploaded': {}
}

# 缓存大小限制
MAX_CACHE_SIZE = 10


def get_uploaded_file_key(prefix, uploaded_file):
    """Build a stable cache key for Streamlit UploadedFile objects."""
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    file_size = getattr(uploaded_file, 'size', len(file_bytes))
    file_name = getattr(uploaded_file, 'name', 'uploaded_file')
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return f"{prefix}_{file_name}_{file_size}_{file_hash}"


def get_cached_data(cache_type, key):
    """获取缓存数据
    
    Args:
        cache_type: 缓存类型 ('metadata', 'sequence', 'experimental', 'uploaded')
        key: 缓存键（通常是文件路径）
    
    Returns:
        缓存的数据，如果不存在返回None
    """
    if cache_type in DATA_CACHE and key in DATA_CACHE[cache_type]:
        # 更新缓存访问时间
        CACHE_ACCESS[cache_type][key] = datetime.now().timestamp()
        return DATA_CACHE[cache_type][key]
    return None


def add_cached_data(cache_type, key, data):
    """添加数据到缓存
    
    Args:
        cache_type: 缓存类型 ('metadata', 'sequence', 'experimental', 'uploaded')
        key: 缓存键（通常是文件路径）
        data: 要缓存的数据
    """
    if cache_type not in DATA_CACHE:
        return
    
    # 如果缓存已满，使用LRU策略移除最旧的缓存项
    if len(DATA_CACHE[cache_type]) >= MAX_CACHE_SIZE:
        if CACHE_ACCESS[cache_type]:
            # 找到最近最少使用的缓存键
            oldest_key = min(CACHE_ACCESS[cache_type], key=CACHE_ACCESS[cache_type].get)
            # 移除最旧的缓存项
            if oldest_key in DATA_CACHE[cache_type]:
                del DATA_CACHE[cache_type][oldest_key]
            if oldest_key in CACHE_ACCESS[cache_type]:
                del CACHE_ACCESS[cache_type][oldest_key]
    
    # 添加数据到缓存
    DATA_CACHE[cache_type][key] = data
    # 记录缓存访问时间
    CACHE_ACCESS[cache_type][key] = datetime.now().timestamp()

# 侧边栏：参数设置
with st.sidebar:
    st.header("参数设置")
    
    # 使用tabs分组显示参数，优化标签页名称
    tab1, tab2, tab3, tab4 = st.tabs([
        "基本设置", "基因参数", "调控关系", "模板管理"
    ])
    
    with tab1:
        # 基本设置
        st.subheader("基本设置")
        # 选择要可视化的基因
        selected_genes = st.multiselect(
            "选择要显示的基因",
            options=circuit.genes,
            default=["gvpA", "gvpC", "gvpE", "gvpN"]
        )
        # 选择要显示的分子类型
        molecule_type = st.radio(
            "选择分子类型",
            options=["mRNA", "protein"],
            index=1,
            horizontal=True
        )
        
        # 仿真设置
        st.subheader("仿真设置")
        # 使用容器组织滑块，使界面更整洁
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                t_start = st.slider("起始时间", 0, 40, 0)
                t_end = st.slider("结束时间", 50, 200, 100)
            with col2:
                t_points = st.slider("时间点数量", 50, 500, 200)
        
        t_eval = np.linspace(t_start, t_end, t_points)
    
    with tab2:
        # 基因表达参数
        st.subheader("基因表达参数")
        # 选择要调整的基因
        param_gene = st.selectbox("选择基因", circuit.genes)
        
        # 使用容器组织滑块，使界面更整洁
        with st.container():
            # 调整转录速率
            alpha = st.slider(
                f"{param_gene} 转录速率 (alpha)",
                0.0, 2.0, circuit.params['alpha'][param_gene], 0.05
            )
            circuit.set_parameter('alpha', param_gene, alpha)
            
            # 调整翻译速率
            beta = st.slider(
                f"{param_gene} 翻译速率 (beta)",
                0.0, 2.0, circuit.params['beta'][param_gene], 0.05
            )
            circuit.set_parameter('beta', param_gene, beta)
            
            # 调整降解速率
            gamma = st.slider(
                f"{param_gene} 降解速率 (gamma)",
                0.0, 1.0, circuit.params['gamma'][param_gene], 0.01
            )
            circuit.set_parameter('gamma', param_gene, gamma)
    
    with tab3:
        # 调控关系调整
        st.subheader("调控关系")
        # 使用容器组织选择框和滑块，使界面更整洁
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                regulator = st.selectbox("调控基因", circuit.genes)
            with col2:
                regulated = st.selectbox("被调控基因", circuit.genes)
            
            regulation_strength = st.slider(
                f"{regulator} → {regulated} 调控强度",
                -1.0, 1.0, 0.0, 0.05
            )
            
            if st.button("更新调控关系", use_container_width=True):
                circuit.set_regulation(regulated, regulator, regulation_strength)
                st.success(f"已更新 {regulator} 对 {regulated} 的调控强度为 {regulation_strength}")
    
    with tab4:
        # 模板管理
        st.subheader("模板管理")
        
        # 模板加载
        st.subheader("加载模板")
        
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(template_dir, exist_ok=True)
        
        # 获取所有模板文件
        template_files = [f for f in os.listdir(template_dir) if f.endswith('.json')]
        
        if template_files:
            # 添加内置模板选项
            template_options = ["默认设置"] + [f[:-5] for f in template_files]
            selected_template = st.selectbox("选择模板", template_options)
            
            if st.button("加载所选模板", use_container_width=True):
                if selected_template == "默认设置":
                    # 重置到默认值
                    circuit = GasVesicleCircuit()
                    st.success("已加载默认设置")
                    st.experimental_rerun()
                else:
                    # 加载自定义模板
                    template_path = os.path.join(template_dir, f"{selected_template}.json")
                    
                    with open(template_path, 'r', encoding='utf-8') as f:
                        template = json.load(f)
                    
                    # 加载模板设置
                    settings = template["settings"]
                    
                    # 重置电路
                    circuit = GasVesicleCircuit()
                    
                    # 更新参数
                    circuit.params['alpha'] = settings["params"]["alpha"]
                    circuit.params['beta'] = settings["params"]["beta"]
                    circuit.params['gamma'] = settings["params"]["gamma"]
                    circuit.params['regulation'] = np.array(settings["params"]["regulation"])
                    
                    st.success(f"已加载模板: {selected_template}")
                    st.experimental_rerun()
        else:
            st.info("暂无保存的模板")
        
        # 模板保存
        st.subheader("保存模板")
        with st.container():
            template_name = st.text_input("模板名称", value="my_template")
            template_description = st.text_area("模板描述", value="自定义模板", height=80)
            
            if st.button("保存当前设置为模板", use_container_width=True):
                # 收集当前设置
                template = {
                    "name": template_name,
                    "description": template_description,
                    "date": str(datetime.now()),
                    "settings": {
                        "selected_genes": selected_genes,
                        "molecule_type": molecule_type,
                        "t_start": t_start,
                        "t_end": t_end,
                        "t_points": t_points,
                        "params": {
                            "alpha": circuit.params['alpha'],
                            "beta": circuit.params['beta'],
                            "gamma": circuit.params['gamma'],
                            "regulation": circuit.params['regulation'].tolist()
                        }
                    }
                }
                
                # 保存到文件
                template_dir = os.path.join(os.path.dirname(__file__), "templates")
                os.makedirs(template_dir, exist_ok=True)
                template_path = os.path.join(template_dir, f"{template_name}.json")
                
                with open(template_path, 'w', encoding='utf-8') as f:
                    json.dump(template, f, ensure_ascii=False, indent=2)
                
                st.success(f"模板已保存到 {template_path}")
    
    # 添加参数重置按钮，放在侧边栏底部
    st.subheader("重置设置")
    if st.button("重置所有参数", type="primary", use_container_width=True):
        # 重置到默认值
        circuit = GasVesicleCircuit()
        st.success("所有参数已重置为默认值")
        # 刷新页面
        st.experimental_rerun()

# 主页面：选项卡布局

# 创建选项卡，优化标签名称和顺序，使导航更清晰
main_tabs = st.tabs([
    "线路结构图", 
    "表达动态", 
    "参数分析", 
    "路径分析", 
    "模型验证", 
    "数据管理", 
    "报告生成",
    "图表中心"
])

# 运行仿真
results = circuit.simulate(t_span=(t_start, t_end), t_eval=t_eval)

# 选项卡1：线路结构图
with main_tabs[0]:
    st.header("1. 线路结构图")
    
    # 可视化类型选择，使用水平布局节省空间
    viz_type = st.radio(
        "选择可视化类型",
        options=["2D可视化", "3D可视化", "动态网络"],
        index=0,
        horizontal=True
    )
    
    # 获取网络结构
    network = circuit.get_gene_network()
    
    # 创建NetworkX图
    G = nx.DiGraph()
    
    # 添加节点
    for node in network['nodes']:
        G.add_node(node['id'], label=node['label'])
    
    # 添加边
    for edge in network['edges']:
        G.add_edge(
            edge['source'], 
            edge['target'],
            weight=edge['weight'],
            type=edge['type']
        )
    
    # 网络结构哈希，用于缓存布局
    network_hash = hash(tuple(sorted(G.nodes())) + tuple(sorted(G.edges())))
    
    # 检查是否已有缓存的布局
    layout_cache_key = f"layout_{viz_type}_{network_hash}"
    layout = get_cached_data('metadata', layout_cache_key)  # 使用metadata缓存存储布局

    if viz_type == "2D可视化":
        # 绘制2D图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 布局 - 使用缓存避免重复计算
        if layout is None:
            pos = nx.spring_layout(G, seed=42, k=0.6)
            # 添加到缓存
            add_cached_data('metadata', layout_cache_key, pos)
        else:
            pos = layout
        
        # 使用统一的颜色方案
        node_color = '#64B5F6'
        activation_color = '#81C784'
        inhibition_color = '#E57373'
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=1000, 
            node_color=node_color,
            ax=ax,
            alpha=0.8
        )
        
        # 绘制边
        edges = G.edges(data=True)
        
        # 分开激活和抑制边
        activation_edges = [(u, v) for u, v, d in edges if d['type'] == 'activation']
        inhibition_edges = [(u, v) for u, v, d in edges if d['type'] == 'inhibition']
        
        # 绘制激活边
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=activation_edges,
            edge_color=activation_color,
            style='solid',
            width=2,
            ax=ax
        )
        
        # 绘制抑制边
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=inhibition_edges,
            edge_color=inhibition_color,
            style='dashed',
            width=2,
            ax=ax
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(
            G, pos, 
            font_size=12, 
            font_weight='bold',
            ax=ax
        )
        
        # 绘制边标签（调控强度）
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels,
            font_size=10,
            ax=ax
        )
        
        # 设置图形样式
        ax.set_title("气囊基因调控网络", fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
    elif viz_type == "3D可视化":
        # 绘制3D图形
        # 3D布局 - 使用缓存避免重复计算
        if layout is None:
            pos = nx.spring_layout(G, seed=42, k=0.5, dim=3)
            # 添加到缓存
            add_cached_data('metadata', layout_cache_key, pos)
        else:
            pos = layout
        
        # 创建节点坐标列表
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(f"{node}")
            node_color.append("#64B5F6")
        
        # 创建边坐标列表
        edge_x = []
        edge_y = []
        edge_z = []
        edge_color = []
        
        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            
            # 根据调控类型设置边的颜色
            if edge[2]['type'] == 'activation':
                edge_color.append('#81C784')
            else:
                edge_color.append('#E57373')
        
        # 创建3D图形
        fig = go.Figure()
        
        # 添加边
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                color=edge_color,
                width=2
            ),
            opacity=0.7
        ))
        
        # 添加节点
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=10,
                color=node_color,
                opacity=0.9
            ),
            text=node_text,
            textposition="top center",
            textfont=dict(
                size=12,
                color="black"
            )
        ))
        
        # 设置图形布局
        fig.update_layout(
            title="气囊基因调控网络（3D）",
            scene=dict(
                xaxis_title="",
                yaxis_title="",
                zaxis_title="",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=40),
            height=600
        )
        
        # 显示3D图形
        st.plotly_chart(fig, use_container_width=True)
    else:  # 动态网络
        # 运行仿真获取时间序列数据
        t_eval_dynamic = np.linspace(0, 50, 20)  # 20个时间点
        
        # 缓存仿真结果
        dynamic_sim_cache_key = f"dynamic_sim_{network_hash}_{tuple(t_eval_dynamic)}_50"
        results_dynamic = get_cached_data('sequence', dynamic_sim_cache_key)  # 使用sequence缓存存储动态仿真结果
        
        if results_dynamic is None:
            results_dynamic = circuit.simulate(t_span=(0, 50), t_eval=t_eval_dynamic)
            add_cached_data('sequence', dynamic_sim_cache_key, results_dynamic)
        
        # 2D布局（固定）- 使用缓存避免重复计算
        if layout is None:
            pos = nx.spring_layout(G, seed=42, k=0.5)
            # 添加到缓存
            add_cached_data('metadata', layout_cache_key, pos)
        else:
            pos = layout
        
        # 创建时间点选择器
        selected_time = st.slider(
            "选择时间点",
            min_value=0,
            max_value=len(t_eval_dynamic)-1,
            value=0,
            format="%d"
        )
        
        # 绘制当前时间点的网络
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 根据基因表达水平计算节点大小和颜色
        protein_conc = results_dynamic['protein']
        node_size = []
        node_color = []
        
        for node in G.nodes():
            if node in protein_conc:
                # 节点大小：根据蛋白质浓度调整，范围200-1800
                size = 200 + protein_conc[node][selected_time] * 1200
                node_size.append(max(200, min(1800, size)))
                # 节点颜色：根据蛋白质浓度调整，从浅蓝色到深蓝色
                color_intensity = min(1.0, protein_conc[node][selected_time] / 2.0)
                # 使用渐变色
                r = 100 + int(65 * color_intensity)
                g = 181 + int(74 * color_intensity)
                b = 246 - int(90 * color_intensity)
                node_color.append(f'#{r:02x}{g:02x}{b:02x}')
            else:
                node_size.append(1000)
                node_color.append('#64B5F6')
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_size, 
            node_color=node_color,
            ax=ax
        )
        
        # 绘制边
        edges = G.edges(data=True)
        
        # 分开激活和抑制边
        activation_edges = [(u, v) for u, v, d in edges if d['type'] == 'activation']
        inhibition_edges = [(u, v) for u, v, d in edges if d['type'] == 'inhibition']
        
        # 根据调控强度和当前时间点的调控效果调整边的宽度
        edge_width_activation = []
        for u, v, d in G.edges(data=True):
            if d['type'] == 'activation':
                # 边的宽度：根据调控强度和蛋白质浓度调整
                if u in protein_conc:
                    width = abs(d['weight']) * (1 + protein_conc[u][selected_time]) * 2
                    edge_width_activation.append(max(1, min(5, width)))
                else:
                    edge_width_activation.append(2)
        
        edge_width_inhibition = []
        for u, v, d in G.edges(data=True):
            if d['type'] == 'inhibition':
                # 边的宽度：根据调控强度和蛋白质浓度调整
                if u in protein_conc:
                    width = abs(d['weight']) * (1 + protein_conc[u][selected_time]) * 2
                    edge_width_inhibition.append(max(1, min(5, width)))
                else:
                    edge_width_inhibition.append(2)
        
        # 绘制激活边（绿色，实线）
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=activation_edges,
            edge_color='#81C784',
            style='solid',
            width=edge_width_activation,
            ax=ax
        )
        
        # 绘制抑制边（红色，虚线）
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=inhibition_edges,
            edge_color='#E57373',
            style='dashed',
            width=edge_width_inhibition,
            ax=ax
        )
        
        # 绘制节点标签
        nx.draw_networkx_labels(
            G, pos, 
            font_size=12, 
            font_weight='bold',
            ax=ax
        )
        
        # 绘制边标签（调控强度）
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(
            G, pos, 
            edge_labels=edge_labels,
            font_size=10,
            ax=ax
        )
        
        # 设置图形样式
        ax.set_title(f"动态基因调控网络（时间点：{selected_time+1}/{len(t_eval_dynamic)}，t={t_eval_dynamic[selected_time]:.1f}）", fontsize=16, pad=20)
        ax.axis('off')
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
        
        # 显示当前时间点的蛋白质浓度，使用卡片式布局
        with st.container():
            st.subheader(f"时间点 {selected_time+1}/{len(t_eval_dynamic)} 的蛋白质浓度")
            conc_data = {node: protein_conc[node][selected_time] for node in G.nodes() if node in protein_conc}
            conc_df = pd.DataFrame(list(conc_data.items()), columns=["基因", "蛋白质浓度"])
            # 使用数据框显示，支持排序和筛选
            st.dataframe(conc_df, use_container_width=True)

    # 显示网络统计信息，使用卡片式布局
    with st.container():
        st.subheader("网络统计信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("节点数量", len(network['nodes']))
        with col2:
            st.metric("边数量", len(network['edges']))
        with col3:
            # 计算网络密度
            density = nx.density(G)
            st.metric("网络密度", f"{density:.4f}")

# 选项卡2：表达动态
with main_tabs[1]:
    st.header("2. 表达动态")

        # 创建2×2的网格
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    # 绘制折线图 - 左上
    with col1:
        st.subheader("折线图")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 使用统一的颜色方案
        line_colors = ['#64B5F6', '#81C784', '#FFB74D', '#E57373', '#9575CD', '#4DB6AC']
        line_width = 2.0
        
        for i, gene in enumerate(selected_genes):
            if gene in results[molecule_type]:
                ax.plot(
                    results['t'],
                    results[molecule_type][gene],
                    label=gene,
                    linewidth=line_width,
                    color=line_colors[i % len(line_colors)]
                )
        
        # 设置图形样式
        ax.set_title(f"{molecule_type} 浓度随时间变化", fontsize=14)
        ax.set_xlabel("时间", fontsize=10)
        ax.set_ylabel(f"{molecule_type} 浓度", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
    
    # 绘制面积图 - 右上
    with col2:
        st.subheader("面积图")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, gene in enumerate(selected_genes):
            if gene in results[molecule_type]:
                ax.fill_between(
                    results['t'],
                    results[molecule_type][gene],
                    alpha=0.3,
                    label=gene,
                    color=line_colors[i % len(line_colors)]
                )
        
        # 设置图形样式
        ax.set_title(f"{molecule_type} 浓度随时间变化", fontsize=14)
        ax.set_xlabel("时间", fontsize=10)
        ax.set_ylabel(f"{molecule_type} 浓度", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
    
    # 绘制散点图 - 左下
    with col3:
        st.subheader("散点图")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        marker_size = 8
        
        for i, gene in enumerate(selected_genes):
            if gene in results[molecule_type]:
                ax.scatter(
                    results['t'],
                    results[molecule_type][gene],
                    label=gene,
                    s=marker_size * 5,
                    alpha=0.7,
                    color=line_colors[i % len(line_colors)]
                )
        
        # 设置图形样式
        ax.set_title(f"{molecule_type} 浓度随时间变化", fontsize=14)
        ax.set_xlabel("时间", fontsize=10)
        ax.set_ylabel(f"{molecule_type} 浓度", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        
        # 显示图形
        st.pyplot(fig)
    
    # 绘制雷达图 - 右下
    with col4:
        st.subheader("雷达图")
        # 只显示部分时间点数据用于雷达图
        time_indices = np.linspace(0, len(results['t'])-1, 5, dtype=int)  # 5个时间点
        time_points = results['t'][time_indices]
        
        # 准备雷达图数据
        radar_data = {
            'time': time_points,
            'genes': selected_genes,
            'values': []
        }
        
        for gene in selected_genes:
            if gene in results[molecule_type]:
                radar_data['values'].append(results[molecule_type][gene][time_indices])
        
        # 使用Plotly绘制雷达图
        fig = go.Figure()
        
        for i, gene in enumerate(selected_genes):
            if gene in results[molecule_type]:
                fig.add_trace(go.Scatterpolar(
                    r=radar_data['values'][i],
                    theta=[f"t={t:.1f}" for t in time_points],
                    fill='toself',
                    name=gene,
                    line=dict(color=line_colors[i % len(line_colors)])
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(vals) for vals in radar_data['values']) * 1.2]
                )),
            showlegend=True,
            title=f"{molecule_type} 浓度雷达图",
            width=600,
            height=400
        )
        
        # 显示雷达图
        st.plotly_chart(fig, use_container_width=True)
    
    # 显示最终浓度值，使用数据框
    st.subheader("最终浓度值")
    
    final_concentrations = {}
    for gene in selected_genes:
        if gene in results[molecule_type]:
            final_concentrations[gene] = results[molecule_type][gene][-1]
    
    # 将结果转换为表格
    df_final = pd.DataFrame(
        list(final_concentrations.items()),
        columns=["基因", f"最终{'' if molecule_type == 'protein' else 'mRNA'}浓度"]
    )
    
    # 使用数据框显示，支持排序和筛选
    st.dataframe(df_final, use_container_width=True)

# 选项卡3：参数分析
with main_tabs[2]:
    st.header("3. 参数分析")
    
    # 分析类型选择
    analysis_type = st.radio(
        "选择分析类型",
        options=["参数敏感性", "元数据分析", "序列数据分析", "AI组合预测"],
        index=0,
        horizontal=True
    )
    
    if analysis_type == "参数敏感性":
        # 参数选择区域
        with st.container():
            st.subheader("参数敏感性分析")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                sensitivity_param = st.selectbox(
                    "参数类型",
                    options=['alpha', 'beta', 'gamma'],
                    index=0
                )
            with col2:
                sensitivity_gene = st.selectbox(
                    "分析基因",
                    circuit.genes,
                    index=0
                )
            with col3:
                analysis_metric = st.selectbox(
                    "分析指标",
                    options=["最终浓度", "峰值浓度", "达到峰值时间", "曲线下面积"],
                    index=0
                )
        
        # 自动运行敏感性分析
        with st.spinner("正在运行敏感性分析..."):
                # 保存原始参数值
                original_value = circuit.params[sensitivity_param][sensitivity_gene]
                
                # 参数变化范围
                param_range = np.linspace(original_value * 0.5, original_value * 1.5, 10)
                
                # 存储结果
                sensitivity_results = []
                
                for param_value in param_range:
                    # 设置新参数值
                    circuit.set_parameter(sensitivity_param, sensitivity_gene, param_value)
                    
                    # 运行仿真
                    results_sensitivity = circuit.simulate(t_span=(t_start, t_end), t_eval=t_eval)
                    
                    # 计算分析指标
                    analysis_values = {}
                    for gene in selected_genes:
                        if gene in results_sensitivity[molecule_type]:
                            data = results_sensitivity[molecule_type][gene]
                            if analysis_metric == "最终浓度":
                                analysis_values[gene] = data[-1]
                            elif analysis_metric == "峰值浓度":
                                analysis_values[gene] = np.max(data)
                            elif analysis_metric == "达到峰值时间":
                                peak_idx = np.argmax(data)
                                analysis_values[gene] = results_sensitivity['t'][peak_idx]
                            else:  # 曲线下面积
                                analysis_values[gene] = np.trapz(data, results_sensitivity['t'])
                    
                    sensitivity_results.append(analysis_values)
                
                # 恢复原始参数值
                circuit.set_parameter(sensitivity_param, sensitivity_gene, original_value)
                
                # 创建2×2的网格布局
                col1, col2 = st.columns(2)
                
                # 绘制敏感性曲线 - 左上
                with col1:
                    st.subheader("敏感性曲线")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # 使用统一的颜色方案
                    line_colors = ['#64B5F6', '#81C784', '#FFB74D', '#E57373', '#9575CD', '#4DB6AC']
                    
                    for i, gene in enumerate(selected_genes):
                        concs = [res[gene] for res in sensitivity_results]
                        ax.plot(param_range, concs, label=gene, linewidth=2.0, marker='o', markersize=6, color=line_colors[i % len(line_colors)])
                    
                    # 设置图形样式
                    ax.set_title(f"{sensitivity_gene} {sensitivity_param} 参数敏感性", fontsize=14)
                    ax.set_xlabel(f"{sensitivity_param} 参数值", fontsize=10)
                    ax.set_ylabel(f"{analysis_metric}", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, loc='best')
                    plt.tight_layout()
                    
                    # 显示图形
                    st.pyplot(fig)
                
                # 绘制敏感性热力图 - 右上
                with col2:
                    st.subheader("敏感性热力图")
                    # 准备热力图数据
                    heatmap_data = np.zeros((len(selected_genes), len(param_range)))
                    for i, gene in enumerate(selected_genes):
                        for j, param_value in enumerate(param_range):
                            heatmap_data[i, j] = sensitivity_results[j][gene]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(heatmap_data, cmap='coolwarm', aspect='auto')
                    
                    # 添加色标
                    cbar = ax.figure.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel(f"{analysis_metric}", rotation=-90, va="bottom", fontsize=10)
                    
                    # 设置坐标轴
                    ax.set_xticks(range(len(param_range)))
                    ax.set_yticks(range(len(selected_genes)))
                    ax.set_xticklabels([f"{p:.2f}" for p in param_range], rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
                    ax.set_yticklabels(selected_genes, fontsize=10)
                    
                    ax.set_title(f"{sensitivity_gene} {sensitivity_param} 敏感性热力图", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 添加敏感性分析结果下载功能
                st.subheader("敏感性分析结果")
                
                # 准备敏感性分析结果数据
                sensitivity_df = pd.DataFrame(param_range, columns=[f'{sensitivity_param}参数值'])
                for gene in selected_genes:
                    sensitivity_df[gene] = [res[gene] for res in sensitivity_results]
                
                # 显示敏感性结果表格
                st.dataframe(sensitivity_df, use_container_width=True)
                
                # 转换为CSV格式
                sensitivity_csv = sensitivity_df.to_csv(index=False, encoding='utf-8-sig')
                
                # 提供下载按钮
                st.download_button(
                    label=f"下载{sensitivity_gene} {sensitivity_param} 敏感性分析结果 (CSV)",
                    data=sensitivity_csv,
                    file_name=f"sensitivity_analysis_{sensitivity_gene}_{sensitivity_param}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # 添加敏感性统计分析
                st.subheader("敏感性统计分析")
                
                # 计算敏感性系数
                sensitivity_coefficients = {}
                for gene in selected_genes:
                    data = np.array([res[gene] for res in sensitivity_results])
                    # 计算皮尔逊相关系数作为敏感性指标
                    coeff = np.corrcoef(param_range, data)[0, 1]
                    sensitivity_coefficients[gene] = coeff
                
                # 显示敏感性系数
                coeff_df = pd.DataFrame.from_dict(sensitivity_coefficients, orient='index', columns=["敏感性系数"])
                coeff_df = coeff_df.sort_values(by="敏感性系数", ascending=False)
                # 重置索引，将基因名从索引转换为列
                coeff_df = coeff_df.reset_index().rename(columns={"index": "基因"})
                
                # 绘制敏感性系数条形图
                fig, ax = plt.subplots(figsize=(10, 6))
                # 使用统一的颜色方案
                colors = ['#64B5F6' if coeff > 0 else '#E57373' for coeff in coeff_df["敏感性系数"]]
                ax.bar(coeff_df["基因"], coeff_df["敏感性系数"], color=colors)
                ax.set_title(f"{sensitivity_gene} {sensitivity_param} 敏感性系数", fontsize=14)
                ax.set_xlabel("基因", fontsize=10)
                ax.set_ylabel("敏感性系数", fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示敏感性系数表格
                st.dataframe(coeff_df, use_container_width=True)
    
    elif analysis_type == "元数据分析":
        st.subheader("元数据分析")
        
        # 元数据文件夹路径
        metadata_dir = os.path.join(os.path.dirname(__file__), "../data/gene_metadata")
        
        # 获取所有元数据文件
        try:
            metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('_metadata.csv')]
            
            if metadata_files:
                # 选择要查看的元数据文件
                selected_file = st.selectbox(
                    "选择基因元数据文件",
                    metadata_files
                )
                
                if selected_file:
                    # 读取元数据，使用缓存避免重复读取
                    file_path = os.path.join(metadata_dir, selected_file)
                    
                    # 检查缓存
                    df = get_cached_data('metadata', file_path)
                    
                    if df is None:
                        # 缓存中没有，读取文件
                        df = pd.read_csv(file_path)
                        # 添加到缓存
                        add_cached_data('metadata', file_path, df)
                    
                    # 显示元数据基本信息
                    st.subheader(f"{selected_file[:-14]} 基因元数据")
                    
                    # 显示数据概览，使用卡片式布局
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("数据行数", len(df))
                        with col2:
                            st.metric("数据列数", len(df.columns))
                        with col3:
                            # 计算非空值比例
                            non_null_ratio = df.notnull().sum().sum() / (len(df) * len(df.columns)) * 100
                            st.metric("数据完整性", f"{non_null_ratio:.2f}%")
                    
                    # 显示数据前几行
                    st.subheader("数据预览")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # 显示数据统计信息
                    st.subheader("数据统计信息")
                    st.dataframe(df.describe(), use_container_width=True)
                    
                    # 基因长度分布直方图和生物体分布饼图 - 2×1布局
                    if 'sequence_length' in df.columns or 'organism' in df.columns:
                        st.subheader("分布分析")
                        if 'sequence_length' in df.columns and 'organism' in df.columns:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("基因长度分布")
                                fig, ax = plt.subplots(figsize=(7, 5))
                                ax.hist(df['sequence_length'], bins=20, alpha=0.7, color='#64B5F6', edgecolor='black')
                                ax.set_title(f"{selected_file[:-14]} 基因长度分布", fontsize=12)
                                ax.set_xlabel("序列长度", fontsize=10)
                                ax.set_ylabel("数量", fontsize=10)
                                ax.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                            with col2:
                                st.subheader("生物体分布")
                                # 取前10个最常见的生物体
                                top_organisms = df['organism'].value_counts().head(10)
                                
                                fig, ax = plt.subplots(figsize=(7, 5))
                                ax.pie(top_organisms.values, labels=top_organisms.index, autopct='%1.1f%%', startangle=90, colors=['#64B5F6', '#81C784', '#FFB74D', '#E57373', '#9575CD', '#4DB6AC', '#4FC3F7', '#81D4FA', '#B3E5FC', '#E1F5FE'])
                                ax.axis('equal')  # 保持饼图为圆形
                                ax.set_title(f"{selected_file[:-14]} 基因在不同生物体中的分布", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig)
                        elif 'sequence_length' in df.columns:
                            st.subheader("基因长度分布")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.hist(df['sequence_length'], bins=20, alpha=0.7, color='#64B5F6', edgecolor='black')
                            ax.set_title(f"{selected_file[:-14]} 基因长度分布", fontsize=14)
                            ax.set_xlabel("序列长度", fontsize=12)
                            ax.set_ylabel("数量", fontsize=12)
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:  # 'organism' in df.columns
                            st.subheader("生物体分布")
                            # 取前10个最常见的生物体
                            top_organisms = df['organism'].value_counts().head(10)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.pie(top_organisms.values, labels=top_organisms.index, autopct='%1.1f%%', startangle=90, colors=['#64B5F6', '#81C784', '#FFB74D', '#E57373', '#9575CD', '#4DB6AC', '#4FC3F7', '#81D4FA', '#B3E5FC', '#E1F5FE'])
                            ax.axis('equal')  # 保持饼图为圆形
                            ax.set_title(f"{selected_file[:-14]} 基因在不同生物体中的分布", fontsize=14)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # 增强统计分析功能
                    st.subheader("高级统计分析")
                    
                    # 相关性分析
                    if len(df._get_numeric_data().columns) > 1:
                        st.subheader("相关性分析")
                        
                        # 计算相关系数矩阵
                        corr_matrix = df._get_numeric_data().corr()
                        
                        # 绘制相关性热图
                        st.subheader("相关性热图")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(corr_matrix, cmap='coolwarm')
                        
                        # 添加色标
                        cbar = ax.figure.colorbar(im, ax=ax)
                        cbar.ax.set_ylabel("相关性系数", rotation=-90, va="bottom")
                        
                        # 设置坐标轴
                        numeric_cols = df._get_numeric_data().columns
                        ax.set_xticks(range(len(numeric_cols)))
                        ax.set_yticks(range(len(numeric_cols)))
                        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
                        ax.set_yticklabels(numeric_cols, fontsize=10)
                        
                        # 添加数值标签
                        for i in range(len(numeric_cols)):
                            for j in range(len(numeric_cols)):
                                text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                                            ha="center", va="center", color="w", fontsize=8)
                        
                        ax.set_title(f"{selected_file[:-14]} 基因元数据相关性热图", fontsize=14)
                        fig.tight_layout()
                        st.pyplot(fig)
                    
                    # 缺失值分析
                    st.subheader("缺失值分析")
                    missing_values = df.isnull().sum()
                    if missing_values.any():
                        # 绘制缺失值柱状图
                        st.subheader("缺失值分布")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        missing_values.plot(kind='bar', ax=ax, color='#E57373')
                        ax.set_title(f"{selected_file[:-14]} 基因元数据缺失值分布", fontsize=12)
                        ax.set_xlabel("列名", fontsize=10)
                        ax.set_ylabel("缺失值数量", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.success("没有缺失值")
                    
                    # 添加元数据下载功能
                    st.subheader("数据下载")
                    
                    # 转换为CSV格式
                    metadata_csv = df.to_csv(index=False, encoding='utf-8-sig')
                    
                    # 提供下载按钮
                    st.download_button(
                        label=f"下载{selected_file[:-14]} 基因元数据 (CSV)",
                        data=metadata_csv,
                        file_name=f"{selected_file[:-14]}_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("未找到基因元数据文件")
        except Exception as e:
            st.error(f"读取元数据文件时出错: {e}")
    
    elif analysis_type == "AI组合预测":
        st.subheader("AI赋能的模块化元件组合预测")
        st.caption("基于启动子、RBS、CDS、终止子组合进行机器学习表达预测，并联动ODE仿真验证。")

        elements_dir = os.path.join(os.path.dirname(__file__), "../data/regulatory_elements")
        try:
            library = ModularElementLibrary(elements_dir)

            # 训练配置
            col_train_1, col_train_2 = st.columns([1, 1])
            with col_train_1:
                n_samples = st.slider("训练样本数", min_value=120, max_value=500, value=260, step=20)
            with col_train_2:
                random_seed = st.number_input("随机种子", min_value=1, max_value=9999, value=42, step=1)

            if st.button("训练AI预测模型", use_container_width=True):
                with st.spinner("正在训练模型..."):
                    model, metrics = train_predictor_from_library(
                        library,
                        n_samples=int(n_samples),
                        random_state=int(random_seed),
                    )
                    st.session_state["ai_predictor_model"] = model
                    st.session_state["ai_predictor_metrics"] = metrics
                    st.success("模型训练完成")

            if "ai_predictor_metrics" in st.session_state:
                metrics = st.session_state["ai_predictor_metrics"]
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("训练集R²", f"{metrics['train_r2']:.4f}")
                with mcol2:
                    st.metric("测试集R²", f"{metrics['test_r2']:.4f}")
                with mcol3:
                    st.metric("样本数", int(metrics["samples"]))

            st.markdown("### 元件组合设计")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                selected_promoter = st.selectbox("启动子", options=library.element_options("promoter"))
                selected_rbs = st.selectbox("RBS", options=library.element_options("rbs"))
            with dcol2:
                selected_cds = st.selectbox("编码序列 (CDS)", options=library.element_options("cds"))
                selected_terminator = st.selectbox("终止子", options=library.element_options("terminator"))

            design = {
                "promoter": selected_promoter,
                "rbs": selected_rbs,
                "cds": selected_cds,
                "terminator": selected_terminator,
            }
            target_gene = st.selectbox("映射到模型的目标基因", options=circuit.genes, index=0)

            if st.button("预测并仿真当前组合", type="primary", use_container_width=True):
                with st.spinner("正在执行AI预测与ODE仿真..."):
                    if "ai_predictor_model" not in st.session_state:
                        model, metrics = train_predictor_from_library(library, n_samples=260, random_state=42)
                        st.session_state["ai_predictor_model"] = model
                        st.session_state["ai_predictor_metrics"] = metrics

                    model = st.session_state["ai_predictor_model"]
                    pred_expr = predict_design_expression(model, library, design)
                    sim_result = simulate_design_in_circuit(
                        circuit=circuit,
                        design=design,
                        library=library,
                        target_gene=target_gene,
                        t_end=float(t_end),
                        t_points=int(t_points),
                    )

                    st.session_state["ai_pred_expr"] = pred_expr
                    st.session_state["ai_sim_result"] = sim_result

            if "ai_pred_expr" in st.session_state and "ai_sim_result" in st.session_state:
                pred_expr = st.session_state["ai_pred_expr"]
                sim_result = st.session_state["ai_sim_result"]

                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    st.metric("AI预测表达强度", f"{pred_expr:.4f}")
                with ecol2:
                    st.metric("ODE仿真最终蛋白浓度", f"{sim_result['final_protein']:.4f}")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(sim_result["t"], sim_result["protein"], color="#64B5F6", linewidth=2.2, label="蛋白浓度")
                ax.plot(sim_result["t"], sim_result["mRNA"], color="#81C784", linewidth=1.8, linestyle="--", label="mRNA浓度")
                ax.set_title(f"{target_gene} 在当前元件组合下的仿真轨迹", fontsize=13)
                ax.set_xlabel("时间")
                ax.set_ylabel("浓度")
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)

                design_df = pd.DataFrame([design])
                st.dataframe(design_df, use_container_width=True)
        except Exception as e:
            st.error(f"AI组合预测模块运行出错: {e}")

    else:  # 序列数据分析
        st.subheader("序列数据分析")
        
        # 序列数据文件夹路径
        sequence_dir = os.path.join(os.path.dirname(__file__), "../data/gvp_sequences")
        
        # 获取所有序列文件，支持更多格式
        try:
            sequence_files = [f for f in os.listdir(sequence_dir) if f.endswith(('.fasta', '.fa', '.fna', '.faa'))]
            
            if sequence_files:
                # 选择要查看的序列文件
                selected_file = st.selectbox(
                    "选择基因序列文件",
                    sequence_files
                )
                
                if selected_file:
                    # 读取序列，使用缓存避免重复读取
                    file_path = os.path.join(sequence_dir, selected_file)
                    
                    # 检查缓存
                    sequences = get_cached_data('sequence', file_path)
                    
                    if sequences is None:
                        # 缓存中没有，读取文件
                        from Bio import SeqIO
                        sequences = list(SeqIO.parse(file_path, "fasta"))
                        # 添加到缓存
                        add_cached_data('sequence', file_path, sequences)
                    
                    # 显示序列基本信息
                    st.subheader(f"{selected_file[:-6]} 基因序列数据")
                    
                    # 显示序列数量
                    st.metric("序列数量", len(sequences))
                    
                    # 选择要查看的序列
                    selected_seq_idx = st.selectbox(
                        "选择要查看的序列",
                        options=range(len(sequences)),
                        format_func=lambda x: sequences[x].id
                    )
                    
                    # 显示选中的序列
                    selected_seq = sequences[selected_seq_idx]
                    st.subheader(f"序列: {selected_seq.id}")
                    st.write(f"描述: {selected_seq.description}")
                    st.write(f"长度: {len(selected_seq.seq)}")
                    # 使用代码块显示序列，支持复制
                    st.code(selected_seq.seq, language="text")
                    
                    # 显示序列统计信息
                    st.subheader("序列统计信息")
                    
                    from collections import Counter
                    seq_count = Counter(str(selected_seq.seq))
                    seq_stats = pd.DataFrame.from_dict(seq_count, orient='index', columns=['数量'])
                    seq_stats['百分比'] = seq_stats['数量'] / len(selected_seq.seq) * 100
                    
                    # 使用数据框显示，支持排序
                    st.dataframe(seq_stats, use_container_width=True)
                    
                    # 绘制碱基组成饼图
                    fig, ax = plt.subplots(figsize=(8, 6))
                    # 使用统一的颜色方案
                    colors = ['#64B5F6', '#81C784', '#FFB74D', '#E57373']
                    ax.pie(seq_count.values(), labels=seq_count.keys(), autopct='%1.1f%%', startangle=90, colors=colors)
                    ax.axis('equal')  # 保持饼图为圆形
                    ax.set_title(f"{selected_file[:-6]} 基因序列碱基组成", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # 添加序列下载功能
                    st.subheader("下载基因序列")
                    
                    # 转换为FASTA格式
                    seq_fasta = f">{selected_seq.description}\n{selected_seq.seq}\n"
                    
                    # 提供下载按钮
                    st.download_button(
                        label=f"下载序列 {selected_seq.id} (FASTA)",
                        data=seq_fasta,
                        file_name=f"{selected_seq.id}.fasta",
                        mime="text/fasta",
                        use_container_width=True
                    )
            else:
                st.info("未找到基因序列文件")
        except Exception as e:
            st.error(f"读取序列文件时出错: {e}")

# 选项卡3：路径分析
with main_tabs[3]:
    st.header("3. 路径分析")
    
    # 路径分析设置
    st.subheader("路径分析设置")
    
    # 参数选择区域 - 3列布局
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # 选择源基因
        source_gene = st.selectbox(
            "源基因",
            options=circuit.genes,
            index=circuit.genes.index("gvpE")  # 默认选择gvpE，作为主要调控基因
        )
    
    with col2:
        # 选择目标基因
        target_gene = st.selectbox(
            "目标基因",
            options=circuit.genes,
            index=circuit.genes.index("gvpA")  # 默认选择gvpA，作为主要结构基因
        )
    
    with col3:
        # 设置最大路径长度
        max_path_length = st.slider(
            "最大路径长度",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        )
    
    # 自动运行路径分析
    with st.spinner("正在查找调控路径..."):
            # 查找调控路径
            paths = circuit.find_regulatory_paths(source_gene, target_gene, max_length=max_path_length)
            
            if paths:
                st.success(f"找到 {len(paths)} 条从 {source_gene} 到 {target_gene} 的调控路径")
                
                # 计算路径强度
                path_strengths = []
                for path in paths:
                    strength = circuit.calculate_path_strength(path)
                    path_strengths.append({
                        "路径": " → ".join(path),
                        "长度": len(path) - 1,
                        "强度": strength
                    })
                
                # 创建路径强度DataFrame
                path_df = pd.DataFrame(path_strengths)
                
                # 按强度排序
                path_df = path_df.sort_values(by="强度", ascending=False)
                
                # 显示路径列表
                st.subheader("调控路径列表")
                st.dataframe(path_df, use_container_width=True)
                
                # 绘制路径强度条形图
                st.subheader("路径强度比较")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 使用较短的路径标签
                short_labels = [f"P{i+1}: {path[:50]}..." if len(path) > 50 else f"P{i+1}: {path}" 
                               for i, path in enumerate(path_df["路径"])]
                
                # 使用统一的颜色方案
                colors = ['#64B5F6' if strength > 0 else '#E57373' for strength in path_df["强度"]]
                
                ax.barh(short_labels, path_df["强度"], color=colors)
                ax.set_title(f"从 {source_gene} 到 {target_gene} 的调控路径强度", fontsize=14)
                ax.set_xlabel("路径强度", fontsize=10)
                ax.set_ylabel("路径", fontsize=10)
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
                
                # 可视化最强路径
                st.subheader("最强调控路径可视化")
                
                # 获取最强路径
                strongest_path = paths[np.argmax([p["强度"] for p in path_strengths])]
                
                # 创建网络
                network = circuit.get_gene_network()
                G = nx.DiGraph()
                
                # 添加所有节点和边
                for node in network['nodes']:
                    G.add_node(node['id'])
                
                for edge in network['edges']:
                    if edge['weight'] != 0:
                        G.add_edge(edge['source'], edge['target'], weight=edge['weight'], type=edge['type'])
                
                # 创建子图，包含最强路径和其直接邻居
                subgraph_nodes = set(strongest_path)
                for node in strongest_path:
                    # 添加所有邻居节点
                    subgraph_nodes.update(list(G.neighbors(node)))
                    subgraph_nodes.update(list(G.predecessors(node)))
                
                # 创建子图
                subgraph = G.subgraph(subgraph_nodes)
                
                # 布局
                pos = nx.spring_layout(subgraph, seed=42, k=0.5)
                
                # 绘制图形
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制所有节点和边（灰色，透明）
                nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='lightgray', alpha=0.5, ax=ax)
                nx.draw_networkx_edges(subgraph, pos, edge_color='lightgray', alpha=0.3, ax=ax)
                
                # 高亮显示最强路径
                path_edges = [(strongest_path[i], strongest_path[i+1]) for i in range(len(strongest_path) - 1)]
                
                # 绘制路径节点
                nx.draw_networkx_nodes(subgraph, pos, nodelist=strongest_path, node_size=800, node_color='#64B5F6', ax=ax)
                
                # 绘制路径边
                nx.draw_networkx_edges(subgraph, pos, edgelist=path_edges, edge_color='#FFB74D', width=3, ax=ax)
                
                # 绘制所有节点标签
                nx.draw_networkx_labels(subgraph, pos, font_size=9, font_weight='bold', ax=ax)
                
                # 绘制边标签（仅路径边）
                path_edge_labels = {(u, v): f"{subgraph[u][v]['weight']:.2f}" for u, v in path_edges}
                nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=path_edge_labels, font_size=8, font_weight='bold', ax=ax)
                
                ax.set_title(f"从 {source_gene} 到 {target_gene} 的最强调控路径", fontsize=14)
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info(f"没有找到从 {source_gene} 到 {target_gene} 的调控路径")

# 选项卡4：模型验证
with main_tabs[4]:
    st.header("4. 模型验证")
    
    # 实验数据上传
    st.subheader("实验数据上传")
    uploaded_file = st.file_uploader(
        "上传实验数据文件（CSV格式，包含time列和基因列）",
        type=["csv"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            # 使用文件的唯一标识符作为缓存键
            file_key = get_uploaded_file_key('experimental', uploaded_file)
            
            # 检查缓存
            experimental_data = get_cached_data('experimental', file_key)
            
            if experimental_data is None:
                # 缓存中没有，读取文件
                experimental_data = pd.read_csv(uploaded_file)
                # 添加到缓存
                add_cached_data('experimental', file_key, experimental_data)
            
            st.success(f"成功加载实验数据: {uploaded_file.name}")
            
            # 显示实验数据预览
            st.subheader("实验数据预览")
            st.dataframe(experimental_data.head(10), use_container_width=True)
            
            # 选择要比较的基因
            available_genes = [col for col in experimental_data.columns if col != 'time']
            comparison_genes = st.multiselect(
                "选择要比较的基因",
                options=available_genes,
                default=available_genes[:3] if len(available_genes) > 0 else []
            )
            
            if comparison_genes:
                # 运行模型仿真
                t_exp = experimental_data['time'].values
                results = circuit.simulate(t_span=(t_exp.min(), t_exp.max()), t_eval=t_exp)
                
                # 模型与实验数据比较
                st.subheader("模型预测 vs 实验数据")
                
                # 绘制比较图表
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 使用统一的颜色方案
                line_colors = ['#64B5F6', '#81C784', '#FFB74D', '#E57373', '#9575CD', '#4DB6AC']
                
                for i, gene in enumerate(comparison_genes):
                    if gene in results['protein'] and gene in experimental_data.columns:
                        # 模型预测数据
                        ax.plot(
                            results['t'],
                            results['protein'][gene],
                            label=f"{gene} (模型预测)",
                            linewidth=2.0,
                            linestyle='-',
                            marker='',
                            color=line_colors[i % len(line_colors)]
                        )
                        # 实验数据
                        ax.scatter(
                            experimental_data['time'],
                            experimental_data[gene],
                            label=f"{gene} (实验数据)",
                            s=80,
                            alpha=0.7,
                            color=line_colors[i % len(line_colors)]
                        )
                
                # 设置图形样式
                ax.set_title("模型预测与实验数据比较", fontsize=16)
                ax.set_xlabel("时间", fontsize=12)
                ax.set_ylabel("蛋白质浓度", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10, loc='best')
                plt.tight_layout()
                
                # 显示图形
                st.pyplot(fig)
                
                # 计算并显示拟合度
                st.subheader("模型拟合度")
                try:
                    fitness = circuit.calculate_model_fitness(experimental_data)
                    st.metric("均方误差 (MSE)", f"{fitness:.6f}")
                    # 计算R²值
                    total_variance = np.var(experimental_data[comparison_genes].values)
                    r_squared = 1 - fitness / total_variance if total_variance != 0 else 0
                    st.metric("拟合优度 (R²)", f"{r_squared:.6f}")
                except Exception as e:
                    st.error(f"计算拟合度时出错: {e}")
                
                # 参数校准
                st.subheader("参数校准")
                
                # 选择要校准的参数类型
                param_types = st.multiselect(
                    "选择要校准的参数类型",
                    options=['alpha', 'beta', 'gamma'],
                    default=['alpha', 'beta']
                )
                
                # 校准迭代次数
                iterations = st.slider(
                    "校准迭代次数",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10
                )
                
                if st.button("运行参数校准", type="primary"):
                    with st.spinner("正在运行参数校准..."):
                        try:
                            # 保存当前参数用于比较
                            original_params = {
                                'alpha': circuit.params['alpha'].copy(),
                                'beta': circuit.params['beta'].copy(),
                                'gamma': circuit.params['gamma'].copy()
                            }
                            
                            # 运行参数校准
                            best_params, best_fitness = circuit.calibrate_parameters(
                                experimental_data=experimental_data,
                                param_types=param_types,
                                iterations=iterations
                            )
                            
                            # 显示校准结果
                            st.success("参数校准完成!")
                            st.subheader("校准结果")
                            st.metric("校准前均方误差 (MSE)", f"{fitness:.6f}")
                            st.metric("校准后均方误差 (MSE)", f"{best_fitness:.6f}")
                            
                            # 重新运行仿真，使用校准后的参数
                            calibrated_results = circuit.simulate(t_span=(t_exp.min(), t_exp.max()), t_eval=t_exp)
                            
                            # 绘制校准前后的比较图表
                            st.subheader("校准前后比较")
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            for i, gene in enumerate(comparison_genes):
                                if gene in calibrated_results['protein'] and gene in experimental_data.columns:
                                    # 校准前数据
                                    ax.plot(
                                        results['t'],
                                        results['protein'][gene],
                                        label=f"{gene} (校准前)",
                                        linewidth=2.0,
                                        linestyle='--',
                                        color='gray',
                                        alpha=0.7
                                    )
                                    # 校准后数据
                                    ax.plot(
                                        calibrated_results['t'],
                                        calibrated_results['protein'][gene],
                                        label=f"{gene} (校准后)",
                                        linewidth=2.5,
                                        linestyle='-',
                                        color=line_colors[i % len(line_colors)]
                                    )
                                    # 实验数据
                                    ax.scatter(
                                        experimental_data['time'],
                                        experimental_data[gene],
                                        label=f"{gene} (实验数据)",
                                        s=80,
                                        alpha=0.7,
                                        color=line_colors[i % len(line_colors)]
                                    )
                            
                            # 设置图形样式
                            ax.set_title("参数校准前后模型预测与实验数据比较", fontsize=16)
                            ax.set_xlabel("时间", fontsize=12)
                            ax.set_ylabel("蛋白质浓度", fontsize=12)
                            ax.grid(True, alpha=0.3)
                            ax.legend(fontsize=10, loc='best')
                            plt.tight_layout()
                            
                            # 显示图形
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"参数校准时出错: {e}")
        except Exception as e:
            st.error(f"读取实验数据时出错: {e}")
    else:
        st.info("请上传实验数据文件以进行模型验证")

# 选项卡5：数据管理（合并了原数据下载和用户上传数据）
with main_tabs[5]:
    st.header("5. 数据管理")
    
    # 数据管理选项
    data_option = st.radio(
        "选择数据管理功能",
        options=["仿真结果下载", "网络结构下载", "参数下载", "用户数据上传"],
        index=0,
        horizontal=True
    )
    
    if data_option == "仿真结果下载":
        # 仿真结果下载
        st.subheader("仿真结果下载")
        
        # 准备完整的仿真结果数据
        full_results = pd.DataFrame(results['t'], columns=['时间'])
        for gene in circuit.genes:
            full_results[f"{gene}_mRNA"] = results['mRNA'][gene]
            full_results[f"{gene}_protein"] = results['protein'][gene]
        
        # 转换为CSV格式
        csv = full_results.to_csv(index=False, encoding='utf-8-sig')
        
        # 提供下载按钮
        st.download_button(
            label="下载完整仿真结果 (CSV)",
            data=csv,
            file_name=f"gene_expression_simulation_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # 显示数据预览
        st.subheader("数据预览")
        st.dataframe(full_results.head(10), use_container_width=True)
    
    elif data_option == "网络结构下载":
        # 网络结构下载
        st.subheader("网络结构下载")
        
        # 获取网络结构
        network = circuit.get_gene_network()
        
        # 转换为JSON格式
        network_json = json.dumps(network, ensure_ascii=False, indent=2)
        
        # 提供下载按钮
        st.download_button(
            label="下载网络结构 (JSON)",
            data=network_json,
            file_name=f"gene_network_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # 显示网络结构预览
        st.subheader("网络结构预览")
        st.json(network)
    
    elif data_option == "参数下载":
        # 参数下载
        st.subheader("模型参数下载")
        
        # 准备参数数据
        params_data = {
            "alpha": circuit.params['alpha'],
            "beta": circuit.params['beta'],
            "gamma": circuit.params['gamma'],
            "regulation": circuit.params['regulation'].tolist()
        }
        
        # 转换为JSON格式
        params_json = json.dumps(params_data, ensure_ascii=False, indent=2)
        
        # 提供下载按钮
        st.download_button(
            label="下载模型参数 (JSON)",
            data=params_json,
            file_name=f"model_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # 显示参数预览
        st.subheader("参数预览")
        # 显示主要参数
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("转录速率 (alpha)")
            alpha_df = pd.DataFrame(list(circuit.params['alpha'].items()), columns=["基因", "转录速率"])
            st.dataframe(alpha_df, use_container_width=True)
        with col2:
            st.subheader("翻译速率 (beta)")
            beta_df = pd.DataFrame(list(circuit.params['beta'].items()), columns=["基因", "翻译速率"])
            st.dataframe(beta_df, use_container_width=True)
        with col3:
            st.subheader("降解速率 (gamma)")
            gamma_df = pd.DataFrame(list(circuit.params['gamma'].items()), columns=["基因", "降解速率"])
            st.dataframe(gamma_df, use_container_width=True)
    
    else:  # 用户数据上传
        st.subheader("用户数据上传与分析")
        
        # 文件上传，支持更多格式
        uploaded_file = st.file_uploader(
            "上传基因数据文件",
            type=["csv", "fasta", "txt", "fa", "fna", "faa", "xlsx", "xls", "json"],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                # 生成文件的唯一标识符作为缓存键
                file_key = get_uploaded_file_key('uploaded', uploaded_file)
                
                # 检查缓存
                data = get_cached_data('uploaded', file_key)
                
                if data is None:
                    # 根据文件类型读取数据
                    if file_extension in ['.csv']:
                        # 读取CSV文件
                        data = pd.read_csv(uploaded_file)
                        data_type = 'csv'
                    elif file_extension in ['.xlsx', '.xls']:
                        # 读取Excel文件
                        data = pd.read_excel(uploaded_file)
                        data_type = 'excel'
                    elif file_extension in ['.json']:
                        # 读取JSON文件
                        import json
                        data = json.load(uploaded_file)
                        data_type = 'json'
                    elif file_extension in ['.fasta', '.fa', '.fna', '.faa', '.txt']:
                        # 读取FASTA或文本文件
                        from Bio import SeqIO
                        data = list(SeqIO.parse(uploaded_file, "fasta"))
                        data_type = 'sequence'
                    else:
                        st.error(f"不支持的文件格式: {file_extension}")
                        data_type = 'unknown'
                    
                    # 添加到缓存
                    add_cached_data('uploaded', file_key, {'data': data, 'type': data_type})
                else:
                    # 从缓存中获取数据和类型
                    data = data['data']
                    data_type = data['type']
                
                # 处理不同类型的数据
                if data_type in ['csv', 'excel']:
                    # 显示数据基本信息
                    st.subheader(f"上传的数据: {uploaded_file.name}")
                    
                    # 显示数据概览
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("数据行数", len(data))
                    with col2:
                        st.metric("数据列数", len(data.columns))
                    with col3:
                        # 计算非空值比例
                        non_null_ratio = data.notnull().sum().sum() / (len(data) * len(data.columns)) * 100
                        st.metric("数据完整性", f"{non_null_ratio:.2f}%")
                    
                    # 显示数据前几行
                    st.subheader("数据预览")
                    st.dataframe(data.head(10), use_container_width=True)
                    
                    # 显示数据统计信息
                    st.subheader("数据统计信息")
                    st.dataframe(data.describe(), use_container_width=True)
                    
                elif data_type == 'sequence':
                    # 显示序列基本信息
                    st.subheader(f"上传的序列数据: {uploaded_file.name}")
                    
                    # 显示序列数量
                    st.metric("序列数量", len(data))
                    
                    # 选择要查看的序列
                    selected_seq_idx = st.selectbox(
                        "选择要查看的序列",
                        options=range(len(data)),
                        format_func=lambda x: data[x].id
                    )
                    
                    # 显示选中的序列
                    selected_seq = data[selected_seq_idx]
                    st.subheader(f"序列: {selected_seq.id}")
                    st.write(f"描述: {selected_seq.description}")
                    st.write(f"长度: {len(selected_seq.seq)}")
                    # 使用代码块显示序列，支持复制
                    st.code(selected_seq.seq, language="text")
                    
                elif data_type == 'json':
                    # 显示JSON数据
                    st.subheader(f"上传的JSON数据: {uploaded_file.name}")
                    
                    # 尝试将JSON数据转换为DataFrame以便更好地显示
                    try:
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                            st.dataframe(df.head(10), use_container_width=True)
                        elif isinstance(data, dict):
                            # 对于字典类型的JSON，显示为表格
                            df = pd.DataFrame.from_dict(data, orient='index')
                            st.dataframe(df.head(10), use_container_width=True)
                        else:
                            st.json(data)
                    except Exception as e:
                        st.json(data)
                
                st.success(f"成功加载文件: {uploaded_file.name}")
            except Exception as e:
                st.error(f"读取上传文件时出错: {e}")
                st.exception(e)

# 选项卡6：报告生成
with main_tabs[6]:
    st.header("6. 报告生成")
    
    # 报告生成设置
    st.subheader("报告生成设置")
    
    # 报告标题
    report_title = st.text_input("报告标题", value="气囊基因线路分析报告")
    
    # 报告作者
    report_author = st.text_input("报告作者", value="系统自动生成")
    
    # 报告日期
    report_date = st.date_input("报告日期", value=datetime.now())
    
    # 选择要包含的内容
    st.subheader("选择报告内容")
    include_circuit = st.checkbox("包含线路结构图", value=True)
    include_dynamics = st.checkbox("包含表达动态", value=True)
    include_sensitivity = st.checkbox("包含参数分析", value=True)
    include_pathway = st.checkbox("包含路径分析", value=True)
    include_validation = st.checkbox("包含模型验证", value=True)
    
    # 报告格式选择
    report_format = st.selectbox("报告格式", options=["Markdown", "HTML"], index=0)
    
    # 生成报告按钮
    if st.button("生成分析报告", type="primary"):
        with st.spinner("正在生成报告..."):
            # 生成报告内容
            report_content = f"""# {report_title}

**作者**: {report_author}
**日期**: {report_date.strftime('%Y-%m-%d')}

## 1. 报告概述

本报告提供了气囊基因线路的详细分析，包括线路结构、表达动态、参数分析、路径分析和模型验证等内容。

## 2. 模型参数

### 2.1 基本设置
- 基因数量: {len(circuit.genes)}
- 仿真时间范围: {t_start} 到 {t_end} (共 {t_points} 个时间点)
- 选择的基因: {', '.join(selected_genes)}
- 分子类型: {molecule_type}

### 2.2 主要参数

#### 转录速率 (alpha)
| 基因 | 转录速率 |
|------|----------|
"""
            
            # 添加转录速率表格
            for gene in selected_genes:
                report_content += f"| {gene} | {circuit.params['alpha'][gene]:.4f} |\n"
            
            report_content += f"""

#### 翻译速率 (beta)
| 基因 | 翻译速率 |
|------|----------|
"""
            
            # 添加翻译速率表格
            for gene in selected_genes:
                report_content += f"| {gene} | {circuit.params['beta'][gene]:.4f} |\n"
            
            report_content += f"""

#### 降解速率 (gamma)
| 基因 | 降解速率 |
|------|----------|
"""
            
            # 添加降解速率表格
            for gene in selected_genes:
                report_content += f"| {gene} | {circuit.params['gamma'][gene]:.4f} |\n"
            
            report_content += f"""

## 3. 线路结构分析

### 3.1 网络统计
- 节点数量: {len(network['nodes'])}
- 边数量: {len(network['edges'])}
- 网络密度: {nx.density(G):.4f}

### 3.2 调控关系

| 调控基因 | 被调控基因 | 调控强度 | 调控类型 |
|----------|------------|----------|----------|
"""
            
            # 添加调控关系表格
            network = circuit.get_gene_network()
            for edge in network['edges']:
                report_content += f"| {edge['source']} | {edge['target']} | {edge['weight']:.4f} | {edge['type']} |\n"
            
            report_content += f"""

## 4. 表达动态分析

### 4.1 最终浓度

| 基因 | {molecule_type} 浓度 |
|------|-------------------|
"""
            
            # 添加最终浓度表格
            for gene in selected_genes:
                if gene in results[molecule_type]:
                    final_conc = results[molecule_type][gene][-1]
                    report_content += f"| {gene} | {final_conc:.6f} |\n"
            
            report_content += f"""

## 5. 路径分析

### 5.1 关键调控路径

以下是从 gvpE 到 gvpA 的主要调控路径：

"""
            
            # 添加路径分析结果
            paths = circuit.find_regulatory_paths("gvpE", "gvpA", max_length=3)
            if paths:
                for i, path in enumerate(paths[:5]):  # 显示前5条路径
                    strength = circuit.calculate_path_strength(path)
                    report_content += f"- 路径 {i+1}: {' → '.join(path)} (强度: {strength:.4f})\n"
            else:
                report_content += "- 未找到从 gvpE 到 gvpA 的调控路径\n"
            
            report_content += f"""

## 6. 模型性能评估

### 6.1 仿真设置
- 仿真时间: {t_start} 到 {t_end}
- 时间点数量: {t_points}

### 6.2 计算资源使用
- 仿真时间: 完成
- 内存使用: 正常

## 7. 结论与建议

### 7.1 主要发现
- 基因线路表现稳定，各基因表达水平符合预期
- 关键调控路径已识别，为进一步研究提供了方向
- 模型参数敏感性分析有助于理解系统行为

### 7.2 建议
- 进一步优化关键基因的表达参数
- 增加实验数据验证模型预测
- 考虑更多环境因素对基因表达的影响
"""
            
            # 根据选择的格式生成报告
            if report_format == "HTML":
                # 转换为HTML格式
                # 先处理report_content
                processed_content = report_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>').replace('#### ', '<h4>').replace('|', '<td>').replace('- ', '<li>').replace('\n\n', '</li>\n<li>').replace('\n', '<br>')
                # 然后构建HTML
                report_content_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .section {{ margin-bottom: 30px; }}
        .highlight {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #3498db; }}
    </style>
</head>
<body>
    {processed_content}
</body>
</html>
"""
                # 提供下载按钮
                st.download_button(
                    label="下载分析报告 (HTML)",
                    data=report_content_html,
                    file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html",
                    use_container_width=True
                )
            else:
                # 提供下载按钮
                st.download_button(
                    label="下载分析报告 (Markdown)",
                    data=report_content,
                    file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            st.success("报告生成完成！")
            # 显示报告预览
            st.subheader("报告预览")
            st.markdown(report_content)
            
