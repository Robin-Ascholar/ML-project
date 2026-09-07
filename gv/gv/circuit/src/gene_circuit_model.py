#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
气囊基因线路模型

该模块实现了气囊基因（gvp系列）的线路模型，包括基因相互作用网络和仿真功能。
"""

import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime

class GasVesicleCircuit:
    """气囊基因线路模型类"""
    
    def __init__(self):
        """初始化基因线路模型"""
        # 基因列表和初始浓度 - 扩展到15个气囊基因
        self.genes = ["gvpA", "gvpB", "gvpC", "gvpD", "gvpE", "gvpF", "gvpG", 
                      "gvpJ", "gvpK", "gvpL", "gvpM", "gvpN", "gvpS", "gvpU", "gvpV"]
        
        # 模型参数
        self.params = {
            # 转录速率 (base rate)
            'alpha': {
                'gvpA': 0.5,  'gvpB': 0.45, 'gvpC': 0.4,  'gvpD': 0.3,
                'gvpE': 0.6,  'gvpF': 0.35, 'gvpG': 0.25, 'gvpJ': 0.2,
                'gvpK': 0.3,  'gvpL': 0.28, 'gvpM': 0.26, 'gvpN': 0.45,
                'gvpS': 0.22, 'gvpU': 0.2,  'gvpV': 0.24
            },
            # 翻译速率
            'beta': {
                'gvpA': 1.2,  'gvpB': 1.1,  'gvpC': 1.0,  'gvpD': 0.9,
                'gvpE': 1.3,  'gvpF': 0.85, 'gvpG': 0.75, 'gvpJ': 0.6,
                'gvpK': 0.8,  'gvpL': 0.7,  'gvpM': 0.65, 'gvpN': 0.95,
                'gvpS': 0.55, 'gvpU': 0.5,  'gvpV': 0.6
            },
            # 降解速率
            'gamma': {
                'gvpA': 0.15, 'gvpB': 0.14, 'gvpC': 0.12, 'gvpD': 0.18,
                'gvpE': 0.2,  'gvpF': 0.16, 'gvpG': 0.14, 'gvpJ': 0.13,
                'gvpK': 0.15, 'gvpL': 0.14, 'gvpM': 0.13, 'gvpN': 0.17,
                'gvpS': 0.12, 'gvpU': 0.11, 'gvpV': 0.13
            },
            # 基因调控矩阵 (regulatory interactions)
            # rows: regulated gene, columns: regulator gene
            # 正值表示激活，负值表示抑制，0表示无作用
            'regulation': np.array([
                [0, 0.15, 0.2, -0.1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],    # gvpA regulation
                [0.1, 0, 0.1, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      # gvpB regulation
                [0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        # gvpC regulation
                [0, 0, 0, 0, 0, 0.15, 0, 0, 0, 0, 0, 0.2, 0, 0, 0],       # gvpD regulation
                [0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # gvpE regulation
                [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],          # gvpF regulation
                [0, 0, 0, 0, 0.1, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0],        # gvpG regulation
                [0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # gvpJ regulation
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0],          # gvpK regulation
                [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0, 0],        # gvpL regulation
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0],          # gvpM regulation
                [0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # gvpN regulation
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0],          # gvpS regulation
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.1],        # gvpU regulation
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0]           # gvpV regulation
            ])
        }
        
        # 初始浓度 (mRNA 和蛋白质)
        self.initial_conditions = {
            'mRNA': np.array([0.1 for _ in self.genes]),
            'protein': np.array([0.05 for _ in self.genes])
        }
        
        # 实验数据存储（用于模型验证与校准）
        self.experimental_data = None
        
        # 结果缓存，用于避免重复计算
        self._simulation_cache = {}  # 使用字典作为缓存，键为缓存键，值为结果
        self._cache_access = {}      # 记录缓存访问时间，用于LRU策略
        # 缓存大小限制，可根据系统资源调整
        self._max_cache_size = 20     # 增加缓存大小，提高缓存命中率
    
    def _generate_cache_key(self, t_span, t_eval, initial_conditions):
        """生成缓存键
        
        Args:
            t_span: 仿真时间范围
            t_eval: 评估时间点列表
            initial_conditions: 初始条件
        
        Returns:
            cache_key: 唯一的缓存键
        """
        # 使用更高效的缓存键生成方法
        # 初始条件处理
        if initial_conditions is None:
            ic_key = hash(tuple(self.initial_conditions['mRNA'].tobytes())) ^ hash(tuple(self.initial_conditions['protein'].tobytes()))
        else:
            ic_key = hash(tuple(initial_conditions['mRNA'].tobytes())) ^ hash(tuple(initial_conditions['protein'].tobytes()))
        
        # 参数处理 - 使用numpy数组的哈希值，更高效
        param_key = hash(tuple(np.round(list(self.params['alpha'].values()), 4))) ^ \
                   hash(tuple(np.round(list(self.params['beta'].values()), 4))) ^ \
                   hash(tuple(np.round(list(self.params['gamma'].values()), 4))) ^ \
                   hash(self.params['regulation'].tobytes())
        
        # 时间参数处理
        time_key = hash(tuple(np.round(t_span, 2))) ^ (hash(tuple(np.round(t_eval, 2))) if t_eval is not None else 0)
        
        # 组合所有哈希值
        return (param_key, time_key, ic_key)
    
    def set_parameter(self, param_type, gene, value):
        """设置模型参数
        
        Args:
            param_type: 参数类型 ('alpha', 'beta', 'gamma')
            gene: 基因名称
            value: 参数值
        """
        if param_type in self.params and gene in self.params[param_type]:
            self.params[param_type][gene] = value
            # 参数变化，清除缓存
            self._simulation_cache.clear()
    
    def set_regulation(self, regulated_gene, regulator_gene, value):
        """设置基因间的调控关系
        
        Args:
            regulated_gene: 被调控基因
            regulator_gene: 调控基因
            value: 调控强度 (正值激活，负值抑制)
        """
        reg_idx = self.genes.index(regulated_gene)
        gene_idx = self.genes.index(regulator_gene)
        self.params['regulation'][reg_idx, gene_idx] = value
        # 参数变化，清除缓存
        self._simulation_cache.clear()
    
    def _ode_system(self, t, y):
        """ODE系统：描述基因表达动力学
        
        Args:
            t: 时间
            y: 状态向量 [mRNA1, mRNA2, ..., mRNA8, protein1, protein2, ..., protein8]
        
        Returns:
            dy/dt: 状态导数向量
        """
        n_genes = len(self.genes)
        mRNA = y[:n_genes]
        protein = y[n_genes:]
        
        # 向量化实现：计算mRNA变化率
        # 将参数转换为数组形式
        alpha_arr = np.array([self.params['alpha'][gene] for gene in self.genes])
        beta_arr = np.array([self.params['beta'][gene] for gene in self.genes])
        gamma_arr = np.array([self.params['gamma'][gene] for gene in self.genes])
        
        # 基础转录速率
        base_transcription = alpha_arr
        
        # 调控作用（向量化矩阵乘法）
        regulation = np.dot(self.params['regulation'], protein)
        
        # mRNA降解
        degradation_mRNA = gamma_arr * mRNA
        
        # mRNA变化率
        dmRNA_dt = base_transcription + regulation - degradation_mRNA
        
        # 向量化实现：计算蛋白质变化率
        # 翻译
        translation = beta_arr * mRNA
        
        # 蛋白质降解
        degradation_protein = gamma_arr * protein
        
        # 蛋白质变化率
        dprotein_dt = translation - degradation_protein
        
        return np.concatenate((dmRNA_dt, dprotein_dt))
    
    def simulate(self, t_span=(0, 100), t_eval=None, initial_conditions=None):
        """仿真基因线路动态
        
        Args:
            t_span: 仿真时间范围 (start, end)
            t_eval: 评估时间点列表
            initial_conditions: 初始条件字典，默认为模型默认值
            
        Returns:
            t: 时间点数组
            results: 结果字典，包含各基因的mRNA和蛋白质浓度
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(t_span, t_eval, initial_conditions)
        
        # 检查缓存中是否已存在结果
        if cache_key in self._simulation_cache:
            # 更新缓存访问时间
            self._cache_access[cache_key] = datetime.now().timestamp()
            return self._simulation_cache[cache_key]
        
        # 使用默认初始条件或用户提供的初始条件
        if initial_conditions is None:
            y0 = np.concatenate((self.initial_conditions['mRNA'], self.initial_conditions['protein']))
        else:
            y0 = np.concatenate((initial_conditions['mRNA'], initial_conditions['protein']))
        
        # 求解ODE - 使用更高效的求解器，根据系统特性选择最适合的方法
        # 对于基因表达系统，通常是非刚性的，使用LSODA或DOP853求解器更高效
        sol = solve_ivp(
            self._ode_system,
            t_span,
            y0,
            t_eval=t_eval,
            method='LSODA',  # LSODA自动切换刚性/非刚性方法，更高效
            rtol=1e-6,        # 调整相对公差，平衡精度和速度
            atol=1e-9         # 调整绝对公差
        )
        
        # 整理结果 - 使用更高效的向量化操作
        n_genes = len(self.genes)
        t = sol.t
        mRNA_data = sol.y[:n_genes, :]
        protein_data = sol.y[n_genes:, :]
        
        # 使用numpy的结构化数组或更高效的字典生成方式
        results = {
            't': t,
            'mRNA': {gene: mRNA_data[i] for i, gene in enumerate(self.genes)},
            'protein': {gene: protein_data[i] for i, gene in enumerate(self.genes)}
        }
        
        # 将结果存入缓存
        if len(self._simulation_cache) >= self._max_cache_size:
            # 使用LRU策略移除最旧的缓存项
            # 找到最近最少使用的缓存键
            oldest_key = min(self._cache_access, key=self._cache_access.get)
            # 移除最旧的缓存项
            del self._simulation_cache[oldest_key]
            del self._cache_access[oldest_key]
        
        # 存储结果到缓存
        self._simulation_cache[cache_key] = results
        # 记录缓存访问时间
        self._cache_access[cache_key] = datetime.now().timestamp()
        
        return results
    
    def get_gene_network(self):
        """获取基因调控网络结构
        
        Returns:
            network: 网络结构字典，包含节点和边
        """
        nodes = [{'id': gene, 'label': gene} for gene in self.genes]
        
        edges = []
        for i, regulated in enumerate(self.genes):
            for j, regulator in enumerate(self.genes):
                weight = self.params['regulation'][i, j]
                if weight != 0:
                    edges.append({
                        'source': regulator,
                        'target': regulated,
                        'weight': weight,
                        'type': 'activation' if weight > 0 else 'inhibition'
                    })
        
        return {'nodes': nodes, 'edges': edges}
    
    def load_experimental_data(self, file_path):
        """加载实验数据用于模型验证
        
        Args:
            file_path: 实验数据文件路径（CSV格式）
        """
        import pandas as pd
        self.experimental_data = pd.read_csv(file_path)
    
    def calculate_model_fitness(self, experimental_data=None):
        """计算模型与实验数据的拟合度
        
        Args:
            experimental_data: 可选，实验数据DataFrame
        
        Returns:
            fitness: 模型拟合度（均方误差）
        """
        if experimental_data is None:
            experimental_data = self.experimental_data
        
        if experimental_data is None:
            raise ValueError("No experimental data available. Please load data first.")
        
        # 运行仿真
        t_exp = experimental_data['time'].values
        results = self.simulate(t_span=(t_exp.min(), t_exp.max()), t_eval=t_exp)
        
        # 计算均方误差
        mse = 0.0
        count = 0
        
        for gene in self.genes:
            if gene in experimental_data.columns:
                # 获取实验数据
                exp_data = experimental_data[gene].values
                # 获取仿真数据
                sim_data = results['protein'][gene]
                # 计算均方误差
                mse += np.mean((sim_data - exp_data) ** 2)
                count += 1
        
        return mse / count if count > 0 else float('inf')
    
    def calibrate_parameters(self, experimental_data=None, param_types=['alpha', 'beta', 'gamma'], iterations=100):
        """校准模型参数
        
        Args:
            experimental_data: 实验数据DataFrame
            param_types: 要校准的参数类型列表
            iterations: 优化迭代次数
        
        Returns:
            best_params: 校准后的最佳参数
            best_fitness: 最佳拟合度
        """
        from scipy.optimize import differential_evolution
        
        if experimental_data is None:
            experimental_data = self.experimental_data
        
        if experimental_data is None:
            raise ValueError("No experimental data available. Please load data first.")
        
        # 创建参数上下限
        param_bounds = []
        param_names = []
        
        for param_type in param_types:
            for gene in self.genes:
                current_value = self.params[param_type][gene]
                # 参数边界：当前值的0.5到2倍
                param_bounds.append((current_value * 0.5, current_value * 2.0))
                param_names.append(f"{param_type}_{gene}")
        
        # 目标函数：最小化拟合误差
        def objective(params):
            # 更新参数
            for i, param_name in enumerate(param_names):
                param_type, gene = param_name.split('_')
                self.params[param_type][gene] = params[i]
            
            # 计算拟合度
            return self.calculate_model_fitness(experimental_data)
        
        # 运行差分进化算法
        result = differential_evolution(
            objective, 
            param_bounds,
            maxiter=iterations,
            popsize=15,
            tol=1e-6
        )
        
        # 更新最佳参数
        for i, param_name in enumerate(param_names):
            param_type, gene = param_name.split('_')
            self.params[param_type][gene] = result.x[i]
        
        return self.params, result.fun
    
    def find_regulatory_paths(self, source, target, max_length=3):
        """查找从源基因到目标基因的调控路径
        
        Args:
            source: 源基因
            target: 目标基因
            max_length: 最大路径长度
        
        Returns:
            paths: 路径列表，每个路径是基因名称的列表
        """
        import networkx as nx
        
        # 获取基因网络
        network = self.get_gene_network()
        
        # 创建NetworkX图
        G = nx.DiGraph()
        
        # 添加节点
        for node in network['nodes']:
            G.add_node(node['id'])
        
        # 添加边
        for edge in network['edges']:
            if edge['weight'] != 0:
                G.add_edge(edge['source'], edge['target'], weight=edge['weight'], type=edge['type'])
        
        # 查找所有简单路径
        paths = []
        try:
            for length in range(1, max_length + 1):
                simple_paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=length))
                paths.extend(simple_paths)
        except nx.NetworkXNoPath:
            pass
        
        return paths
    
    def calculate_path_strength(self, path):
        """计算调控路径的强度
        
        Args:
            path: 路径列表，包含基因名称
        
        Returns:
            strength: 路径强度，基于调控权重的乘积
        """
        if len(path) < 2:
            return 0.0
        
        strength = 1.0
        
        for i in range(len(path) - 1):
            regulator = path[i]
            regulated = path[i+1]
            
            # 获取调控强度
            reg_idx = self.genes.index(regulated)
            gene_idx = self.genes.index(regulator)
            weight = self.params['regulation'][reg_idx, gene_idx]
            
            strength *= weight
        
        return strength

if __name__ == "__main__":
    # 测试模型
    circuit = GasVesicleCircuit()
    results = circuit.simulate(t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
    
    # 打印结果
    print("仿真结果示例:")
    print(f"时间点数量: {len(results['t'])}")
    print("各基因最终蛋白质浓度:")
    for gene in circuit.genes:
        print(f"{gene}: {results['protein'][gene][-1]:.4f}")
    
    # 打印网络结构
    network = circuit.get_gene_network()
    print(f"\n基因调控网络:")
    print(f"节点数量: {len(network['nodes'])}")
    print(f"边数量: {len(network['edges'])}")
