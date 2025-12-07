"""
Excel-Driven Dependency Network Analysis

Elegant, Excel-based approach for dependency network analysis.
Separates data from logic for maximum usability and business user accessibility.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from typing import Dict, List, Any


class DependencyNetwork:
    """N-level dependency network with method chaining support."""
    
    def __init__(self, base_scores: List[float], visible: List[List[float]], 
                 adjacency: List[List[List[int]]]):
        """
        Initialize network from configuration.
        
        Args:
            base_scores: Base level manual scores
            visible: Visible scores for each level [level][node]
            adjacency: Adjacency matrices for each level transition
        """
        self.level0_manual = np.array(base_scores)
        self.visible = visible
        self.adj = [np.array(a) for a in adjacency]
        
        self.num_levels = len(adjacency) + 1  # Total levels
        self.level_sizes = [len(self.level0_manual)] + [len(vis) for vis in self.visible]
        
        # Store visible scores as arrays
        self.level_visible = [np.array(vis) for vis in self.visible]
        
        # Initialize computed values
        self.level_fundamental = [None] * (self.num_levels - 1)  # No fundamental for level 0
        
    def compute(self):
        """Compute fundamental scores by propagating minimums upward."""
        # Compute fundamental scores for each level (except level 0)
        for level in range(1, self.num_levels):
            level_size = self.level_sizes[level]
            self.level_fundamental[level-1] = np.zeros(level_size)
            
            # Get adjacency matrix for this level
            adj_matrix = self.adj[level-1]
            
            for i in range(level_size):
                # Find connected nodes from previous level
                connected_nodes = np.where(adj_matrix[i, :] == 1)[0]
                
                if len(connected_nodes) == 0:
                    # No connections - set to 0 (or could be an error)
                    self.level_fundamental[level-1][i] = 0.0
                    continue
                
                if level == 1:
                    # Level 1 depends on Level 0 (manual scores)
                    connected_scores = self.level0_manual[connected_nodes]
                else:
                    # Higher levels depend on previous level's fundamental scores
                    connected_scores = self.level_fundamental[level-2][connected_nodes]
                
                self.level_fundamental[level-1][i] = np.min(connected_scores)
        
        return self
    
    def summary(self):
        """Print a summary table of scores."""
        print("\n" + "="*80)
        print("DEPENDENCY NETWORK ANALYSIS SUMMARY")
        print("="*80)
        
        # Level 0 (Base - Manual Scores Only)
        print(f"\nLEVEL 0 (Base - Manual Scores Only):")
        for i in range(self.level_sizes[0]):
            print(f"  L0-{i}: Manual = {self.level0_manual[i]:.2f}")
        
        # All other levels (with fundamental and visible scores)
        for level in range(1, self.num_levels):
            level_size = self.level_sizes[level]
            print(f"\nLEVEL {level}:")
            print(f"  {'Node':<8} {'Fundamental':<15} {'Visible':<15} {'Status'}")
            print(f"  {'-'*8} {'-'*15} {'-'*15} {'-'*20}")
            
            for i in range(level_size):
                fund = self.level_fundamental[level-1][i]
                vis = self.level_visible[level-1][i]
                status = self._get_status(fund, vis)
                print(f"  L{level}-{i:<6} {fund:<14.2f} {vis:<14.2f} {status}")
        
        print("\n" + "="*80)
        print("KEY: Fundamental = min of connected lower nodes (weakest link)")
        print("     Visible = manually-set apparent score")
        print("     WARNING = Visible score exceeds what dependencies support")
        print("="*80)
        
        return self
    
    def visualize(self, figsize=(14, 10)):
        """Create a network visualization."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        G = nx.DiGraph()
        
        # Create graph structure
        positions = self._create_positions()
        self._add_nodes_and_edges(G, positions)
        
        # Draw edges
        nx.draw_networkx_edges(G, positions, ax=ax, edge_color='#666666', 
                              arrows=True, arrowsize=20, width=2, alpha=0.7)
        
        # Draw all nodes
        all_nodes = list(G.nodes())
        nx.draw_networkx_nodes(G, positions, nodelist=all_nodes, ax=ax,
                              node_color='#f0f0f0', node_size=2800, node_shape='o',
                              edgecolors='#cccccc', linewidths=1)
        
        # Highlight inflated nodes
        inflated_nodes = self._get_inflated_nodes()
        if inflated_nodes:
            nx.draw_networkx_nodes(G, positions, nodelist=inflated_nodes, ax=ax,
                                  node_color='#f0f0f0', node_size=2800, node_shape='o',
                                  edgecolors='red', linewidths=3)
        
        # Add labels
        labels = self._create_labels()
        nx.draw_networkx_labels(G, positions, labels, ax=ax, font_size=9, 
                               font_weight='bold', font_color='#333333')
        
        # Add level annotations
        self._add_level_annotations(ax)
        
        # Add legend
        self._add_legend(ax)
        
        # Styling
        ax.set_title('Configuration-Driven Dependency Network\nFundamental vs Visible Scores', 
                    fontsize=14, fontweight='bold', pad=20, color='#333333')
        ax.text(0.02, 0.98, 
                'F = Fundamental (min of lower fundamentals ONLY)\n'
                'V = Visible (manual score)\nRed border = V > F (inflated)', 
                transform=ax.transAxes, fontsize=10, va='top', color='#555555',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                         edgecolor='lightgray'))
        
        # Dynamic limits based on number of levels
        max_level = self.num_levels - 1
        ax.set_xlim(-3, 12)
        ax.set_ylim(-1, max_level * 3 + 1)
        ax.axis('off')
        plt.tight_layout()
        
        return self
    
    @staticmethod
    def _get_status(fundamental, visible):
        """Get status string for a node."""
        if visible > fundamental:
            return f"WARNING: INFLATED by {(visible-fundamental):.2f}"
        elif visible == fundamental:
            return "OK: Aligned"
        else:
            return f"OK: Conservative by {(fundamental-visible):.2f}"
    
    def _create_positions(self):
        """Create node positions for visualization."""
        positions = {}
        
        # Level 0 (base)
        for i in range(self.level_sizes[0]):
            positions[f'L0-{i}'] = (i * 3, 0)
        
        # All other levels
        for level in range(1, self.num_levels):
            level_size = self.level_sizes[level]
            y_pos = level * 3  # Vertical spacing
            for i in range(level_size):
                x_pos = i * (4 + level * 0.5) + level * 0.5  # Horizontal spacing
                positions[f'L{level}-{i}'] = (x_pos, y_pos)
        
        return positions
    
    def _add_nodes_and_edges(self, G, positions):
        """Add nodes and edges to the graph."""
        # Add nodes
        for node in positions:
            G.add_node(node)
        
        # Add edges between consecutive levels
        for level in range(1, self.num_levels):
            level_size = self.level_sizes[level]
            prev_level_size = self.level_sizes[level-1]
            adj_matrix = self.adj[level-1]
            
            for i in range(level_size):
                for j in range(prev_level_size):
                    if adj_matrix[i, j] == 1:
                        G.add_edge(f'L{level-1}-{j}', f'L{level}-{i}')
    
    def _get_inflated_nodes(self):
        """Get list of inflated nodes (V > F)."""
        inflated = []
        for level in range(1, self.num_levels):
            level_size = self.level_sizes[level]
            for i in range(level_size):
                if self.level_visible[level-1][i] > self.level_fundamental[level-1][i]:
                    inflated.append(f'L{level}-{i}')
        return inflated
    
    def _create_labels(self):
        """Create node labels."""
        labels = {}
        
        # Level 0 (base - manual scores only)
        for i in range(self.level_sizes[0]):
            labels[f'L0-{i}'] = f'L0-{i}\n{self.level0_manual[i]:.2f}'
        
        # All other levels (fundamental + visible scores)
        for level in range(1, self.num_levels):
            level_size = self.level_sizes[level]
            for i in range(level_size):
                fund = self.level_fundamental[level-1][i]
                vis = self.level_visible[level-1][i]
                labels[f'L{level}-{i}'] = (f'L{level}-{i}\nF:{fund:.2f}\nV:{vis:.2f}')
        
        return labels
    
    def _add_level_annotations(self, ax):
        """Add level annotations to the plot."""
        for level in range(self.num_levels):
            y_pos = level * 3
            if level == 0:
                label = f'LEVEL {level}\n(Base)'
            else:
                # Dynamic levels - no hardcoded limit
                label = f'LEVEL {level}'
            
            ax.text(-1.5, y_pos, label, ha='center', va='center', 
                    fontsize=11, fontweight='bold', color='#555555')
    
    @staticmethod
    def _add_legend(ax):
        """Add legend to the plot."""
        legend = [
            mpatches.Circle((0,0), 0.1, facecolor='#f0f0f0', edgecolor='#cccccc', 
                           label='Normal Node'),
            mpatches.Circle((0,0), 0.1, facecolor='#f0f0f0', edgecolor='red', 
                           linewidth=2, label='Inflated (V > F)'),
            mpatches.FancyArrowPatch((0,0), (0.1,0), arrowstyle='->', 
                                    color='#666666', label='Dependencies')
        ]
        ax.legend(handles=legend, loc='upper right', fontsize=10, title='Legend')


def load_default_config() -> Dict[str, Any]:
    """Load default network configuration for fallback."""
    return {
        'base_scores': [0.80, 0.60, 0.90, 0.70],
        'visible': [
            [0.55, 0.55, 0.95],  # Level 1
            [0.50, 0.50]         # Level 2
        ],
        'adjacency': [
            [[1, 1, 1, 0],       # L0→L1
             [0, 1, 1, 1],
             [1, 0, 1, 0]],
            
            [[1, 0, 1],          # L1→L2
             [0, 1, 1]]
        ]
    }


def analyze_excel_structure(path: str) -> Dict[str, Any]:
    """Analyze Excel file and build flexible data model."""
    print(f"Analyzing Excel structure: {path}")
    
    excel_file = pd.ExcelFile(path)
    sheet_info = {}
    
    for sheet_name in excel_file.sheet_names:
        print(f"  Analyzing sheet: {sheet_name}")
        df = pd.read_excel(path, sheet_name=sheet_name)
        
        # Analyze column types and patterns
        column_analysis = {
            'description_columns': [col for col in df.columns if 'description' in col.lower() or 'desc' in col.lower()],
            'score_columns': [col for col in df.columns if 'score' in col.lower()],
            'dependency_columns': [col for col in df.columns if col.startswith('L') and '_' in col],
            'id_columns': [col for col in df.columns if 'id' in col.lower()],
            'empty_columns': [col for col in df.columns if col.strip() == ''],
            'all_columns': list(df.columns),
            'data_types': {col: str(df[col].dtype) for col in df.columns}
        }
        
        # Detect level number from sheet name
        level_num = None
        if sheet_name.startswith('Level_'):
            try:
                level_num = int(sheet_name.split('_')[1])
            except:
                level_num = None
        
        sheet_info[sheet_name] = {
            'data': df,
            'columns': column_analysis,
            'num_rows': len(df),
            'level_number': level_num,
            'is_base_level': level_num == 0 if level_num is not None else False
        }
        
        print(f"    Columns: {column_analysis['all_columns']}")
        print(f"    Description columns: {column_analysis['description_columns']}")
        print(f"    Score columns: {column_analysis['score_columns']}")
        print(f"    Dependency columns: {column_analysis['dependency_columns']}")
        print(f"    Level number: {level_num}")
    
    return sheet_info


def load_from_excel(path: str = 'cdn_network.xlsx') -> DependencyNetwork:
    """Load network from Excel configuration file with flexible structure analysis."""
    print(f"Loading network configuration from: {path}")
    
    # Analyze Excel structure
    sheet_info = analyze_excel_structure(path)
    
    # Find level sheets and sort them
    level_sheets = [(name, info) for name, info in sheet_info.items() 
                   if info['level_number'] is not None]
    level_sheets.sort(key=lambda x: x[1]['level_number'])
    
    if not level_sheets:
        raise ValueError("No Level_* sheets found in Excel file")
    
    print(f"  Found {len(level_sheets)} levels: {[name for name, _ in level_sheets]}")
    
    # Process base level (Level_0)
    base_sheet_name, base_info = level_sheets[0]
    base_df = base_info['data']
    
    # Find score column (flexible naming)
    score_cols = base_info['columns']['score_columns']
    if not score_cols:
        raise ValueError(f"No score columns found in {base_sheet_name}")
    
    # Use first score column found
    score_col = score_cols[0]
    base_scores = base_df[score_col].tolist()
    print(f"  {base_sheet_name}: {len(base_scores)} base nodes loaded")
    print(f"    Using score column: '{score_col}'")
    
    # Process all other levels
    visible_scores = []
    adjacency_matrices = []
    
    for i, (sheet_name, sheet_info) in enumerate(level_sheets[1:], 1):
        df = sheet_info['data']
        
        # Find visible score column
        score_cols = sheet_info['columns']['score_columns']
        if not score_cols:
            raise ValueError(f"No score columns found in {sheet_name}")
        
        visible_col = score_cols[0]  # Use first score column
        visible_scores.append(df[visible_col].tolist())
        
        # Find dependency columns (previous level)
        prev_level = f"L{i-1}_"
        dep_cols = [col for col in df.columns if col.startswith(prev_level)]
        
        if not dep_cols:
            print(f"    Warning: No dependency columns found for {prev_level}* in {sheet_name}")
            # Create empty adjacency matrix
            adj_matrix = [[0] * len(base_scores) for _ in range(len(visible_scores[-1]))]
        else:
            adj_matrix = df[dep_cols].values.tolist()
        
        adjacency_matrices.append(adj_matrix)
        
        print(f"  {sheet_name}: {len(visible_scores[-1])} nodes loaded")
        print(f"    Using score column: '{visible_col}'")
        print(f"    Dependencies on {prev_level}*: {len(dep_cols)} columns")
    
    # Prepare configuration
    config = {
        'base_scores': base_scores,
        'visible': visible_scores,
        'adjacency': adjacency_matrices
    }
    
    return DependencyNetwork(**config)


def create_example_excel(path: str = 'cdn_network.xlsx', num_levels: int = 3):
    """Create an example Excel file with flexible structure that demonstrates header-aware reading."""
    print(f"Creating example Excel file: {path}")
    print(f"  Generating {num_levels} levels (Level_0 to Level_{num_levels-1})")
    print(f"  Demonstrating flexible column naming and structure")
    
    # Level 0 (Base) data - Description first, then score
    l0_data = {
        'Description': ['Node A', 'Node B', 'Node C', 'Node D'],  # First column
        'Base_Score': [0.80, 0.60, 0.90, 0.70]  # Score column
    }
    
    # Generate all other levels dynamically
    level_data = [l0_data]
    
    for level in range(1, num_levels):
        # Create visible scores (decreasing with level)
        visible_scores = [0.7 - level * 0.1, 0.6 - level * 0.1, 0.9 - level * 0.05]
        if level > 1:
            visible_scores = visible_scores[:2]  # Fewer nodes at higher levels
        
        # Create data structure with Description first
        level_dict = {
            'Description': [f'Level {level} Node {i}' for i in range(len(visible_scores))],  # First column
            'Apparent_Score': visible_scores,  # Score column
            '': ['', '', ''] if len(visible_scores) == 3 else ['', ''],  # Visual separator
        }
        
        # Add dependency columns (previous level)
        prev_level = f"L{level-1}_"
        prev_level_size = len(level_data[level-1]['Base_Score' if level == 1 else 'Apparent_Score'])
        
        for i in range(prev_level_size):
            # Create some random but reasonable dependencies
            dep_values = []
            for j in range(len(visible_scores)):
                if level == 1:
                    # Level 1 dependencies on Level 0
                    dep_values.append(1 if (i + j) % 2 == 0 else 0)
                else:
                    # Higher level dependencies - ensure at least one connection per node
                    if j == i % len(visible_scores):  # Ensure each node has at least one connection
                        dep_values.append(1)
                    else:
                        dep_values.append(1 if (i + j) % 3 == 0 else 0)
            level_dict[f'{prev_level}{i}'] = dep_values
        
        level_data.append(level_dict)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for level in range(num_levels):
            sheet_name = f'Level_{level}'
            level_df = pd.DataFrame(level_data[level])
            level_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            if level == 0:
                print(f"  - {sheet_name}: {len(level_df)} base nodes")
                print(f"    Columns: {list(level_df.columns)}")
            else:
                print(f"  - {sheet_name}: {len(level_df)} nodes")
                print(f"    Columns: {list(level_df.columns)}")
    
    print(f"  Example Excel file created successfully!")
    print(f"  - {num_levels} levels with flexible column naming")
    print(f"  - Demonstrates header-aware reading capabilities")
    print(f"  - Edit any Level_* sheet to modify the network")
    print(f"  - Program will auto-detect 'score' columns and 'L*_' dependencies")


def detect_levels_from_excel(file_path: str) -> int:
    """Detect number of levels from Excel file."""
    try:
        sheet_info = analyze_excel_structure(file_path)
        level_sheets = [name for name, info in sheet_info.items() if info['level_number'] is not None]
        return len(level_sheets)
    except:
        return 3  # Default fallback


def main():
    """Execute dependency network analysis with method chaining."""
    # Configuration - change this to use different Excel files
    EXCEL_FILE = 'cdn_network.xlsx'
    
    print("=== Excel-Driven Dependency Network Analysis ===")
    
    try:
        # Load from Excel file
        net = load_from_excel(EXCEL_FILE)
    except FileNotFoundError:
        print(f"Excel file not found, creating example file: {EXCEL_FILE}")
        # Auto-detect levels from existing Excel or use default
        num_levels = detect_levels_from_excel(EXCEL_FILE)
        print(f"  Detected {num_levels} levels, creating example file...")
        create_example_excel(EXCEL_FILE, num_levels)
        net = load_from_excel(EXCEL_FILE)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print("Falling back to default configuration...")
        config = load_default_config()
        net = DependencyNetwork(**config)
    
    # Method chaining - elegant and concise
    net.compute().summary().visualize()
    
    print('\nOK: Analysis complete!')
    print('  - Excel-driven approach')
    print('  - Method chaining for elegant code')
    print('  - Business-user friendly configuration')
    
    plt.show()


if __name__ == '__main__':
    main()
