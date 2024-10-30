import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import os

# Load Granger results and ANOVA contrasts from files
granger_results_file = r'E:\ConnectivityMatrix\consolidated_output.xlsx'  # Update with your Granger results file path
anova_results_file = r'E:\ConnectivityMatrix\RM_ANOVA_results_with_interactions.xlsx'  # Update with your ANOVA results file path

# Define electrode positions in 3D using standard EEG 10-20 system coordinates
electrode_positions_3d = {
    'Fp1': (-2, 8, 5), 'Fp2': (2, 8, 5),
    'F7': (-6, 6, 5), 'F3': (-2, 6, 6), 'Fz': (0, 6, 7), 'F4': (2, 6, 6), 'F8': (6, 6, 5),
    'T7': (-8, 0, 5), 'C3': (-2, 0, 7), 'Cz': (0, 0, 8), 'C4': (2, 0, 7), 'T8': (8, 0, 5),
    'P7': (-6, -6, 5), 'P3': (-2, -6, 6), 'Pz': (0, -6, 7), 'P4': (2, -6, 6), 'P8': (6, -6, 5),
    'O1': (-2, -8, 5), 'Oz': (0, -8, 5), 'O2': (2, -8, 5),
    'CPz': (0, -3, 7), 'POz': (0, -7, 6),
    # Add any missing electrodes with appropriate 3D positions
}

# Load ANOVA results
anova_results = pd.ExcelFile(anova_results_file)

# Extract Granger connectivity data from participant-level data
def extract_granger_data(granger_data):
    # Exclude 'ID', 'Group', and 'Condition' columns to get electrode pairs
    electrode_pairs = [col for col in granger_data.columns if col not in ['ID', 'Group', 'Condition']]
    
    # Melt the DataFrame to long format
    granger_long = granger_data.melt(id_vars=['ID', 'Group', 'Condition'], value_vars=electrode_pairs,
                                     var_name='electrode_pair', value_name='connectivity_strength')
    
    # Split 'electrode_pair' into 'electrode1' and 'electrode2'
    granger_long[['electrode1', 'electrode2']] = granger_long['electrode_pair'].str.split('_', expand=True)
    
    # Drop the 'electrode_pair' column
    granger_long = granger_long.drop(columns=['electrode_pair'])
    
    # Now, compute the average connectivity_strength for each electrode pair
    granger_avg = granger_long.groupby(['electrode1', 'electrode2']).agg({'connectivity_strength': 'mean'}).reset_index()
    
    return granger_avg, granger_long

# Extract significant effects from ANOVA contrasts
def extract_anova_effects(anova_results, granger_long):
    significant_connections = []
    
    # Loop through relevant sheets and find significant connections
    for sheet_name in anova_results.sheet_names:
        if sheet_name.endswith('_Mixed_ANOVA'):
            df = pd.read_excel(anova_results, sheet_name)
            df['p-unc'] = pd.to_numeric(df['p-unc'], errors='coerce')
            df = df.dropna(subset=['p-unc'])
            
            # Keep only significant connections with p < 0.001
            significant_df = df[df['p-unc'] < 0.001]
            electrodes = sheet_name.replace('_Mixed_ANOVA', '').split('_')
            if len(electrodes) >= 2:
                electrode1 = electrodes[0]
                electrode2 = electrodes[1]
            else:
                continue
            
            # For this electrode pair, extract connectivity data
            conn_data = granger_long[
                ((granger_long['electrode1'] == electrode1) & (granger_long['electrode2'] == electrode2))
            ]
            
            for _, row in significant_df.iterrows():
                effect_type = row['Source']
                p_value = row['p-unc']
                
                if effect_type == 'Group':
                    # Calculate mean connectivity for each group for this connection
                    group_means = conn_data.groupby('Group')['connectivity_strength'].mean()
                    groups = group_means.index.tolist()
                    if len(groups) == 2:
                        group1, group2 = groups
                        mean1, mean2 = group_means[group1], group_means[group2]
                        if mean1 > mean2:
                            effect_detail = f"Group ({group1} > {group2})"
                        else:
                            effect_detail = f"Group ({group2} > {group1})"
                    else:
                        effect_detail = "Group Effect"
                elif effect_type == 'Condition':
                    # Calculate mean connectivity for each condition for this connection
                    condition_means = conn_data.groupby('Condition')['connectivity_strength'].mean()
                    conditions = condition_means.index.tolist()
                    if len(conditions) == 2:
                        cond1, cond2 = conditions
                        mean1, mean2 = condition_means[cond1], condition_means[cond2]
                        if mean1 > mean2:
                            effect_detail = f"Condition ({cond1} > {cond2})"
                        else:
                            effect_detail = f"Condition ({cond2} > {cond1})"
                    else:
                        effect_detail = "Condition Effect"
                elif effect_type == 'Interaction':
                    # For interaction effects, find the group-condition combination with the highest mean
                    interaction_means = conn_data.groupby(['Group', 'Condition'])['connectivity_strength'].mean()
                    if interaction_means.empty:
                        continue  # Skip if no data
                    (group_high, cond_high), mean_high = interaction_means.idxmax(), interaction_means.max()
                    effect_detail = f"Interaction ({group_high}, {cond_high} > others)"
                else:
                    effect_detail = effect_type  # Use the effect type as is
                
                effect = {
                    'electrode1': electrode1,
                    'electrode2': electrode2,
                    'p_value': p_value,
                    'effect_type': effect_detail,
                }
                significant_connections.append(effect)
    
    significant_effects_df = pd.DataFrame(significant_connections)
    return significant_effects_df

# Combine Granger data and ANOVA results
def combine_data(granger_data, anova_effects_df):
    # Merge granger_data and anova_effects_df on 'electrode1', 'electrode2'
    combined_df = pd.merge(granger_data, anova_effects_df, on=['electrode1', 'electrode2'], how='inner')
    
    # **Update Labels: Replace 'hispanic' with 'Latin Americans' and 'non-hispanic' with 'non-Hispanic Whites'**
    combined_df['effect_type'] = combined_df['effect_type'].str.replace('(?i)\bhispanic\b', 'Latin Americans', regex=True)
    combined_df['effect_type'] = combined_df['effect_type'].str.replace('(?i)\bnon-hispanic\b', 'non-Hispanic Whites', regex=True)
    
    return combined_df

# Filter the combined data to retain only the top N strongest connections per effect_type
def filter_top_connections(combined_df, top_n=5, by='absolute'):
    if by == 'absolute':
        combined_df['abs_connectivity'] = combined_df['connectivity_strength'].abs()
        sorted_df = combined_df.sort_values(by=['effect_type', 'abs_connectivity'], ascending=[True, False])
    elif by == 'p_value':
        sorted_df = combined_df.sort_values(by=['effect_type', 'p_value'], ascending=[True, True])
    else:
        raise ValueError("Parameter 'by' must be either 'absolute' or 'p_value'.")
    
    # Group by effect_type and take the top_n
    filtered_df = sorted_df.groupby('effect_type').head(top_n).reset_index(drop=True)
    
    # Drop the auxiliary column if it exists
    if 'abs_connectivity' in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=['abs_connectivity'])
    
    return filtered_df

# Perform network analysis
def perform_network_analysis(filtered_df):
    G = nx.MultiDiGraph()  # Use MultiDiGraph to allow multiple edges between the same nodes
    for _, row in filtered_df.iterrows():
        e1 = row['electrode1']
        e2 = row['electrode2']
        weight = row['connectivity_strength']
        p_value = row['p_value']
        effect_type = row['effect_type']
        # Add edge with attributes
        G.add_edge(e1, e2, weight=weight, p_value=p_value, effect_type=effect_type)
    return G

# Plot network in 3D and save as an HTML file
def plot_network_3d(G, title, filename, effect_type_filter=None):
    # Extract node positions
    pos = electrode_positions_3d
    node_x, node_y, node_z, node_labels = [], [], [], []
    for node in G.nodes():
        x, y, z = pos.get(node, (0, 0, 0))
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_labels.append(node)
    
    # Filter edges by effect_type if specified
    edges = [(u, v, key) for u, v, key in G.edges(keys=True) if effect_type_filter is None or effect_type_filter == G[u][v][key]['effect_type']]
    
    edge_traces = []
    for u, v, key in edges:
        x0, y0, z0 = pos.get(u, (0, 0, 0))
        x1, y1, z1 = pos.get(v, (0, 0, 0))
        edge_trace = go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='text'
        )
        edge_traces.append(edge_trace)

    # Node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers',
        marker=dict(size=5, color='red'),
        text=node_labels, hoverinfo='text'
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(title=title)
    
    # Save figure as HTML
    fig.write_html(filename)

# Function to create a master HTML file to open all figures
def create_master_html(figures_dir, output_file):
    figure_files = [f for f in os.listdir(figures_dir) if f.endswith('.html')]
    with open(output_file, 'w') as f:
        f.write('<html><body>\n')
        for fig in figure_files:
            f.write(f'<iframe src="{fig}" width="800" height="600"></iframe>\n')
        f.write('</body></html>')

# Main function
if __name__ == '__main__':
    # Step 1: Load Granger data
    granger_data_raw = pd.read_excel(granger_results_file)
    
    # Step 2: Extract data
    granger_connectivity, granger_long = extract_granger_data(granger_data_raw)
    
    anova_effects_df = extract_anova_effects(anova_results, granger_long)
    
    # Step 3: Combine data for final analysis
    combined_results_df = combine_data(granger_connectivity, anova_effects_df)
    
    # Step 4: Filter to retain only the top N strongest connections per effect_type
    top_n = 5
    filtered_results_df = filter_top_connections(combined_results_df, top_n=top_n)
    
    # Step 5: Create network graph
    G = perform_network_analysis(filtered_results_df)
    
    # Step 6: Plot networks for each effect type in 3D and save as individual HTML files
    figures_dir = 'figures'  # Directory to save the individual figures
    os.makedirs(figures_dir, exist_ok=True)
    
    effect_types = ['Group', 'Condition', 'Interaction']
    for effect_type in effect_types:
        plot_filename = os.path.join(figures_dir, f'network_{effect_type.lower()}.html')
        plot_network_3d(G, f"Network - {effect_type}", plot_filename)
    
    # Step 7: Create a master HTML file that includes all individual figures
    create_master_html(figures_dir, 'all_figures.html')
    print("Master HTML file created: all_figures.html")
