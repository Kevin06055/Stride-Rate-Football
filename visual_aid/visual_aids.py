import matplotlib.pyplot as plt
import pandas as pd
def plot_stride_rate_fluctuations(all_players_data, frame_num,active_ids):
    plt.figure(figsize=(8, 6))

    # Color palette for different players (e.g., a set of distinct colors)
    colors = plt.cm.get_cmap('tab10', len(all_players_data))  # 'tab10' provides 10 distinct colors

    # Loop through each player's data and plot their stride rate fluctuation over time
    for idx, (tracker_id, data) in enumerate(all_players_data.items()):
        if tracker_id not in active_ids:
            continue
             
        df = pd.DataFrame(data)  # Convert data into DataFrame for plotting
        
        # Plot the stride rate fluctuations as a scatter plot (x-axis: frame_num, y-axis: stride_rate)
        plt.scatter(df['frame'], df['stride_rate'], label=f'Player {tracker_id}', s=10, color=colors(idx))

    plt.title(f"Stride Rate Fluctuations Over Time at Frame {frame_num}")
    plt.xlabel('Frame Number')
    plt.ylabel('Stride Rate')
    plt.grid(True)

    max_legend_entries = 20
    handles,labels = plt.gca().get_legend_handles_labels()

    if len(labels) > max_legend_entries:
        handles = handles[:max_legend_entries]
        labels = labels[:max_legend_entries]
        labels[-1] = f"+{len(labels)-max_legend_entries}More..."
         
    # Save the plot as a separate image file
    plot_image_path = f"frames/stride_rate_fluctuations_frame_{frame_num}.png"

    
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=8, title="Players")
    plt.tight_layout()
    plt.savefig(plot_image_path)
    plt.close()