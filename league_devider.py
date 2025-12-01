import random
import numpy as np
from init import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from collections import defaultdict

def flatten_availability_matrix(matrix):
    return [
        matrix[day][slot]
        for day in days_of_week
        for slot in time_slots
    ]

def assign_knn_teams_to_groups(league, group_sizes, plot):

    team_names = list(league.keys())
    team_data = [flatten_availability_matrix(league[team]) for team in team_names]

    num_clusters = len(group_sizes)
    if plot:
        print(f"\nConstrained KMeans: {len(team_names)} teams into exact sizes {group_sizes}")

    # Built in kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans.fit(team_data)
    centroids = kmeans.cluster_centers_

    #All team distances to centroids
    team_distances = [
        (name, vec, [((i, np.linalg.norm(vec - centroids[i]))) for i in range(num_clusters)])
        for name, vec in zip(team_names, team_data)
    ]

    # Sort teams by their closeness to all centroids
    assignment_pool = []
    for name, vec, distances_to_all in team_distances:
        for cluster_id, dist in distances_to_all:
            assignment_pool.append((dist, name, vec, cluster_id))

    assignment_pool.sort()

    # Greedy assignment to clusters with size limits
    total_distance = 0
    assigned = defaultdict(list)
    team_assigned = set()
    cluster_limits = {i: size for i, size in enumerate(group_sizes)}

    for dist, name, vec, cluster_id in assignment_pool:
        if name in team_assigned:
            continue
        if len(assigned[cluster_id]) < cluster_limits[cluster_id]:
            assigned[cluster_id].append((name, league[name]))
            team_assigned.add(name)
            total_distance += dist

        # Early exit if all assigned
        if len(team_assigned) == len(team_names):
            break

    if plot:
        print("Assigned cluster sizes:")
        for cluster_id in sorted(assigned):
            print(f"Group {cluster_id+1}: {len(assigned[cluster_id])} teams")

    # Calculate individual distances
    team_to_distance = {}
    for dist, name, _, cluster_id in assignment_pool:
        if name in team_assigned and name not in team_to_distance:
            team_to_distance[name] = dist
    if plot:
        print("\nIndividual Distances to Assigned Centroid (sorted):")
        sorted_teams = sorted(team_to_distance.items(), key=lambda x: x[1])
        for name, dist in sorted_teams:
            print(f"  {name:25} -> {dist:.4f}")

        # Average distance for good measure i think
        average_distance = sum(team_to_distance.values()) / len(team_to_distance)
        print(f"\nAvg. Distance to Assigned Centroid: {average_distance:.4f} (lower = more compact groups)")


    subgroups = [dict(assigned[cid]) for cid in sorted(assigned)]
    return subgroups

def assign_pairwise_optimized_groups(league, group_sizes, plot):

    if plot: print(f"\nPairwise Optimization: {len(league)} teams into groups {group_sizes}")

    team_names = list(league.keys())
    team_vectors = np.array([flatten_availability_matrix(league[name]) for name in team_names])

    # Compute pairwise distances
    dist_matrix = squareform(pdist(team_vectors, metric='euclidean'))

    # Greedy grouping
    unassigned = set(range(len(league)))
    groups = []

    for size in group_sizes:
        if len(unassigned) < size:
            print("Not enough teams left to form another group!")
            break

        # Start group with one team randomly # TODO: implement better selection
        current_group = [unassigned.pop()]
        while len(current_group) < size:
            # Find the team not yet in group with smallest average distance to current group
            candidates = list(unassigned)
            avg_dists = [
                np.mean([dist_matrix[i][j] for i in current_group])
                for j in candidates
            ]
            next_team = candidates[np.argmin(avg_dists)]
            current_group.append(next_team)
            unassigned.remove(next_team)

        groups.append(current_group)

    # Build final output
    grouped_teams = []
    for group in groups:
        group_dict = {team_names[i]: league[team_names[i]] for i in group}
        grouped_teams.append(group_dict)

    if plot: # mondjuk semmitmondóak ezek a számok de legalább vannak
        print("Intra-group average distances:")
        for i, group in enumerate(groups):
            if len(group) < 2:
                print(f"Group {i+1}: N/A")
                continue
            pairwise = [dist_matrix[a][b] for a in group for b in group if a < b]
            avg_dist = sum(pairwise) / len(pairwise)
            print(f"Group {i+1}: {avg_dist:.4f}")

    return grouped_teams

def assign_random_teams_to_groups(league_items, group_sizes):
    
    random.shuffle(league_items)

    divided = []
    start = 0
    for size in group_sizes:
        group = dict(league_items[start:start + size])
        divided.append(group)
        start += size

    return divided

def plot_all_leagues_clusters(divided_leagues, division_counts, method="tsne"):
    #import umap.umap_ as umap
    from sklearn.decomposition import PCA
    
    reducer_cls = {
        "tsne": lambda X: TSNE(n_components=2, random_state=42, perplexity=10).fit_transform(X),
        "umap": lambda X: umap.UMAP(n_components=2, random_state=42, n_neighbors=min(10, len(X)-1)).fit_transform(X),
        "pca": lambda X: PCA(n_components=2).fit_transform(X)
    }

    if method not in reducer_cls:
        raise ValueError("Method must be 'tsne', 'umap', or 'pca'")

    reducer = reducer_cls[method]

    # Split groups into their original leagues
    grouped_by_league = []
    idx = 0
    for count in division_counts:
        grouped_by_league.append(divided_leagues[idx:idx+count])
        idx += count

    n_leagues = len(grouped_by_league)
    fig, axes = plt.subplots(1, n_leagues, figsize=(6 * n_leagues, 6), squeeze=False)

    for league_idx, league_groups in enumerate(grouped_by_league):
        all_teams = []
        all_vectors = []
        labels = []

        for group_idx, group in enumerate(league_groups):
            for team_name, availability in group.items():
                all_teams.append(team_name)
                all_vectors.append(flatten_availability_matrix(availability))
                labels.append(group_idx)

        all_vectors = np.array(all_vectors)
        reduced = reducer(all_vectors)

        n_groups = len(league_groups)
        cmap_name = 'tab10' if n_groups <= 10 else 'tab20'
        cmap = cm.get_cmap(cmap_name, n_groups)
        colors = [cmap(label) for label in labels]

        ax = axes[0][league_idx]
        scatter = ax.scatter(
            reduced[:, 0], reduced[:, 1],
            color=colors,
            s=100,
            edgecolors='k',
            alpha=0.85
        )

        # Plot centroids for KMeans
        if devision_strategy == "knn":
            group_means = []
            for group_id in range(n_groups):
                group_points = reduced[np.array(labels) == group_id]
                centroid = group_points.mean(axis=0)
                group_means.append(centroid)

            for idx, center in enumerate(group_means):
                ax.scatter(*center, color=cmap(idx), marker='X', s=150, edgecolor='white')

        #ax.set_title(f"League {league_idx+1}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True)

        # Legend
        handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(i),
                   markeredgecolor='k', label=f'Division {i+1}', markersize=10)
            for i in range(n_groups)
        ]
        ax.legend(handles=handles, loc='best')#, title='Divisions')

    plt.tight_layout()
    plt.show()

def calculate_division_counts(league, max_group_size):
    n = len(league)
    if n <= max_group_size:
        return [n]

    num_groups = ((n + max_group_size - 1) // max_group_size)
    base_group_size = n // num_groups
    remainder = n % num_groups

    group_sizes = [base_group_size + 1] * remainder + [base_group_size] * (num_groups - remainder)
    return group_sizes

def devide_leagues(leagues, strategy="random", plot=True):
    divided = []
    division_counts = []

    for league in leagues:
        group_sizes = calculate_division_counts(league, 12)
        division_counts.append(len(group_sizes))

        league_items = list(league.items())
        subgroups = []
        if strategy == "random":
            subgroups = assign_random_teams_to_groups(league_items, group_sizes)
        if strategy == "knn":
            subgroups = assign_knn_teams_to_groups(league, group_sizes, plot)
        if strategy == "pairwise":
            subgroups = assign_pairwise_optimized_groups(league, group_sizes, plot)
        divided.extend(subgroups)

    if plot:
        #plot_all_leagues_clusters(divided, division_counts, method="tsne")
        #plot_all_leagues_clusters(divided, division_counts, method="umap")
        plot_all_leagues_clusters(divided, division_counts, method="pca")
    return divided, division_counts