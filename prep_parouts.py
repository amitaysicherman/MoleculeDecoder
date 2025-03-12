import json
import os

from matplotlib import pyplot as plt
import networkx as nx
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from networkx.drawing.nx_pydot import graphviz_layout
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from tqdm import tqdm
import random
from collections import defaultdict

matplotlib.use("TkAgg")
RDLogger.DisableLog('rdApp.*')


def preprocess_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def extract_smiles_tree(node, graph=None, parent=None):
    """Recursively extract SMILES from molecules while maintaining the tree structure in a graph."""
    if graph is None:
        graph = nx.DiGraph()

    if node["type"] == "mol":
        smiles = preprocess_smiles(node["smiles"])
        graph.add_node(smiles)
        if parent:
            graph.add_edge(parent, smiles)
        for child in node.get("children", []):
            extract_smiles_tree(child, graph, smiles)
    elif node["type"] == "reaction":
        for child in node.get("children", []):
            extract_smiles_tree(child, graph, parent)

    return graph


def smiles_to_image(smiles):
    """Generate a molecule image from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        return img
    return None


def get_graph_root(graph):
    """Get the root node of the graph."""
    roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
    if len(roots) != 1:
        raise ValueError(f"Expected 1 root node, found {len(roots)}")
    return roots[0]


def draw_smiles_tree(graph):
    """Generate and plot the SMILES tree structure using a tree layout with molecule images."""
    plt.figure(figsize=(8, 6))

    # Compute tree layout
    # pos = nx.drawing.bfs_layout(graph, get_graph_root(graph))
    pos = graphviz_layout(graph, prog="dot")

    # Draw nodes and edges
    nx.draw(graph, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=2000)

    # Add molecule images to nodes
    ax = plt.gca()
    for node, (x, y) in pos.items():
        img = smiles_to_image(node)
        img = img.resize((100, 100))
        imagebox = OffsetImage(img, zoom=0.75)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    plt.show()


def tree_to_ancestors_children(graph):
    """Convert a graph structure to lists of nodes, ancestors, and children."""
    nodes = list(graph.nodes())
    ancestors = {}
    children = {}

    # Process each node
    for node in nodes:
        ancestors[node] = []
        current = node

        # Walk up the graph to find all ancestors
        for parent in graph.predecessors(current):
            ancestors[node].append(parent)
            current_ancestors = ancestors.get(parent, [])
            ancestors[node].extend(current_ancestors)

        # Find all direct children for this node
        direct_children = list(graph.successors(node))
        if direct_children:  # Only add if there are children
            children[node] = direct_children

    return nodes, ancestors, children


import os
import requests

url = "https://zenodo.org/record/7341155/files/all_loaded_routes.json.gz?download=1"
filename = "all_loaded_routes.json.gz"
if not os.path.exists(filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        pbar = tqdm(
            total=total_size, desc=os.path.basename(filename), unit="B", unit_scale=True
        )
        with open(filename, "wb") as fileobj:
            for chunk in response.iter_content(chunk_size=1024):
                fileobj.write(chunk)
                pbar.update(len(chunk))
        pbar.close()

json_file = "all_loaded_routes.json"

if not os.path.exists(json_file):
    cmd = " gunzip all_loaded_routes.json.gz"
    os.system(cmd)
json_obj = json.load(open(json_file))
all_nodes = []
all_ancestors = []
all_children = []
pbar = tqdm(json_obj)
for route in pbar:
    smiles_tree = extract_smiles_tree(route)
    nodes, ancestors, children = tree_to_ancestors_children(smiles_tree)
    for node in nodes:
        if node in children:
            all_nodes.append(node)
            all_ancestors.append(".".join(ancestors[node][::-1]))
            all_children.append(".".join(children[node]))
    # draw_smiles_tree(smiles_tree)
    pbar.set_description(f"Processed {len(all_nodes):,} nodes")

triplets = list(zip(all_nodes, all_ancestors, all_children))
print(f"Total triplets: {len(triplets)}")
triplets = list(set(triplets))
print(f"Unique triplets: {len(triplets)}")

node_to_triplets = defaultdict(list)
for triplet in triplets:
    node_to_triplets[triplet[0]].append(triplet)

# Step 3: Shuffle nodes (so we randomize assignment)
unique_nodes = list(node_to_triplets.keys())
random.shuffle(unique_nodes)
train_set, valid_set, test_set = [], [], []
n_total = len(unique_nodes)
n_train = int(n_total * 0.8)
n_valid = int(n_total * 0.1)

for i, node in enumerate(unique_nodes):
    if i < n_train:

        train_set.extend(node_to_triplets[node])
    elif i < n_train + n_valid:
        valid_set.extend(node_to_triplets[node])
    else:
        test_set.extend(node_to_triplets[node])


def get_lists(triplets):
    nodes, ancestors, children = zip(*triplets)
    return list(nodes), list(ancestors), list(children)


train_nodes, train_ancestors, train_children = get_lists(train_set)
valid_nodes, valid_ancestors, valid_children = get_lists(valid_set)
test_nodes, test_ancestors, test_children = get_lists(test_set)


# Step 4: Save the data
def save_data(nodes, ancestors, children, filename_src, filename_tgt, base_dir="PaRoutes"):
    os.makedirs(base_dir, exist_ok=True)
    with open(f'{base_dir}/{filename_src}', "w") as f:
        for node, ancestor in zip(nodes, ancestors):
            line = node
            if ancestor:
                line += f".{ancestor}"
            f.write(f"{line}\n")

    with open(f'{base_dir}/{filename_tgt}', "w") as f:
        for c in children:
            f.write(f"{c}\n")


save_data(train_nodes, train_ancestors, train_children, "train.src", "train.tgt")
save_data(valid_nodes, valid_ancestors, valid_children, "valid.src", "valid.tgt")
save_data(test_nodes, test_ancestors, test_children, "test.src", "test.tgt")
