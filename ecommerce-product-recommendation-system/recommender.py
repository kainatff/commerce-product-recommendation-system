import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

def load_dataset(file_path, row_limit=1005):
    dataset = pd.read_csv(file_path)  # Load dataset from file
    dataset = dataset.head(row_limit)  # Limit to the specified number of rows
    return dataset

def preprocess_data(dataset):
    filtered_data = dataset[['user_id', 'product_name', 'add_to_cart_order']].dropna()

    filtered_data.rename(columns={
        'user_id': 'CustomerID',
        'product_name': 'ProductName',  
        'add_to_cart_order': 'PurchaseOrder'
    }, inplace=True)
    
    return filtered_data

def build_customer_product_graph(data):
    graph = nx.Graph()
    for _, entry in data.iterrows():
        customer = entry['CustomerID']
        product = entry['ProductName']
        purchase_order = entry['PurchaseOrder']

        # Add nodes for customers and products
        if not graph.has_node(customer):
            graph.add_node(customer, type='customer')
        if not graph.has_node(product):
            graph.add_node(product, type='product')

        # Add an edge with weight representing purchase order
        graph.add_edge(customer, product, weight=purchase_order)

    return graph

def calculate_user_similarity(data):
    user_product_matrix = data.pivot_table(index='CustomerID', columns='ProductName', values='PurchaseOrder', fill_value=0)
    similarity_matrix = cosine_similarity(user_product_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_product_matrix.index, columns=user_product_matrix.index)
    return similarity_df

def recommend_products_based_on_similarity(graph, customer, similarity_df, data, top_n=5):
    if customer not in similarity_df.index:
        print(f"No similarity data available for Customer {customer}.")
        return []

    # Get the most similar customers
    similar_customers = similarity_df[customer].sort_values(ascending=False)[1:6]  # Exclude the customer themselves

    # Collect products purchased by similar customers
    similar_products = set()
    for similar_customer in similar_customers.index:
        purchased_products = data[data['CustomerID'] == similar_customer]['ProductName'].unique()
        similar_products.update(purchased_products)

    # Exclude products already purchased by the target customer
    purchased_by_target = set(data[data['CustomerID'] == customer]['ProductName'].unique())
    recommended_products = list(similar_products - purchased_by_target)

    return recommended_products[:top_n]

def show_recommendations(recommendation_data):
    for customer, recommendations in recommendation_data.items():
        if recommendations:
            print(f"Customer {customer}: Recommended -> {', '.join(map(str, recommendations))}")
        else:
            print(f"Customer {customer}: No product recommendations found.")

def visualize_graph(graph):
    plt.figure(figsize=(16, 12)) 
    
    node_colors = [
        '#FFA500' if graph.nodes[node].get('type') == 'customer' else '#87CEEB'
        for node in graph.nodes()
    ]
    
    # Define edge widths based on weights
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    edge_widths = [1 + weight * 0.1 for weight in edge_weights]  # Scale edge width

    # Customize node sizes
    node_sizes = [
        800 if graph.nodes[node].get('type') == 'customer' else 500
        for node in graph.nodes()
    ]

    # Node and edge labels
    pos = nx.spring_layout(graph, k=0.15, iterations=30) 
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="#A9A9A9", alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black", font_weight="bold")
    
    # Add a legend
    customer_patch = plt.Line2D([], [], color='#FFA500', marker='o', markersize=10, linestyle='None', label='Customer')
    product_patch = plt.Line2D([], [], color='#87CEEB', marker='o', markersize=10, linestyle='None', label='Product')
    plt.legend(handles=[customer_patch, product_patch], loc='upper left', fontsize=12)
    
    plt.title("Customer-Product Interaction Graph", fontsize=18, fontweight="bold", color="#333333")
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    # Hide axis
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "/content/ECommerce_consumer behaviour.csv"  # Update with your file path
    dataset = load_dataset(dataset_path, row_limit=200)
    cleaned_data = preprocess_data(dataset)
    
    # Build the graph
    customer_product_graph = build_customer_product_graph(cleaned_data)
    
    # Visualize the graph
    visualize_graph(customer_product_graph)

    # Calculate user similarity
    user_similarity_df = calculate_user_similarity(cleaned_data)
    
    # Generate recommendations based on user similarity
    recommendations = {}
    for customer in cleaned_data['CustomerID'].unique():
        recommendations[customer] = recommend_products_based_on_similarity(
            customer_product_graph, customer, user_similarity_df, cleaned_data, top_n=5
        )

    show_recommendations(recommendations)
